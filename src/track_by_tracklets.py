#!/usr/bin/env python3
"""
track_by_tracklets.py -- offline, tracklet-level target tracker.

A replacement for the per-frame identity logic in auto_track_target.py.
auto_track_target.py is NOT modified; this file only borrows its torso-crop
function, cosine helper and OSNet weights path so the embeddings match.

How it works (all offline, using the cached YOLO+ByteTrack output):

  1. Detect camera cuts. Split every ByteTrack ID at every cut, so an ID can
     never carry the target across a cut onto someone else.
       tracklet = one ByteTrack ID inside one shot

  2. Gather evidence on a regular grid of frames (default every 10th frame),
     cached to disk so re-runs are almost free:
       - faces: dlib HOG detection + 128-d encodings (same as before)
       - bodies: OSNet embedding of every box's torso

  3. Assign each face to at most ONE box (Hungarian). A face only counts for a
     box if it sits in that box's head region, so a big box that happens to
     contain someone else's face no longer inherits their identity.

  4. Label each tracklet from ALL its samples, not one lucky frame:
       FACE_TARGET  -- face votes say it is the target
       FACE_OTHER   -- face votes say it is someone else
       UNKNOWN      -- no usable face

  5. Build a body gallery from THIS clip: positives from FACE_TARGET
     tracklets, negatives from FACE_OTHER tracklets. An UNKNOWN tracklet is
     accepted as the target only if its body is clearly closer to the target
     gallery than to everyone else (margin threshold calibrated per clip
     from the face-labelled tracklets). If the gallery can't separate the
     target from others on a clip, body-only acceptance is switched off.

  6. One target per frame. Face evidence beats body evidence; a body-only
     tracklet that would put the target somewhere other than a
     face-confirmed box is rejected.

  7. Positions: target box centre; short gaps inside a shot are interpolated;
     longer gaps hold the nearest known position IN THE SAME SHOT and are
     marked lost; shots where the target is never seen are marked lost with a
     frame-centre fallback. Nothing is carried across a cut. Smoothing is
     zero-lag and resets at every cut.

Output format is unchanged: {"frame": {"pos": [x, y], "lost": bool}}.
A sidecar <output>_decisions.json records why every tracklet was accepted or
rejected.
"""

import argparse
import bisect
import json
import os
import time
from collections import defaultdict
import importlib.util

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------- tunables (each one explained) ----------------
SAMPLE_EVERY = 10          # evidence grid: every 10th frame (6/s at 60 fps)
FACE_MATCH = 0.48          # per-sample face distance that counts as a target vote
FACE_NONMATCH = 0.68       # per-sample face distance that counts as an "other" vote. Between
                           # FACE_MATCH and this is ambiguous (e.g. the target in profile and a
                           # similar-looking person both land at 0.52-0.65): face abstains and
                           # the clip's body gallery decides. Measured on two clips: target
                           # 0.20-0.45 frontal, clear strangers 0.75-0.91, nothing at 0.65-0.70.
FACE_STRONG = 0.42         # a single face sample this close is enough on its own
# InsightFace (ArcFace) thresholds, as distance = 1 - cosine similarity. Measured
# on Deen (120 frames): target 0.72-0.98 similarity, others 0.03-0.10.
INSIGHT_MATCH, INSIGHT_NONMATCH, INSIGHT_STRONG = 0.50, 0.70, 0.40   # sim >=0.50 / <0.30 / >=0.60
ARCFACE_GROUP_LINK = 0.40  # two seed groups are the same person if their faces match at this ArcFace similarity
ARCFACE_ONLY_MIN_MATCH = 3 # a tracklet ArcFace accepts but dlib does not must have at least this many face matches
                           # (seen: a 0.6 s stranger tracklet with 2 ArcFace matches, dlib ambiguous, body unlike target)
ARCFACE_GALLERY_CAP = 300  # max embeddings kept in target_reference_<name>_arcface.json
HEAD_REGION = 0.45         # face centre must be in the top 45% of the box
OVERLAP_IOU = 0.30         # body samples overlapping another box more than this are ignored
MIN_BODY_SAMPLES = 2       # body-only acceptance needs at least this many clean samples
BODY_MARGIN_FLOOR = 0.05   # minimum body margin, even if calibration says lower
MAX_INTERP_GAP = 45        # frames (~0.75 s): interpolate gaps up to this inside a shot
HOLD_BEFORE_LOST = 15      # frames: a longer gap holds position this long before "lost"
EMA_ALPHA = 0.3
CUT_THRESH = 0.35
SWAP_IOU = 0.5             # frame-to-frame box overlap below this is suspicious...
SWAP_ACCEL = 0.25          # ...and only a swap if the jump is this much of the box width beyond the previous step's motion
SOLE_MIN_BOX_H = 0.40     # sole-subject follow: the one person must be at least this tall
SOLE_MIN_FRAMES = 30       # ...for at least this many consecutive frames (0.5 s at 60 fps)
DUP_EVIDENCE_RATIO = 3.0   # duplicate target: reject the weaker if the stronger has 3x its face matches
ABSENT_MIN_BOX_H = 0.20    # people shorter than 20% of frame height can't be crop subjects; ignored
ABSENT_BODY_MARGIN = -0.05 # a no-face tracklet counts as "someone else" only below this body margin


def load_module(path):
    spec = importlib.util.spec_from_file_location("att_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def detect_cuts(clip, thresh):
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    dists, prev = [], None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        dists.append(0.0 if prev is None else float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)))
        prev = h
    cap.release()
    d = np.array(dists)
    cuts = []
    for i in range(1, len(d)):
        neigh = np.concatenate([d[max(0, i - 15):i], d[i + 1:i + 16]])
        local = float(np.median(neigh)) if neigh.size else 0.0
        if d[i] > thresh and d[i] > 3 * local + 0.05 and (not cuts or i - cuts[-1] >= 30):
            cuts.append(i)
    return cuts, fps


# --------------------------------------------------------------------------
def load_insightface(model):
    """InsightFace detector + ArcFace, on the Mac neural engine (CoreML) when
    available, else CPU. Returns None if it can't load (caller falls back)."""
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
    except Exception as e:
        print(f"InsightFace not available ({e}) -- falling back to dlib HOG")
        return None
    avail = ort.get_available_providers()
    tries = []
    if "CoreMLExecutionProvider" in avail:
        tries.append(["CoreMLExecutionProvider", "CPUExecutionProvider"])
    tries.append(["CPUExecutionProvider"])
    for provs in tries:
        try:
            app = FaceAnalysis(name=model, providers=provs, allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"InsightFace {model} loaded ({provs[0]})")
            return app
        except Exception as e:
            print(f"InsightFace with {provs[0]} failed: {e}")
    return None


def gather_evidence(clip, frames_needed, tracks, att, face_refs, use_reid, cache_path,
                    backend="dlib", insight_app=None):
    """faces[f] = [(loc(t,r,b,l), enc, dlib_enc)], body[(f, tid)] = embedding.
    backend "dlib": HOG detection, enc = dlib 128-d (dlib_enc is the same).
    backend "insight": InsightFace detection, enc = ArcFace 512-d, dlib_enc =
    dlib encoding of the SAME face box (None if dlib unavailable), used to seed
    the ArcFace gallery from the existing dlib reference.
    Cached in an .npz; frames already cached are not recomputed."""
    faces, body = {}, {}
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cached_backend = str(z["backend"]) if "backend" in z.files else "dlib"
        if cached_backend != backend:
            raise SystemExit(f"Evidence cache {cache_path} was made with the {cached_backend} backend, "
                             f"not {backend} -- use a different --evidence-cache.")
        dl = z["face_dlib"] if "face_dlib" in z.files else None
        for i, (f, loc, enc) in enumerate(zip(z["face_frames"], z["face_locs"], z["face_encs"])):
            if backend == "dlib":
                aux = enc
            else:
                aux = dl[i] if dl is not None and not np.isnan(dl[i]).any() else None
            faces.setdefault(int(f), []).append((tuple(int(v) for v in loc), enc, aux))
        for f in z["face_done"]:
            faces.setdefault(int(f), [])
        for f, tid, e in zip(z["body_frames"], z["body_tids"], z["body_embs"]):
            body[(int(f), int(tid))] = e
        print(f"Loaded evidence cache: {len(z['face_done'])} frames.")
    todo = sorted(f for f in frames_needed if f not in faces)
    if not todo:
        return faces, body

    try:
        import face_recognition
    except Exception:
        face_recognition = None
        if backend == "dlib":
            raise
    extractor = None
    if use_reid:
        from torchreid.reid.utils import FeatureExtractor
        extractor = FeatureExtractor(model_name="osnet_x1_0", model_path=att.REID_WEIGHTS, device="cpu")

    print(f"Computing face + body evidence on {len(todo)} frames (cached afterwards)...")
    cap = cv2.VideoCapture(clip)
    wi, f, t0 = 0, 0, time.time()
    while wi < len(todo):
        if not cap.grab():
            break
        if f != todo[wi]:
            f += 1
            continue
        ok, frame = cap.retrieve()
        wi += 1
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if backend == "insight":
                fh, fw_ = frame.shape[:2]
                got = insight_app.get(frame)
                locs, embs = [], []
                for fc in got:
                    x1, y1, x2, y2 = [int(round(v)) for v in fc.bbox]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(fw_, x2), min(fh, y2)
                    if x2 - x1 < 8 or y2 - y1 < 8 or getattr(fc, "normed_embedding", None) is None:
                        continue
                    locs.append((y1, x2, y2, x1))
                    embs.append(np.asarray(fc.normed_embedding, dtype=np.float32))
                dencs = [None] * len(locs)
                if face_recognition is not None and locs and face_refs:
                    try:
                        got_d = face_recognition.face_encodings(rgb, known_face_locations=locs)
                        if len(got_d) == len(locs):   # never let a dlib hiccup drop InsightFace faces
                            dencs = [np.asarray(e) for e in got_d]
                    except Exception:
                        pass
                faces[f] = [(l, e, de) for l, e, de in zip(locs, embs, dencs)]
            else:
                try:
                    locs = face_recognition.face_locations(rgb, model="hog")
                except Exception:
                    locs = []
                encs = face_recognition.face_encodings(rgb, known_face_locations=locs) if locs else []
                faces[f] = [(tuple(int(v) for v in l), np.asarray(e), np.asarray(e)) for l, e in zip(locs, encs)]
            if extractor is not None:
                crops, tids = [], []
                for b in tracks.get(f, []):
                    c = att.crop_box_torso(frame, b)
                    if c is not None and c.shape[0] >= 8 and c.shape[1] >= 8:
                        crops.append(c)
                        tids.append(int(b[0]))
                if crops:
                    feats = extractor(crops)
                    for tid, ft in zip(tids, feats):
                        body[(f, tid)] = ft.cpu().numpy().flatten()
        if wi % 50 == 0:
            el = time.time() - t0
            print(f"  {wi}/{len(todo)}  ({el:.0f}s elapsed, ~{el / wi * (len(todo) - wi):.0f}s left)")
        f += 1
    cap.release()

    if cache_path:
        ff, fl, fe, fd = [], [], [], []
        for fr, lst in faces.items():
            for loc, enc, aux in lst:
                ff.append(fr); fl.append(loc); fe.append(enc)
                if backend == "insight":
                    fd.append(aux if aux is not None else np.full(128, np.nan))
        bk = list(body.keys())
        dim = 512 if backend == "insight" else 128
        np.savez_compressed(
            cache_path,
            backend=np.array(backend),
            face_dlib=np.array(fd, dtype=np.float64).reshape(len(fd), 128) if fd else np.zeros((0, 128)),  # insight only
            face_done=np.array(sorted(faces), dtype=np.int64),
            face_frames=np.array(ff, dtype=np.int64),
            face_locs=np.array(fl, dtype=np.int64).reshape(-1, 4),
            face_encs=np.array(fe, dtype=np.float64).reshape(len(fe), -1) if fe else np.zeros((0, dim)),
            body_frames=np.array([k[0] for k in bk], dtype=np.int64),
            body_tids=np.array([k[1] for k in bk], dtype=np.int64),
            body_embs=np.array([body[k] for k in bk], dtype=np.float32) if bk else np.zeros((0, 512), np.float32),
        )
        print(f"Saved evidence cache to {cache_path}")
    return faces, body


def assign_faces(face_list, boxes):
    """Hungarian assignment of faces to boxes. A face can only go to a box
    whose head region contains it; cost = distance from the box's expected
    head point, normalised by box width. Returns {box_index: face_index}."""
    if not face_list or not boxes:
        return {}
    BIG = 1e6
    C = np.full((len(face_list), len(boxes)), BIG)
    for i, entry in enumerate(face_list):
        t, r, bt, l = entry[0]
        cx, cy, fw = (l + r) / 2, (t + bt) / 2, (r - l)
        for j, b in enumerate(boxes):
            x1, y1, x2, y2 = b[1:5]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            if not (x1 <= cx <= x2 and y1 <= cy <= y1 + HEAD_REGION * h):
                continue
            if fw > 0.95 * w or fw < 0.06 * w:     # face size implausible for this box
                continue
            # distance from the box's expected head point, in face widths,
            # plus a penalty when the box is much wider/narrower than a
            # single person with this face size (~3.5 face widths), so a
            # loose box around two people loses to the person's own box
            hx, hy = (x1 + x2) / 2, y1 + 0.15 * h
            dist = (((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5) / max(fw, 1)
            C[i, j] = dist + abs(np.log(w / (3.5 * max(fw, 1))))
    rows, cols = linear_sum_assignment(C)
    return {int(c): int(r) for r, c in zip(rows, cols) if C[r, c] < BIG}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--tracks-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cuts-json", default=None, help="reuse cuts.json from diagnose_tracking.py")
    ap.add_argument("--evidence-cache", default=None, help="default: <output>_evidence.npz")
    ap.add_argument("--tracker", default=os.path.expanduser("~/Downloads/auto_track_target.py"))
    ap.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    ap.add_argument("--no-body", action="store_true", help="face evidence only")
    ap.add_argument("--face-backend", choices=["dlib", "insight"], default="dlib",
                    help="dlib = HOG + dlib (original). insight = InsightFace detection + ArcFace "
                         "identity (17x faster on Apple Silicon, finds turned faces); falls back to dlib.")
    ap.add_argument("--insight-model", default="buffalo_l")
    args = ap.parse_args()

    if os.path.exists(args.output):
        raise SystemExit(f"Refusing to overwrite existing output {args.output} -- pick a new name.")

    att = load_module(args.tracker)
    with open(args.reference) as f:
        ref = json.load(f)
    face_refs = [np.array(e) for e in ref.get("face_encodings", [])]
    if not face_refs:
        raise SystemExit("Reference has no face encodings; this tracker needs at least one.")
    if os.path.exists(args.tracks_cache):
        with open(args.tracks_cache) as f:
            tracks = {int(k): v for k, v in json.load(f).items()}
    else:
        # no cache yet (e.g. a fresh clip inside run_pipeline.py): run
        # YOLO+ByteTrack with the exact same function the old tracker uses,
        # which also writes the cache for future re-runs
        print(f"No YOLO cache at {args.tracks_cache} -- running YOLO+ByteTrack once...")
        tracks = att.run_yolo_tracking(args.clip, args.tracks_cache)

    cap = cv2.VideoCapture(args.clip)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_frames = max(n_video, max(tracks) + 1 if tracks else 0)

    if args.cuts_json and os.path.exists(args.cuts_json):
        cj = json.load(open(args.cuts_json))
        cuts, fps = [int(c) for c in cj["cuts"]], float(cj["fps"])
        print(f"Using {len(cuts)} cuts from {args.cuts_json}")
    else:
        print("Detecting camera cuts...")
        cuts, fps = detect_cuts(args.clip, CUT_THRESH)
    bounds = [0] + cuts + [n_frames]
    shots = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
    shot_of = lambda fr: bisect.bisect_right(cuts, fr)
    print(f"{len(cuts)} cuts -> {len(shots)} shots")

    # ---- tracklets (same naming as diagnose_tracking.py when cuts match) ----
    tl = defaultdict(lambda: {"frames": [], "boxes": {}})
    for fr in sorted(tracks):
        for b in tracks[fr]:
            k = (int(b[0]), shot_of(fr))
            tl[k]["frames"].append(fr)
            tl[k]["boxes"][fr] = [float(v) for v in b[1:5]]
    # ---- split tracklets at ByteTrack ID swaps inside a shot ----
    # A swap shows up as a sudden jump of the box between consecutive frames
    # that doesn't continue the previous motion. Steady fast movement (large
    # but consistent per-frame steps) is NOT split.
    def centre(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    swaps = 0
    for k in list(tl):
        fr_list, boxes = tl[k]["frames"], tl[k]["boxes"]
        segs, cur = [], [fr_list[0]]
        prev_step = None
        for a, b in zip(fr_list, fr_list[1:]):
            ba, bb = boxes[a], boxes[b]
            ca, cb = centre(ba), centre(bb)
            step = (cb[0] - ca[0], cb[1] - ca[1])
            jump = False
            if b == a + 1 and iou(ba, bb) < SWAP_IOU:
                ps = prev_step or (0.0, 0.0)
                accel = ((step[0] - ps[0]) ** 2 + (step[1] - ps[1]) ** 2) ** 0.5
                jump = accel > SWAP_ACCEL * max(ba[2] - ba[0], 1)
            if jump:
                segs.append(cur)
                cur = [b]
                prev_step = None
                swaps += 1
            else:
                cur.append(b)
                prev_step = step if b == a + 1 else None
        segs.append(cur)
        if len(segs) > 1:
            base = tl.pop(k)
            for i, seg in enumerate(segs):
                tl[(k[0], k[1], i)] = {"frames": seg, "boxes": {f: base["boxes"][f] for f in seg}}
    if swaps:
        print(f"Split {swaps} ByteTrack ID swap(s) inside shots")

    keys = sorted(tl, key=lambda k: (tl[k]["frames"][0], k[0]))
    T = []
    for i, k in enumerate(keys):
        T.append({"name": f"T{i:03d}", "tid": k[0], "shot": k[1], "seg": k[2] if len(k) > 2 else 0,
                  "frames": tl[k]["frames"], "boxes": tl[k]["boxes"]})
    # (tid, frame) -> tracklet, so swapped segments of one ID resolve correctly
    tl_at = {}
    for t in T:
        for fr in t["frames"]:
            tl_at[(t["tid"], fr)] = t

    # ---- evidence frames: global grid + middle frame of short tracklets ----
    need = set(range(0, n_frames, args.sample_every))
    for t in T:
        if len(t["frames"]) >= 6 and not any(fr in need for fr in t["frames"]):
            need.add(t["frames"][len(t["frames"]) // 2])
    need = sorted(fr for fr in need if fr in tracks and tracks[fr])
    global FACE_MATCH, FACE_NONMATCH, FACE_STRONG
    backend, insight_app = args.face_backend, None
    if backend == "insight":
        insight_app = load_insightface(args.insight_model)
        if insight_app is None:
            backend = "dlib"
    suffix = "_evidence_insight.npz" if backend == "insight" else "_evidence.npz"
    cache_path = args.evidence_cache or os.path.splitext(args.output)[0] + suffix
    faces, body = gather_evidence(args.clip, need, tracks, att, face_refs, not args.no_body, cache_path,
                                  backend=backend, insight_app=insight_app)

    # ---- identity per face sample ----
    refs_arr = np.array(face_refs)

    def dlib_dist(denc):
        return float(np.min(np.linalg.norm(refs_arr - denc, axis=1)))

    def state_of(d, m_thr, o_thr, s_thr):
        nm = sum(1 for x in d if x <= m_thr)
        no = sum(1 for x in d if x > o_thr)
        if nm >= 1 and nm >= 2 * no and (nm >= 2 or min(d) <= s_thr):
            return "FACE_TARGET", nm, no
        if no >= 1 and nm == 0:
            return "FACE_OTHER", nm, no
        return ("FACE_AMBIG" if d else "UNKNOWN"), nm, no

    # ---- per-sample face assignment (one face per box) + body samples ----
    for t in T:
        t["face_ents"], t["body"] = [], []
    for fr in need:
        boxes = tracks.get(fr, [])
        fl = faces.get(fr, [])
        amap = assign_faces(fl, boxes)
        for j, b in enumerate(boxes):
            t = tl_at.get((int(b[0]), fr))
            if t is None:
                continue
            if j in amap:
                t["face_ents"].append(fl[amap[j]])
            e = body.get((fr, int(b[0])))
            if e is not None:
                others = [ob for k2, ob in enumerate(boxes) if k2 != j]
                clean = all(iou(b[1:5], ob[1:5]) <= OVERLAP_IOU for ob in others)
                # a sample whose face was assigned to this exact box is
                # identity-verified, so it may feed the galleries even if
                # the box overlaps someone
                t["body"].append((e, clean, j in amap))

    # ---- dlib verdict per tracklet (the proven rules; used by both backends) ----
    for t in T:
        t["dlib_d"] = [dlib_dist(e[2]) for e in t["face_ents"] if e[2] is not None]
        t["dlib_state"] = state_of(t["dlib_d"], FACE_MATCH, FACE_NONMATCH, FACE_STRONG)[0]

    face_mode = "dlib HOG"
    use_arc = {}
    if backend == "insight":
        # ArcFace target gallery, built safely:
        #  1. seeds come only from tracklets the dlib rules ALREADY accept as the
        #     target (tracklet vote, not single faces -- one lucky dlib match on a
        #     stranger used to poison the gallery);
        #  2. seed groups (one per tracklet, plus the saved gallery) must look like
        #     each other; only the largest mutually-consistent group is kept;
        #  3. a tracklet is never scored against its own seeds (leave-one-out), so
        #     it cannot confirm itself;
        #  4. ArcFace and dlib must not contradict each other (see below).
        side = os.path.splitext(args.reference)[0] + "_arcface_v2.json"
        saved = []
        if os.path.exists(side):
            try:
                saved = [np.array(e, dtype=np.float32) for e in json.load(open(side)).get("arcface_embeddings", [])]
            except Exception as e:
                print(f"warning: could not read {side}: {e}")
        groups = {}
        for t in T:
            if t["dlib_state"] != "FACE_TARGET":
                continue
            s = [np.asarray(e[1], dtype=np.float32) for e in t["face_ents"]
                 if e[2] is not None and dlib_dist(e[2]) <= FACE_MATCH]
            if s:
                groups[t["name"]] = np.stack(s)
        if saved:
            groups["saved"] = np.stack(saved)
        names = list(groups)

        def gsim(a, b):
            # median over a's faces of the best similarity to b's faces
            return float(np.median(np.max(groups[a] @ groups[b].T, axis=1)))

        link = {n: set() for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if 0.5 * (gsim(a, b) + gsim(b, a)) >= ARCFACE_GROUP_LINK:
                    link[a].add(b)
                    link[b].add(a)
        comps, seen = [], set()
        for n in names:
            if n in seen:
                continue
            stack, comp = [n], set()
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(link[x] - comp)
            seen |= comp
            comps.append(comp)
        keep_groups = max(comps, key=lambda c: sum(len(groups[n]) for n in c)) if comps else set()
        dropped = [n for n in names if n not in keep_groups]
        print(f"ArcFace seed groups: {len(names)} ({', '.join(f'{n}:{len(groups[n])}' for n in names)})")
        if dropped:
            print(f"  dropped (faces don't match the rest of the target's faces): {', '.join(dropped)}")
        if keep_groups:
            face_mode = "InsightFace + ArcFace (leave-one-out, dlib cross-check)"
            kept = {n: groups[n] for n in keep_groups}
            for t in T:
                rest = [g for n, g in kept.items() if n != t["name"]]
                if rest and t["face_ents"]:
                    G = np.concatenate(rest)
                    use_arc[t["name"]] = G
            # remember the consistent seeds for future clips (diverse ones only)
            fresh = [e for n in keep_groups if n != "saved" for e in groups[n]]
            if fresh and len(keep_groups) >= 2:
                keep = list(saved)
                for e in fresh:
                    if len(keep) >= ARCFACE_GALLERY_CAP:
                        break
                    if not keep or float(np.max(np.stack(keep) @ e)) < 0.97:
                        keep.append(e)
                if len(keep) > len(saved):
                    tmp = side + ".tmp"
                    with open(tmp, "w") as fh:
                        json.dump({"source_reference": os.path.basename(args.reference),
                                   "arcface_embeddings": [k.tolist() for k in keep]}, fh)
                    os.replace(tmp, side)
                    print(f"Saved {len(keep) - len(saved)} new ArcFace embeddings to {side} ({len(keep)} total)")
        else:
            face_mode = "InsightFace detection + dlib identity (no ArcFace gallery yet)"
    print(f"Face identity: {face_mode}")

    overruled = weak_arc = 0
    for t in T:
        if t["name"] in use_arc:
            G = use_arc[t["name"]]
            d = [1.0 - float(np.max(G @ np.asarray(e[1], dtype=np.float32))) for e in t["face_ents"]]
            st, nm, no = state_of(d, INSIGHT_MATCH, INSIGHT_NONMATCH, INSIGHT_STRONG)
            # cross-check: neither model may overrule a clear verdict of the other
            if (st == "FACE_TARGET" and t["dlib_state"] == "FACE_OTHER") or \
               (st == "FACE_OTHER" and t["dlib_state"] == "FACE_TARGET"):
                st = "FACE_AMBIG"
                overruled += 1
            # ArcFace alone (dlib not confirming) needs more than a glimpse
            elif st == "FACE_TARGET" and t["dlib_state"] != "FACE_TARGET" and nm < ARCFACE_ONLY_MIN_MATCH:
                st = "FACE_AMBIG"
                weak_arc += 1
            t["face_id"] = "arcface"
        else:
            d = t["dlib_d"]
            st, nm, no = state_of(d, FACE_MATCH, FACE_NONMATCH, FACE_STRONG)
            t["face_id"] = "dlib"
        t["face_d"] = d
        t["n_face"], t["n_match"], t["n_other"] = len(d), nm, no
        t["face_med"] = float(np.median(d)) if d else None
        t["face_state"] = st
    if backend == "insight":
        print(f"ArcFace/dlib disagreements set to ambiguous (never followed): {overruled}")
        print(f"ArcFace-only matches too brief to trust (< {ARCFACE_ONLY_MIN_MATCH} faces), set to ambiguous: {weak_arc}")


    # ---- clip-specific body gallery + per-clip calibration ----
    pos = [(t["name"], e) for t in T if t["face_state"] == "FACE_TARGET"
           for e, clean, verified in t["body"] if clean or verified]
    # negatives: everyone whose face was seen and did NOT confirm the target --
    # clear strangers and ambiguous faces alike, so a similar-bodied person
    # (e.g. another shirtless man) is in the gallery when he turns away
    neg = [(t["name"], e) for t in T if t["face_state"] in ("FACE_OTHER", "FACE_AMBIG")
           for e, clean, verified in t["body"] if clean or verified]

    def margin(t):
        """median over the tracklet's clean samples of
        (best sim to target gallery) - (best sim to others), excluding itself."""
        ms = []
        for e, clean, verified in t["body"]:
            if not (clean or verified):
                continue
            sp = [cos(e, g) for n, g in pos if n != t["name"]]
            sn = [cos(e, g) for n, g in neg if n != t["name"]]
            if sp and sn:
                ms.append(max(sp) - max(sn))
        return (float(np.median(ms)), len(ms)) if ms else (None, 0)

    for t in T:
        t["body_margin"], t["n_body"] = margin(t)

    tgt_m = [t["body_margin"] for t in T if t["face_state"] == "FACE_TARGET" and t["body_margin"] is not None]
    oth_m = [t["body_margin"] for t in T if t["face_state"] == "FACE_OTHER" and t["body_margin"] is not None]
    body_enabled = not args.no_body and len(tgt_m) >= 2 and len(oth_m) >= 3
    body_thr = None
    if body_enabled:
        hi = float(np.percentile(oth_m, 95)) if len(oth_m) >= 10 else max(oth_m)
        body_thr = max(BODY_MARGIN_FLOOR, hi + 0.02)
        if body_thr >= float(np.median(tgt_m)):
            body_enabled = False

    for t in T:
        if t["face_state"] == "FACE_TARGET":
            t["decision"], t["score"] = "TARGET_FACE", 2.0 + (0.6 - t["face_med"])
        elif (t["face_state"] == "UNKNOWN" and body_enabled and t["body_margin"] is not None
              and t["n_body"] >= MIN_BODY_SAMPLES and t["body_margin"] >= body_thr):
            t["decision"], t["score"] = "TARGET_BODY", 1.0 + t["body_margin"]
        else:
            t["decision"], t["score"] = "NOT_TARGET", 0.0

    # ---- one target per frame; body-only tracklets that contradict face ones are rejected ----
    face_box = defaultdict(list)
    for t in T:
        if t["decision"] == "TARGET_FACE":
            for fr, bx in t["boxes"].items():
                face_box[fr].append(bx)
    for t in T:
        if t["decision"] != "TARGET_BODY":
            continue
        clash = sum(1 for fr, bx in t["boxes"].items()
                    if face_box.get(fr) and all(iou(bx, fb) < OVERLAP_IOU for fb in face_box[fr]))
        if clash > 0.1 * len(t["frames"]):
            t["decision"], t["score"] = "REJECTED_CONTRADICTS_FACE", 0.0

    # ---- the target can't be in two places: a face-confirmed tracklet that
    # co-occurs with a much stronger one, in a separate place, is a picture of
    # the target (phone screen, monitor, poster) -- reject it ----
    face_t = [t for t in T if t["decision"] == "TARGET_FACE"]
    dup_rejected = 0
    for a in face_t:
        for b in face_t:
            if a is b or a["decision"] != "TARGET_FACE" or b["decision"] != "TARGET_FACE":
                continue
            common = [fr for fr in a["boxes"] if fr in b["boxes"]]
            if not common:
                continue
            apart = sum(1 for fr in common if iou(a["boxes"][fr], b["boxes"][fr]) < OVERLAP_IOU)
            weak, strong = (a, b) if a["n_match"] < b["n_match"] else (b, a)
            if (apart >= 0.5 * min(len(a["frames"]), len(b["frames"]))
                    and strong["n_match"] >= DUP_EVIDENCE_RATIO * max(weak["n_match"], 1)):
                weak["decision"], weak["score"] = "REJECTED_DUPLICATE", 0.0
                dup_rejected += 1

    chosen = {}
    conflicts = 0
    for t in T:
        if not t["decision"].startswith("TARGET"):
            continue
        for fr, bx in t["boxes"].items():
            cur = chosen.get(fr)
            if cur is not None and iou(cur[1], bx) < OVERLAP_IOU:
                conflicts += 1
            if cur is None or t["score"] > cur[0]:
                chosen[fr] = (t["score"], bx, t["decision"])

    # ---- positions per shot ----
    raw, lost, kind = {}, {}, {}
    for s, (a, b) in enumerate(shots):
        obs = sorted(fr for fr in range(a, b + 1) if fr in chosen)
        if not obs:
            for fr in range(a, b + 1):
                raw[fr], lost[fr], kind[fr] = (W / 2, H / 2), True, "no_target_in_shot"
            continue
        ctr = {fr: ((chosen[fr][1][0] + chosen[fr][1][2]) / 2, (chosen[fr][1][1] + chosen[fr][1][3]) / 2) for fr in obs}
        for fr in range(a, b + 1):
            if fr in ctr:
                raw[fr], lost[fr], kind[fr] = ctr[fr], False, chosen[fr][2]
                continue
            i = bisect.bisect_left(obs, fr)
            prv = obs[i - 1] if i > 0 else None
            nxt = obs[i] if i < len(obs) else None
            if prv is not None and nxt is not None and nxt - prv <= MAX_INTERP_GAP:
                w = (fr - prv) / (nxt - prv)
                p, q = ctr[prv], ctr[nxt]
                raw[fr], lost[fr], kind[fr] = (p[0] + w * (q[0] - p[0]), p[1] + w * (q[1] - p[1])), False, "interpolated"
            else:
                near = min([x for x in (prv, nxt) if x is not None], key=lambda x: abs(x - fr))
                raw[fr] = ctr[near]
                held = abs(near - fr) <= HOLD_BEFORE_LOST
                lost[fr], kind[fr] = (not held), ("short_hold" if held else "lost_in_shot")

    # ---- absent vs uncertain for every lost frame ----
    # "absent"    = every sizeable person on screen is positively accounted
    #               for as someone else (face says other, or body clearly
    #               closer to others), or nobody is on screen.
    # "uncertain" = at least one sizeable person is unidentified, so the
    #               target might be on screen but unrecognised.
    def explained_as_other(t):
        if t["decision"].startswith("TARGET") or t["decision"] == "REJECTED_CONTRADICTS_FACE":
            return False
        if t["face_state"] == "FACE_OTHER":
            return True
        return (t["body_margin"] is not None and t["n_body"] >= 2
                and t["body_margin"] <= ABSENT_BODY_MARGIN)

    on_screen = defaultdict(list)
    for t in T:
        for fr, bx in t["boxes"].items():
            on_screen[fr].append((t, bx))
    state = {}
    for fr in range(n_frames):
        k = kind[fr]
        if k in ("TARGET_FACE", "TARGET_BODY"):
            state[fr] = "tracked"
        elif k == "interpolated":
            state[fr] = "interpolated"
        elif k == "short_hold":
            state[fr] = "hold"
        else:
            sizeable = [t for t, bx in on_screen.get(fr, []) if (bx[3] - bx[1]) >= ABSENT_MIN_BOX_H * H]
            if not sizeable:
                # nobody on screen (black screen, test card, empty room): not
                # "the camera is on someone else", so it's reported separately
                state[fr] = "empty"
            else:
                unexplained = [t for t in sizeable if not explained_as_other(t)]
                state[fr] = "uncertain" if unexplained else "absent"

    # ---- sole subject: in "uncertain" stretches where exactly one sizeable
    # person is on screen and nothing says it's someone else, frame that person
    # tightly. Identity is NOT claimed: state "sole_subject" is reported apart
    # from "tracked" so clip selection can weigh it separately. ----
    sole = {}
    for fr in range(n_frames):
        if state[fr] != "uncertain":
            continue
        big = [(t, bx) for t, bx in on_screen.get(fr, []) if (bx[3] - bx[1]) >= ABSENT_MIN_BOX_H * H]
        # only when no face was seen at all on that person: a visible face that
        # doesn't match is a reason NOT to frame them as the subject
        if len(big) == 1 and (big[0][1][3] - big[0][1][1]) >= SOLE_MIN_BOX_H * H \
                and big[0][0]["face_state"] == "UNKNOWN" and not explained_as_other(big[0][0]):
            sole[fr] = big[0]
    run = []
    for fr in list(range(n_frames)) + [n_frames]:
        if fr in sole and (not run or (fr == run[-1] + 1 and sole[fr][0] is sole[run[-1]][0])):
            run.append(fr)
            continue
        if len(run) >= SOLE_MIN_FRAMES:
            for r in run:
                bx = sole[r][1]
                raw[r] = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
                lost[r], state[r] = False, "sole_subject"
        run = [fr] if fr in sole else []

    # zero-lag smoothing, reset at every cut
    smooth = {}
    for a, b in shots:
        seq = np.array([raw[fr] for fr in range(a, b + 1)], dtype=float)
        fwd = seq.copy()
        for i in range(1, len(seq)):
            fwd[i] = EMA_ALPHA * seq[i] + (1 - EMA_ALPHA) * fwd[i - 1]
        bwd = fwd.copy()
        for i in range(len(seq) - 2, -1, -1):
            bwd[i] = EMA_ALPHA * fwd[i] + (1 - EMA_ALPHA) * bwd[i + 1]
        for i, fr in enumerate(range(a, b + 1)):
            smooth[fr] = [round(float(bwd[i][0]), 1), round(float(bwd[i][1]), 1)]

    # "pos"/"lost" are the original interface; "shot" and "state" are extra
    # keys older renderers simply ignore.
    out = {str(fr): {"pos": smooth[fr], "lost": bool(lost[fr]), "shot": shot_of(fr),
                     "state": state[fr]} for fr in range(n_frames)}
    with open(args.output, "w") as f:
        json.dump(out, f)

    dec_path = os.path.splitext(args.output)[0] + "_decisions.json"
    with open(dec_path, "w") as f:
        sc = defaultdict(int)
        for fr in range(n_frames):
            sc[state[fr]] += 1
        presence = {
            "tracked_frac": round((sc["tracked"] + sc["interpolated"] + sc["hold"]) / n_frames, 4),
            "sole_subject_frac": round(sc["sole_subject"] / n_frames, 4),
            "empty_frac": round(sc["empty"] / n_frames, 4),
            "uncertain_frac": round(sc["uncertain"] / n_frames, 4),
            "absent_frac": round(sc["absent"] / n_frames, 4),
            "per_shot": [{"shot": i, "start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                          **{k: sum(1 for fr in range(a, b + 1) if state[fr] == k)
                             for k in ("tracked", "interpolated", "hold", "sole_subject", "uncertain", "absent", "empty")}}
                         for i, (a, b) in enumerate(shots)],
        }
        json.dump({"presence": presence, "cuts": cuts, "fps": fps, "body_enabled": body_enabled, "body_threshold": body_thr,
                   "tracklets": [{k: t[k] for k in ("name", "tid", "shot", "decision", "face_state", "n_face",
                                                     "n_match", "n_other", "face_med", "body_margin", "n_body", "dlib_state", "face_id")}
                                 | {"start_s": round(t["frames"][0] / fps, 2), "end_s": round(t["frames"][-1] / fps, 2),
                                    "frames": len(t["frames"])}
                                 for t in T]}, f, indent=1)

    # ---- summary ----
    cnt = defaultdict(int)
    for fr in range(n_frames):
        cnt[kind[fr]] += 1
    print("\n" + "=" * 64)
    print("TRACKLET TRACKER SUMMARY")
    print("=" * 64)
    print(f"tracklets: {len(T)}   TARGET_FACE: {sum(t['decision'] == 'TARGET_FACE' for t in T)}   "
          f"TARGET_BODY: {sum(t['decision'] == 'TARGET_BODY' for t in T)}   "
          f"rejected (contradicts face): {sum(t['decision'] == 'REJECTED_CONTRADICTS_FACE' for t in T)}")
    if tgt_m and oth_m:
        print(f"body margin, face-confirmed target tracklets: min {min(tgt_m):.3f} median {np.median(tgt_m):.3f}")
        print(f"body margin, face-confirmed other tracklets:  median {np.median(oth_m):.3f} max {max(oth_m):.3f}")
    print(f"body-only acceptance: {'ON, threshold %.3f' % body_thr if body_enabled else 'OFF (gallery cannot separate target on this clip)'}")
    print(f"frames where two non-overlapping target boxes competed: {conflicts}")
    print()
    for k in ["TARGET_FACE", "TARGET_BODY", "interpolated", "short_hold", "lost_in_shot", "no_target_in_shot"]:
        print(f"  {k:18s} {cnt[k]:5d} frames  ({cnt[k] / fps:5.1f}s)")
    n_lost = sum(lost.values())
    print(f"  {'LOST total':18s} {n_lost:5d} frames  ({100 * n_lost / n_frames:.1f}%)")
    n_abs = sum(1 for v in state.values() if v == "absent")
    n_unc = sum(1 for v in state.values() if v == "uncertain")
    n_sole = sum(1 for v in state.values() if v == "sole_subject")
    print(f"  {'SOLE SUBJECT':18s} {n_sole:5d} frames  ({n_sole / fps:5.1f}s)  framed tightly: only one unidentified person on screen")
    print(f"  duplicate target tracklets rejected (target's face on a screen/photo): {dup_rejected}")
    n_empty = sum(1 for v in state.values() if v == "empty")
    print(f"     of which EMPTY     {n_empty:5d} frames  ({n_empty / fps:5.1f}s)  nobody on screen (black screen, test card, empty room)")
    print(f"     of which ABSENT    {n_abs:5d} frames  ({n_abs / fps:5.1f}s)  everyone on screen identified as someone else")
    print(f"     of which UNCERTAIN {n_unc:5d} frames  ({n_unc / fps:5.1f}s)  an unidentified person could be the target")
    print()
    for s, (a, b) in enumerate(shots):
        seen = sum(1 for fr in range(a, b + 1) if not lost[fr])
        print(f"  shot {s}: {a / fps:6.2f}s-{b / fps:6.2f}s  target tracked {100 * seen / (b - a + 1):5.1f}%")
    print(f"\nSaved {args.output}")
    print(f"Saved {dec_path}")


if __name__ == "__main__":
    main()
