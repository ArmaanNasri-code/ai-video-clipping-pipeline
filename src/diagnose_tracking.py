#!/usr/bin/env python3
"""
diagnose_tracking.py -- READ-ONLY diagnostics for auto_track_target.py

Does not modify the tracker, the YOLO cache, or any earlier output.
Everything it writes goes into --out-dir.

What it measures:
  1. Shot boundaries (camera cuts), from frame-to-frame color histogram change.
  2. Tracklets: each ByteTrack ID split at every detected cut, so one
     tracklet = one ID within one shot.
  3. Per-tracklet identity evidence, sampled over several frames:
       - face found by the SAME HOG call the tracker uses (full frame)
       - face found only by an upsampled HOG on the box's upper half
         (tells you whether a better face detector setting would help)
       - face distance to the reference (min over faces inside the box)
       - "first-face" distance: what identity_cost() actually uses
       - how many YOLO boxes contain the same face (association ambiguity)
       - OSNet body similarity to the reference centroid
  4. For every lost interval in an existing tracker output: which tracklets
     were on screen, and why the target wasn't recovered.
  5. Frames where the tracker claims "not lost" but is pointing at a box
     whose face evidence says it is NOT the target (suspected wrong person),
     or at no box at all.
  6. Position jumps and out-of-frame positions in the output.
  7. Contact sheets with one thumbnail per tracklet, plus a labels CSV you
     can fill in once to get real correct/wrong-person accuracy later.

Usage:
  python3 diagnose_tracking.py \
      --clip CLIP.mp4 --reference target_reference_kai.json \
      --tracks-cache CLIP_yolo_tracks.json \
      --track-output CLIP_track.json \
      --out-dir diag_out
"""

import argparse
import bisect
import csv
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_tracker_module(path):
    """Import auto_track_target.py so we use its exact constants, torso crop
    and weights path instead of a copy that could drift."""
    spec = importlib.util.spec_from_file_location("att_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt_t(frame, fps):
    return f"{frame / fps:6.2f}s"


def detect_cuts(clip, thresh, min_gap_frames):
    """Pass 1: per-frame HSV histogram distance on a tiny downscale."""
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dists, prev = [], None
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        dists.append(0.0 if prev is None else
                     float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)))
        prev = h
        if len(dists) % 1000 == 0:
            print(f"  shot pass: {len(dists)} frames ({time.time() - t0:.0f}s)")
    cap.release()

    d = np.array(dists)
    cuts = []
    for i in range(1, len(d)):
        lo, hi = max(0, i - 15), min(len(d), i + 16)
        neigh = np.concatenate([d[lo:i], d[i + 1:hi]])
        local = float(np.median(neigh)) if neigh.size else 0.0
        if d[i] > thresh and d[i] > 3 * local + 0.05:
            if not cuts or i - cuts[-1] >= min_gap_frames:
                cuts.append(i)
    return cuts, fps, len(d), width, height


def shot_of(frame, cuts):
    return bisect.bisect_right(cuts, frame)


def box_contains(box_xyxy, x, y, margin=0.0):
    x1, y1, x2, y2 = box_xyxy
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def runs(frames_sorted):
    """Group sorted ints into contiguous [start, end] runs."""
    out = []
    for f in frames_sorted:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--tracks-cache", required=True)
    ap.add_argument("--track-output", required=True,
                    help="An existing tracker output JSON to analyse (read only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tracker", default=os.path.expanduser("~/Downloads/auto_track_target.py"))
    ap.add_argument("--samples-per-tracklet", type=int, default=6)
    ap.add_argument("--min-tracklet-frames", type=int, default=15)
    ap.add_argument("--cut-thresh", type=float, default=0.35)
    ap.add_argument("--min-lost-frames", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_lines = []

    def say(s=""):
        print(s)
        report_lines.append(s)

    print("Importing tracker module (constants, torso crop, OSNet weights path)...")
    att = load_tracker_module(args.tracker)
    import face_recognition
    MATCH = att.MATCH_THRESHOLD
    REID_T = att.REID_SIM_THRESHOLD
    FACE_OTHER = 0.60  # a clearly different face

    with open(args.reference) as f:
        ref = json.load(f)
    face_refs = [np.array(e) for e in ref.get("face_encodings", [])]
    reid_refs = [np.array(e) for e in ref.get("reid_embeddings", [])]
    reid_ref = np.mean(reid_refs, axis=0) if reid_refs else None

    with open(args.tracks_cache) as f:
        tracks = {int(k): v for k, v in json.load(f).items()}
    with open(args.track_output) as f:
        out_raw = json.load(f)
    output = {int(k): v for k, v in out_raw.items()}

    # ---------------- pass 1: shots ----------------
    print("\nPass 1/2: detecting camera cuts...")
    cuts, fps, n_decoded, W, H = detect_cuts(
        args.clip, args.cut_thresh, min_gap_frames=int(0.5 * 60))
    n_frames = max(n_decoded, (max(tracks) + 1) if tracks else 0)

    say("=" * 72)
    say("TRACKING DIAGNOSTIC REPORT")
    say("=" * 72)
    say(f"clip: {args.clip}")
    say(f"frames decoded: {n_decoded}   frames in YOLO cache: {len(tracks)}   fps: {fps:.2f}   size: {W}x{H}")
    say(f"tracker thresholds: MATCH={MATCH}  REID={REID_T}  SPATIAL_GATE={att.SPATIAL_GATE_PX}  "
        f"MAX_RECOVERY={att.MAX_RECOVERY_PX}  MAX_ASSIGN_COST={att.MAX_ASSIGN_COST}")
    say("")
    say(f"[1] CAMERA CUTS: {len(cuts)} detected")
    say("    " + ", ".join(fmt_t(c, fps).strip() for c in cuts) if cuts else "    (none)")

    # ---------------- tracklets ----------------
    tl_map = defaultdict(lambda: {"frames": [], "boxes": []})
    shots_per_id = defaultdict(set)
    for f in sorted(tracks):
        for b in tracks[f]:
            tid = int(b[0])
            s = shot_of(f, cuts)
            tl_map[(tid, s)]["frames"].append(f)
            tl_map[(tid, s)]["boxes"].append([float(v) for v in b[1:5]])
            shots_per_id[tid].add(s)

    keys = sorted(tl_map, key=lambda k: (tl_map[k]["frames"][0], k[0]))
    tracklets = []
    for i, k in enumerate(keys):
        d = tl_map[k]
        tracklets.append({"idx": i, "name": f"T{i:03d}", "tid": k[0], "shot": k[1],
                          "frames": d["frames"], "boxes": d["boxes"],
                          "box_by_frame": dict(zip(d["frames"], d["boxes"]))})
    ids_crossing_cut = [tid for tid, s in shots_per_id.items() if len(s) > 1]
    say("")
    say(f"[2] TRACKLETS: {len(shots_per_id)} ByteTrack IDs -> {len(tracklets)} tracklets after splitting at cuts")
    say(f"    ByteTrack IDs that survive across a cut (same ID, possibly a different person): "
        f"{len(ids_crossing_cut)}  {ids_crossing_cut[:20]}")

    # frame -> list of tracklet idx present
    present = defaultdict(list)
    for t in tracklets:
        for f in t["frames"]:
            present[f].append(t["idx"])

    # ---------------- choose sample frames ----------------
    K = args.samples_per_tracklet
    needed = defaultdict(list)
    for t in tracklets:
        if len(t["frames"]) < args.min_tracklet_frames:
            continue
        chunks = np.array_split(np.arange(len(t["frames"])), min(K, len(t["frames"])))
        for ch in chunks:
            if len(ch) == 0:
                continue
            areas = [(t["boxes"][j][2] - t["boxes"][j][0]) * (t["boxes"][j][3] - t["boxes"][j][1]) for j in ch]
            j = int(ch[int(np.argmax(areas))])
            needed[t["frames"][j]].append(t["idx"])

    # ---------------- pass 2: evidence ----------------
    print(f"\nPass 2/2: face + body evidence on {len(needed)} sampled frames "
          f"(full-frame HOG, same as tracker -- this is the slow part)...")
    reid_extractor = None
    if reid_ref is not None:
        from torchreid.reid.utils import FeatureExtractor
        reid_extractor = FeatureExtractor(model_name="osnet_x1_0", model_path=att.REID_WEIGHTS, device="cpu")

    samples = defaultdict(list)
    thumbs = {}
    cap = cv2.VideoCapture(args.clip)
    want = sorted(needed)
    wi, f, done, t0 = 0, 0, 0, time.time()
    while wi < len(want):
        ok = cap.grab()
        if not ok:
            break
        if f != want[wi]:
            f += 1
            continue
        ok, frame = cap.retrieve()
        wi += 1
        if not ok:
            f += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            locs = face_recognition.face_locations(rgb, model="hog")
        except Exception:
            locs = []
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs) if locs else []
        faces = []
        for loc, enc in zip(locs, encs):
            top, right, bottom, left = loc
            dist = float(np.min(face_recognition.face_distance(face_refs, enc))) if face_refs else None
            faces.append({"cx": (left + right) / 2, "cy": (top + bottom) / 2,
                          "size": bottom - top, "dist": dist})
        all_boxes = tracks.get(f, [])
        for face in faces:
            face["n_boxes_containing"] = sum(
                1 for b in all_boxes if box_contains(b[1:5], face["cx"], face["cy"]))

        for ti in needed[f]:
            t = tracklets[ti]
            x1, y1, x2, y2 = t["box_by_frame"][f]
            inside = [fc for fc in faces if box_contains((x1, y1, x2, y2), fc["cx"], fc["cy"])]
            rec = {"frame": f, "n_faces_in_box": len(inside),
                   "first_dist": inside[0]["dist"] if inside else None,
                   "min_dist": min((fc["dist"] for fc in inside if fc["dist"] is not None), default=None),
                   "shared_face": any(fc["n_boxes_containing"] > 1 for fc in inside),
                   "face_size": max((fc["size"] for fc in inside), default=None),
                   "crop_dist": None, "reid_sim": None}

            # upsampled HOG on the upper half of the box, only if full-frame HOG found nothing
            if not inside:
                cx1, cy1 = int(max(0, x1)), int(max(0, y1))
                cx2, cy2 = int(min(W, x2)), int(min(H, y1 + 0.5 * (y2 - y1)))
                crop = rgb[cy1:cy2, cx1:cx2]
                if crop.size > 0 and crop.shape[0] >= 20 and crop.shape[1] >= 20:
                    crop = np.ascontiguousarray(crop)
                    try:
                        cl = face_recognition.face_locations(crop, number_of_times_to_upsample=2, model="hog")
                    except Exception:
                        cl = []
                    if cl and face_refs:
                        ce = face_recognition.face_encodings(crop, known_face_locations=cl)
                        if ce:
                            rec["crop_dist"] = min(float(np.min(face_recognition.face_distance(face_refs, e))) for e in ce)

            if reid_extractor is not None:
                tc = att.crop_box_torso(frame, [t["tid"], x1, y1, x2, y2])
                if tc is not None and tc.shape[0] >= 8 and tc.shape[1] >= 8:
                    feat = reid_extractor([tc])[0].cpu().numpy()
                    rec["reid_sim"] = att.cosine_sim(reid_ref, feat)
            samples[ti].append(rec)

            area = (x2 - x1) * (y2 - y1)
            if ti not in thumbs or area > thumbs[ti][0]:
                c = frame[int(max(0, y1)):int(min(H, y2)), int(max(0, x1)):int(min(W, x2))]
                if c.size > 0:
                    thumbs[ti] = (area, c.copy())
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(want)} sampled frames  ({el:.0f}s elapsed, ~{el / done * (len(want) - done):.0f}s left)")
        f += 1
    cap.release()

    # ---------------- per-tracklet verdicts ----------------
    def lost_at(fr):
        o = output.get(fr)
        return True if o is None else bool(o.get("lost"))

    for t in tracklets:
        ss = samples.get(t["idx"], [])
        best = [s["min_dist"] if s["min_dist"] is not None else s["crop_dist"] for s in ss]
        best = [b for b in best if b is not None]
        t["n_samples"] = len(ss)
        t["face_hits_full"] = sum(1 for s in ss if s["n_faces_in_box"] > 0)
        t["face_hits_crop_only"] = sum(1 for s in ss if s["n_faces_in_box"] == 0 and s["crop_dist"] is not None)
        t["votes_target"] = sum(1 for b in best if b <= MATCH)
        t["votes_other"] = sum(1 for b in best if b > FACE_OTHER)
        t["best_face"] = min(best) if best else None
        t["first_face_rejections"] = sum(
            1 for s in ss if s["first_dist"] is not None and s["min_dist"] is not None
            and s["first_dist"] > MATCH and s["min_dist"] <= MATCH)
        t["shared_face_samples"] = sum(1 for s in ss if s["shared_face"])
        sims = [s["reid_sim"] for s in ss if s["reid_sim"] is not None]
        t["reid_med"] = float(np.median(sims)) if sims else None
        t["reid_max"] = max(sims) if sims else None

        if not ss:
            v = "UNSAMPLED"
        elif t["votes_target"] >= 1 and t["votes_target"] >= t["votes_other"]:
            v = "FACE_TARGET"
        elif t["votes_other"] >= 1 and t["votes_target"] == 0:
            v = "FACE_OTHER"
        elif not best:
            v = "NO_FACE"
        else:
            v = "FACE_AMBIG"
        t["verdict"] = v
        t["body_like"] = t["reid_med"] is not None and t["reid_med"] >= REID_T

        followed = lost = 0
        for fr, bx in zip(t["frames"], t["boxes"]):
            o = output.get(fr)
            if o is None:
                continue
            if o.get("lost"):
                lost += 1
            elif box_contains(bx, o["pos"][0], o["pos"][1], margin=40):
                followed += 1
        n = len(t["frames"])
        t["followed_pct"] = 100.0 * followed / n
        t["lost_pct"] = 100.0 * lost / n

    sampled = [t for t in tracklets if t["n_samples"]]
    say("")
    say(f"[3] IDENTITY EVIDENCE over {len(sampled)} sampled tracklets "
        f"(>= {args.min_tracklet_frames} frames each, up to {K} samples each)")
    by_v = defaultdict(list)
    for t in sampled:
        by_v[t["verdict"]].append(t)
    for v in ["FACE_TARGET", "FACE_OTHER", "FACE_AMBIG", "NO_FACE"]:
        say(f"    {v:12s}: {len(by_v[v])}")
    tot_s = sum(t["n_samples"] for t in sampled)
    tot_full = sum(t["face_hits_full"] for t in sampled)
    tot_crop = sum(t["face_hits_crop_only"] for t in sampled)
    say(f"    face found by tracker's HOG call: {tot_full}/{tot_s} samples "
        f"({100 * tot_full / max(tot_s, 1):.0f}%)")
    say(f"    face found ONLY by upsampled crop HOG: {tot_crop}/{tot_s} samples "
        f"(extra detections a better face-detector setting would get)")
    say(f"    samples where the face inside a box was also inside another box: "
        f"{sum(t['shared_face_samples'] for t in sampled)}")
    say(f"    'first face in box' rejected a box that also held a matching face: "
        f"{sum(t['first_face_rejections'] for t in sampled)}")

    tgt = by_v["FACE_TARGET"]
    say("")
    say("    Face-confirmed target tracklets (how much of each one the tracker actually followed):")
    say("    name  tid  shot   span              frames  bestFace  reidMed  followed  lost")
    for t in tgt:
        say(f"    {t['name']} {t['tid']:4d} {t['shot']:4d}  {fmt_t(t['frames'][0], fps)}-{fmt_t(t['frames'][-1], fps)}"
            f"  {len(t['frames']):6d}   {t['best_face']:.3f}    "
            f"{(t['reid_med'] if t['reid_med'] is not None else float('nan')):.3f}   "
            f"{t['followed_pct']:5.1f}%  {t['lost_pct']:5.1f}%")

    # body-similarity separation: is OSNet able to separate target from others at all?
    tgt_sims = [t["reid_med"] for t in tgt if t["reid_med"] is not None]
    oth_sims = [t["reid_med"] for t in by_v["FACE_OTHER"] if t["reid_med"] is not None]
    if tgt_sims and oth_sims:
        say("")
        say(f"    OSNet body sim vs stored reference -- face-confirmed TARGET tracklets: "
            f"min {min(tgt_sims):.3f} / median {np.median(tgt_sims):.3f}")
        say(f"    OSNet body sim vs stored reference -- face-confirmed OTHER tracklets:  "
            f"median {np.median(oth_sims):.3f} / max {max(oth_sims):.3f}")
        say(f"    others above REID threshold {REID_T}: {sum(1 for s in oth_sims if s >= REID_T)}/{len(oth_sims)}")

    # ---------------- lost intervals ----------------
    lost_frames = sorted(fr for fr in range(n_frames) if lost_at(fr))
    intervals = [r for r in runs(lost_frames) if r[1] - r[0] + 1 >= args.min_lost_frames]
    say("")
    say(f"[4] LOST INTERVALS in {os.path.basename(args.track_output)} "
        f"(>= {args.min_lost_frames} frames): {len(intervals)}, "
        f"total lost frames {len(lost_frames)}/{n_frames} ({100 * len(lost_frames) / max(n_frames, 1):.1f}%)")
    cat_frames = defaultdict(int)
    for a, b in intervals:
        on = set()
        for fr in range(a, b + 1):
            on.update(present.get(fr, []))
        on_t = [tracklets[i] for i in sorted(on)]
        ft = [t for t in on_t if t["verdict"] == "FACE_TARGET"]
        bl = [t for t in on_t if t["verdict"] != "FACE_OTHER" and t["body_like"]]
        crop_only = [t for t in ft if t["face_hits_full"] == 0]
        if ft and not crop_only:
            cat = "FACE_TARGET_ON_SCREEN_NOT_RECOVERED"
        elif ft:
            cat = "TARGET_FACE_ONLY_WITH_UPSAMPLED_HOG"
        elif bl:
            cat = f"BODY_ONLY_CANDIDATES({len(bl)})"
        elif on_t:
            cat = "NO_TARGET_EVIDENCE"
        else:
            cat = "NO_DETECTIONS"
        cat_frames[cat.split("(")[0]] += b - a + 1
        cuts_in = [c for c in cuts if a - 30 <= c <= b]
        say(f"    {fmt_t(a, fps)} - {fmt_t(b, fps)} ({(b - a + 1) / fps:5.2f}s)  {cat}"
            f"{'  cut@' + ','.join(fmt_t(c, fps).strip() for c in cuts_in) if cuts_in else ''}")
        for t in ft + [x for x in bl if x not in ft]:
            ov = sum(1 for fr in t["frames"] if a <= fr <= b)
            say(f"        {t['name']} tid={t['tid']} verdict={t['verdict']} bestFace="
                f"{t['best_face'] if t['best_face'] is None else round(t['best_face'], 3)} "
                f"reidMed={t['reid_med'] if t['reid_med'] is None else round(t['reid_med'], 3)} "
                f"overlap={ov}f")
    say("    lost frames by cause:")
    for k, v in sorted(cat_frames.items(), key=lambda kv: -kv[1]):
        say(f"        {k:40s} {v:5d} frames ({v / fps:5.1f}s)")

    # ---------------- suspected wrong person / empty space ----------------
    tl_at = {}
    for t in tracklets:
        for fr, bx in zip(t["frames"], t["boxes"]):
            tl_at.setdefault(fr, []).append((t, bx))
    wrong, empty, unverified, verified = [], 0, 0, 0
    for fr in range(n_frames):
        o = output.get(fr)
        if o is None or o.get("lost"):
            continue
        px, py = o["pos"]
        cands = [(t, bx) for t, bx in tl_at.get(fr, []) if box_contains(bx, px, py, margin=40)]
        if not cands:
            empty += 1
            continue
        t, bx = min(cands, key=lambda c: ((c[1][0] + c[1][2]) / 2 - px) ** 2 + ((c[1][1] + c[1][3]) / 2 - py) ** 2)
        if t["verdict"] == "FACE_OTHER":
            wrong.append(fr)
        elif t["verdict"] == "FACE_TARGET":
            verified += 1
        else:
            unverified += 1
    n_claimed = sum(1 for fr in range(n_frames) if fr in output and not output[fr].get("lost"))
    say("")
    say(f"[5] FRAMES THE TRACKER CLAIMS AS TRACKED: {n_claimed}")
    say(f"    on a face-confirmed TARGET tracklet:      {verified}")
    say(f"    on a face-confirmed OTHER person:         {len(wrong)}   <-- suspected wrong person")
    say(f"    on a tracklet with no face evidence:      {unverified}")
    say(f"    pointing at no detected person at all:    {empty}")
    for a, b in runs(wrong)[:15]:
        say(f"        wrong-person run {fmt_t(a, fps)} - {fmt_t(b, fps)}")

    # ---------------- jumps ----------------
    jumps, oob = [], 0
    prev = None
    for fr in sorted(output):
        p = output[fr]["pos"]
        if not (0 <= p[0] <= W and 0 <= p[1] <= H):
            oob += 1
        if prev is not None and ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5 > 150:
            jumps.append(fr)
        prev = p
    say("")
    say(f"[6] OUTPUT POSITION SANITY: jumps >150px/frame: {len(jumps)}   positions outside the frame: {oob}")

    # ---------------- files ----------------
    with open(os.path.join(args.out_dir, "tracklets.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "tid", "shot", "start_s", "end_s", "frames", "verdict", "best_face",
                    "votes_target", "votes_other", "face_hits_full", "face_hits_crop_only",
                    "first_face_rejections", "shared_face_samples", "reid_med", "reid_max",
                    "followed_pct", "lost_pct"])
        for t in tracklets:
            w.writerow([t["name"], t["tid"], t["shot"], round(t["frames"][0] / fps, 2),
                        round(t["frames"][-1] / fps, 2), len(t["frames"]), t.get("verdict"),
                        t.get("best_face"), t.get("votes_target"), t.get("votes_other"),
                        t.get("face_hits_full"), t.get("face_hits_crop_only"),
                        t.get("first_face_rejections"), t.get("shared_face_samples"),
                        t.get("reid_med"), t.get("reid_max"),
                        round(t["followed_pct"], 1), round(t["lost_pct"], 1)])
    with open(os.path.join(args.out_dir, "labels_template.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "is_target"])
        for t in sampled:
            w.writerow([t["name"], ""])
    with open(os.path.join(args.out_dir, "tracklet_frames.json"), "w") as fh:
        json.dump({t["name"]: {"tid": t["tid"], "shot": t["shot"],
                               "frames": t["frames"], "boxes": t["boxes"]} for t in tracklets}, fh)
    with open(os.path.join(args.out_dir, "cuts.json"), "w") as fh:
        json.dump({"fps": fps, "cuts": cuts}, fh)

    # contact sheets
    TW, TH, COLS, ROWS = 150, 280, 8, 4
    tiles = []
    for t in sampled:
        tile = np.full((TH, TW, 3), 30, np.uint8)
        if t["idx"] in thumbs:
            c = thumbs[t["idx"]][1]
            s = min(TW / c.shape[1], (TH - 70) / c.shape[0])
            r = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
            tile[0:r.shape[0], (TW - r.shape[1]) // 2:(TW - r.shape[1]) // 2 + r.shape[1]] = r
        col = {"FACE_TARGET": (0, 200, 0), "FACE_OTHER": (0, 0, 220)}.get(t["verdict"], (0, 200, 230))
        bf = "-" if t["best_face"] is None else f"{t['best_face']:.2f}"
        rm = "-" if t["reid_med"] is None else f"{t['reid_med']:.2f}"
        lines = [f"{t['name']} id{t['tid']}", f"{t['frames'][0] / fps:.1f}-{t['frames'][-1] / fps:.1f}s",
                 f"{t['verdict'][:11]}", f"f{bf} r{rm} trk{t['followed_pct']:.0f}%"]
        for li, txt in enumerate(lines):
            cv2.putText(tile, txt, (4, TH - 54 + li * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        col if li == 2 else (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    per = COLS * ROWS
    sheets = []
    for si in range(0, len(tiles), per):
        chunk = tiles[si:si + per]
        while len(chunk) % COLS:
            chunk.append(np.zeros((TH, TW, 3), np.uint8))
        rows = [np.hstack(chunk[r:r + COLS]) for r in range(0, len(chunk), COLS)]
        p = os.path.join(args.out_dir, f"tracklet_sheet_{si // per + 1:02d}.png")
        cv2.imwrite(p, np.vstack(rows))
        sheets.append(p)

    say("")
    say(f"Files written to {args.out_dir}:")
    say("    report.txt, tracklets.csv, labels_template.csv, tracklet_frames.json, cuts.json")
    for p in sheets:
        say(f"    {os.path.basename(p)}")
    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
