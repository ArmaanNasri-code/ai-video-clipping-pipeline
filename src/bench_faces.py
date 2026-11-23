#!/usr/bin/env python3
"""
bench_faces.py -- read-only benchmark: dlib HOG (current) vs InsightFace.

On evenly sampled frames of one clip it measures:
  * speed per frame for each detector
  * faces found by each, and faces ONLY InsightFace finds (turned / small /
    partly hidden faces HOG misses)
  * identity: InsightFace's own face fingerprints (ArcFace) for the target are
    collected AUTOMATICALLY from faces the current dlib reference already
    confirms (distance <= 0.42) -- no manual seeding. Then every face is scored
    against that set, to see whether ArcFace separates target from others.
Writes into --out-dir:
  bench_report.txt
  insight_only_faces.png   crops of faces only InsightFace found, with their
                           similarity to the target (check by eye)
"""

import argparse
import json
import os
import time

import cv2
import numpy as np


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--reference", required=True, help="existing target_reference_<name>.json (dlib)")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--model", default="buffalo_l", help="buffalo_l (accurate) or buffalo_s (fast)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    import face_recognition
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    avail = ort.get_available_providers()
    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in avail]
    app = None
    for provs in ([providers[0]] + (["CPUExecutionProvider"] if providers[0] != "CPUExecutionProvider" else []),
                  ["CPUExecutionProvider"]):
        try:
            app = FaceAnalysis(name=args.model, providers=provs,
                               allowed_modules=["detection", "recognition", "landmark_3d_68"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            used = provs
            break
        except Exception as e:
            print(f"provider {provs} failed ({e}); trying CPU only")
    if app is None:
        raise SystemExit("InsightFace could not load")

    refs = np.array(json.load(open(args.reference))["face_encodings"])
    cap = cv2.VideoCapture(args.clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    want = sorted(set(np.linspace(0, n - 1, args.samples).astype(int).tolist()))

    # warm-up (first CoreML call compiles the model; don't count it)
    ok, fr0 = cap.read()
    if ok:
        app.get(fr0)

    t_hog = t_ins = 0.0
    rows = []
    for k, f in enumerate(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        locs = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs) if locs else []
        t_hog += time.time() - t0
        hog = []
        for (tp, r, b, l), e in zip(locs, encs):
            d = float(np.min(np.linalg.norm(refs - e, axis=1)))
            hog.append({"box": (l, tp, r, b), "d": d})
        t0 = time.time()
        faces = app.get(frame)
        t_ins += time.time() - t0
        ins = []
        for fc in faces:
            bx = [float(v) for v in fc.bbox]
            emb = fc.normed_embedding if getattr(fc, "normed_embedding", None) is not None else None
            pose = getattr(fc, "pose", None)
            yaw = float(pose[1]) if pose is not None else None
            match = max(hog, key=lambda h: iou(h["box"], bx), default=None)
            m_iou = iou(match["box"], bx) if match else 0.0
            ins.append({"box": bx, "score": float(fc.det_score), "emb": emb, "yaw": yaw,
                        "dlib_d": match["d"] if match and m_iou > 0.3 else None})
        rows.append({"frame": f, "frame_img": frame if any(i["dlib_d"] is None for i in ins) else None,
                     "hog": hog, "ins": ins})
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(want)} frames")
    cap.release()
    m = len(rows)

    # automatic ArcFace gallery: faces dlib already confirms as the target
    gallery = [i["emb"] for r in rows for i in r["ins"] if i["dlib_d"] is not None and i["dlib_d"] <= 0.42 and i["emb"] is not None]
    for r in rows:
        for i in r["ins"]:
            i["sim"] = (max(float(np.dot(i["emb"], g)) for g in gallery if g is not i["emb"])
                        if gallery and i["emb"] is not None and len(gallery) > 1 else None)

    say("=" * 70)
    say(f"FACE DETECTOR BENCHMARK  {os.path.basename(args.clip)}  ({m} frames sampled)")
    say("=" * 70)
    say(f"InsightFace model {args.model}, providers {used}")
    say(f"speed per frame:   HOG {1000 * t_hog / m:6.0f} ms    InsightFace {1000 * t_ins / m:6.0f} ms    "
        f"-> {t_hog / max(t_ins, 1e-9):.1f}x faster")
    nh = sum(len(r["hog"]) for r in rows)
    ni = sum(len(r["ins"]) for r in rows)
    fh = sum(1 for r in rows if r["hog"])
    fi = sum(1 for r in rows if r["ins"])
    only = [(r, i) for r in rows for i in r["ins"] if i["dlib_d"] is None]
    say(f"faces found:       HOG {nh:4d}   InsightFace {ni:4d}   (InsightFace-only: {len(only)})")
    say(f"frames with a face: HOG {fh}/{m}   InsightFace {fi}/{m}")
    yaws = [abs(i["yaw"]) for _, i in only if i["yaw"] is not None]
    if yaws:
        say(f"InsightFace-only faces: median |yaw| {np.median(yaws):.0f} deg, "
            f"{sum(1 for y in yaws if y > 30)} turned >30 deg (HOG mostly misses these)")
    say("")
    say(f"automatic ArcFace gallery from dlib-confirmed target faces: {len(gallery)} faces")
    tgt = [i["sim"] for r in rows for i in r["ins"] if i["dlib_d"] is not None and i["dlib_d"] <= 0.42 and i["sim"] is not None]
    oth = [i["sim"] for r in rows for i in r["ins"] if i["dlib_d"] is not None and i["dlib_d"] > 0.68 and i["sim"] is not None]
    amb = [i["sim"] for r in rows for i in r["ins"] if i["dlib_d"] is not None and 0.42 < i["dlib_d"] <= 0.68 and i["sim"] is not None]
    for name, v in (("dlib says TARGET", tgt), ("dlib says OTHER", oth), ("dlib AMBIGUOUS", amb),
                    ("InsightFace-only", [i["sim"] for _, i in only if i["sim"] is not None])):
        if v:
            v = np.array(v)
            say(f"  ArcFace similarity to target, {name:18s} n={len(v):4d}  min {v.min():.2f}  median {np.median(v):.2f}  max {v.max():.2f}")
    say("  (ArcFace: same person typically > 0.45, different people typically < 0.25)")

    # sheet of InsightFace-only faces, most target-like first
    only.sort(key=lambda ri: -(ri[1]["sim"] if ri[1]["sim"] is not None else -1))
    tiles = []
    for r, i in only[:40]:
        img = r["frame_img"]
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in i["box"]]
        pad = int(0.4 * (x2 - x1))
        c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if c.size == 0:
            continue
        c = cv2.resize(c, (150, 150))
        lab = f"{r['frame'] / fps:.1f}s s={i['sim']:.2f}" if i["sim"] is not None else f"{r['frame'] / fps:.1f}s"
        cv2.rectangle(c, (0, 130), (150, 150), (0, 0, 0), -1)
        cv2.putText(c, lab, (3, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 220, 0) if (i["sim"] or 0) > 0.45 else (0, 0, 255), 1, cv2.LINE_AA)
        if i["yaw"] is not None:
            cv2.putText(c, f"yaw {i['yaw']:.0f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(c)
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros((150, 150, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
        p = os.path.join(args.out_dir, "insight_only_faces.png")
        cv2.imwrite(p, sheet)
        say(f"\nsheet of InsightFace-only faces: {p}  (green = looks like the target)")
    with open(os.path.join(args.out_dir, "bench_report.txt"), "w") as fh_:
        fh_.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
