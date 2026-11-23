#!/usr/bin/env python3
"""
evaluate_tracking.py -- score tracker outputs against hand-labelled tracklets.

Ground truth comes from diagnose_tracking.py's tracklet_frames.json plus a
labels CSV (name,is_target[,note]) where is_target is:
    y  = this tracklet's box is the target
    n  = this tracklet's box is someone else
    ?  = unsure / box covers more than one person (excluded from scoring)
Tracklets not in the CSV (very short ones) count as '?'.

Every frame of every output is classified:
    CORRECT      tracked, crop centre on a target box
    WRONG        tracked, crop centre on a box labelled 'n' (and no target box)
    UNVERIFIED   tracked, crop centre only on '?'/unlabelled boxes
    EMPTY        tracked, crop centre on no detected person
    MISSED       marked lost, but a target box exists in this frame
    LOST_OK      marked lost, no target box labelled in this frame

Usage:
  python3 evaluate_tracking.py --tracklet-frames tracklet_frames.json \
      --labels labels.csv --cuts-json cuts.json \
      --outputs old_track.json new_track.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def inside(b, x, y, m=40):
    return b[0] - m <= x <= b[2] + m and b[1] - m <= y <= b[3] + m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracklet-frames", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cuts-json", required=True)
    ap.add_argument("--outputs", nargs="+", required=True)
    args = ap.parse_args()

    tf = json.load(open(args.tracklet_frames))
    cj = json.load(open(args.cuts_json))
    fps, cuts = float(cj["fps"]), [int(c) for c in cj["cuts"]]

    labels = {}
    with open(args.labels) as fh:
        for row in csv.DictReader(fh):
            v = (row.get("is_target") or "").strip().lower()
            labels[row["name"].strip()] = v if v in ("y", "n") else "?"

    per_frame = defaultdict(list)      # frame -> [(name, label, box)]
    for name, d in tf.items():
        lab = labels.get(name, "?")
        for fr, bx in zip(d["frames"], d["boxes"]):
            per_frame[int(fr)].append((name, lab, bx))
    target_frames = {fr for fr, lst in per_frame.items() if any(l == "y" for _, l, _ in lst)}

    n_y = sum(1 for v in labels.values() if v == "y")
    n_n = sum(1 for v in labels.values() if v == "n")
    print(f"labels: {n_y} target tracklets, {n_n} other, {len(labels) - n_y - n_n} unsure")
    print(f"frames with a labelled target box: {len(target_frames)} ({len(target_frames) / fps:.1f}s)\n")

    results = []
    for path in args.outputs:
        out = {int(k): v for k, v in json.load(open(path)).items()}
        cls = defaultdict(int)
        followed = defaultdict(lambda: [0, ""])
        wrong_frames, prev, jumps, motion = [], None, 0, []
        first_correct = {}
        for fr in sorted(out):
            o = out[fr]
            x, y = o["pos"]
            if prev is not None and fr not in cuts:
                d = ((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5
                motion.append(d)
                if d > 150:
                    jumps += 1
            prev = (x, y)
            if o["lost"]:
                cls["MISSED" if fr in target_frames else "LOST_OK"] += 1
                continue
            cands = [(n, l, b) for n, l, b in per_frame.get(fr, []) if inside(b, x, y)]
            if not cands:
                cls["EMPTY"] += 1
                continue
            if any(l == "y" for _, l, _ in cands):
                cls["CORRECT"] += 1
                s = sum(1 for c in cuts if c <= fr)
                first_correct.setdefault(s, fr)
                n, l, _ = next(c for c in cands if c[1] == "y")
            else:
                n, l, b = min(cands, key=lambda c: ((c[2][0] + c[2][2]) / 2 - x) ** 2 + ((c[2][1] + c[2][3]) / 2 - y) ** 2)
                if l == "n":
                    cls["WRONG"] += 1
                    wrong_frames.append(fr)
                else:
                    cls["UNVERIFIED"] += 1
            followed[n][0] += 1
            followed[n][1] = l

        total = len(out)
        print("=" * 70)
        print(os.path.basename(path))
        print("=" * 70)
        for k in ["CORRECT", "WRONG", "UNVERIFIED", "EMPTY", "MISSED", "LOST_OK"]:
            print(f"  {k:11s} {cls[k]:5d} frames  {cls[k] / fps:5.1f}s  {100 * cls[k] / total:5.1f}%")
        tp = len(target_frames)
        print(f"  -> target-visible frames followed correctly: {100 * cls['CORRECT'] / max(tp, 1):.1f}%")
        tracked = total - cls["MISSED"] - cls["LOST_OK"]
        print(f"  -> of frames claimed as tracked, verified wrong: {100 * cls['WRONG'] / max(tracked, 1):.1f}%")
        print(f"  crop motion: mean {sum(motion) / max(len(motion), 1):.1f}px/frame, jumps >150px outside cuts: {jumps}")

        bounds = [0] + cuts
        print("  time to reacquire after each shot start (first correct frame - first target-visible frame):")
        for s, a in enumerate(bounds):
            b = bounds[s + 1] if s + 1 < len(bounds) else max(out) + 1
            vis = [fr for fr in range(a, b) if fr in target_frames]
            if not vis:
                print(f"    shot {s} ({a / fps:5.1f}s): target never labelled visible")
            elif s in first_correct and first_correct[s] < b:
                print(f"    shot {s} ({a / fps:5.1f}s): {(first_correct[s] - vis[0]) / fps:.2f}s")
            else:
                print(f"    shot {s} ({a / fps:5.1f}s): never reacquired")

        runs = []
        for fr in wrong_frames:
            if runs and fr == runs[-1][1] + 1:
                runs[-1][1] = fr
            else:
                runs.append([fr, fr])
        for a, b in runs[:10]:
            print(f"    WRONG {a / fps:6.2f}s - {b / fps:6.2f}s")
        print("  tracklets this output followed (check any marked n or ?):")
        for n, (cnt, l) in sorted(followed.items(), key=lambda kv: -kv[1][0])[:25]:
            print(f"    {n}  label={l}  {cnt} frames")
        print()
        results.append((path, dict(cls)))


if __name__ == "__main__":
    main()
