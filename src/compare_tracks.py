#!/usr/bin/env python3
"""
compare_tracks.py -- side-by-side check of two tracker outputs on one clip.

Read-only. Writes into --out-dir:
  compare_report.txt   numbers + every interval worth a look
  compare_sheet.png    the actual frames at those intervals, with
                         RED circle   = where the OLD track points
                         GREEN circle = where the NEW track points
                         grey boxes   = every person YOLO detected
                       (a hollow circle means that track marks the frame "lost")

What it looks for:
  DISAGREE    both trackers claim the target, but point at different places
              (one of them is on the wrong person)
  NEW_GAIN    old was lost, new claims the target (is the gain real?)
  NEW_LOST    new is lost (is the target really absent?)
  BODY_ONLY   new accepted a tracklet on body evidence alone (no face)
  CONFLICT    two separate target tracklets in the same frame (new tracker)
  JUMP        a tracker's position jumps >350 px within 0.1 s inside a shot
              (usually an identity switch)
"""

import argparse
import bisect
import json
import os
from collections import defaultdict

import cv2
import numpy as np


def runs(frames):
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--old", required=True, help="old tracker output json")
    ap.add_argument("--new", required=True, help="new (track_by_tracklets) output json")
    ap.add_argument("--decisions", required=True, help="new tracker's *_decisions.json")
    ap.add_argument("--tracks-cache", required=True, help="YOLO cache used by both")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-category", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    old = {int(k): v for k, v in json.load(open(args.old)).items()}
    new = {int(k): v for k, v in json.load(open(args.new)).items()}
    dec = json.load(open(args.decisions))
    yolo = {int(k): v for k, v in json.load(open(args.tracks_cache)).items()}
    fps, cuts = float(dec["fps"]), [int(c) for c in dec["cuts"]]
    n = min(len(old), len(new))
    shot_of = lambda f: bisect.bisect_right(cuts, f)
    cutset = set(cuts)

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    def T(f):
        return f"{f / fps:6.1f}s"

    # ---------------- per-frame categories ----------------
    cat_frames = defaultdict(list)
    agree = disagree = 0
    for f in range(n):
        o, w = old[f], new[f]
        if not o["lost"] and not w["lost"]:
            d = ((o["pos"][0] - w["pos"][0]) ** 2 + (o["pos"][1] - w["pos"][1]) ** 2) ** 0.5
            if d > 200:
                disagree += 1
                cat_frames["DISAGREE"].append(f)
            else:
                agree += 1
        elif o["lost"] and not w["lost"]:
            cat_frames["NEW_GAIN"].append(f)
        elif w["lost"]:
            cat_frames["NEW_LOST"].append(f)

    # body-only tracklets and per-frame conflicts, from the new tracker's decisions
    by_key = {(t["tid"], t["shot"]): t for t in dec["tracklets"]}
    body_spans = [t for t in dec["tracklets"] if t["decision"] == "TARGET_BODY"]
    for t in body_spans:
        cat_frames["BODY_ONLY"].extend(range(int(round(t["start_s"] * fps)), int(round(t["end_s"] * fps)) + 1))
    for f in range(n):
        tb = [b for b in yolo.get(f, [])
              if by_key.get((b[0], shot_of(f)), {}).get("decision", "").startswith("TARGET")]
        if len(tb) >= 2 and any(iou(tb[i][1:5], tb[j][1:5]) < 0.3
                                for i in range(len(tb)) for j in range(i + 1, len(tb))):
            cat_frames["CONFLICT"].append(f)

    # jumps (identity-switch proxy): >350 px within 6 frames, not across a cut
    jumps = {"old": [], "new": []}
    for name, tr in (("old", old), ("new", new)):
        for f in range(6, n):
            if any(f - 6 < c <= f for c in cutset):
                continue
            a, b = tr[f - 6], tr[f]
            if a["lost"] or b["lost"]:
                continue
            if ((a["pos"][0] - b["pos"][0]) ** 2 + (a["pos"][1] - b["pos"][1]) ** 2) ** 0.5 > 350:
                jumps[name].append(f)
    cat_frames["JUMP_OLD"] = jumps["old"]
    cat_frames["JUMP_NEW"] = jumps["new"]

    # ---------------- report ----------------
    lo = sum(old[f]["lost"] for f in range(n))
    ln = sum(new[f]["lost"] for f in range(n))
    say("=" * 70)
    say("TRACK COMPARISON")
    say("=" * 70)
    say(f"clip: {args.clip}   frames: {n}   fps: {fps:.2f}   cuts: {[round(c / fps, 1) for c in cuts]}")
    say(f"lost:  old {100 * lo / n:5.1f}%   new {100 * ln / n:5.1f}%")
    both = agree + disagree
    say(f"both tracking: {both / fps:6.1f}s   agree (<200px apart) {100 * agree / max(both, 1):5.1f}%   "
        f"DISAGREE {disagree / fps:5.1f}s")
    say(f"new tracks where old was lost (NEW_GAIN): {len(cat_frames['NEW_GAIN']) / fps:5.1f}s")
    say(f"identity-jump proxy (>350px in 0.1s, inside a shot):  old {len(runs(jumps['old']))}   new {len(runs(jumps['new']))}")
    say(f"new: body-only tracklets {len(body_spans)}   conflict frames {len(cat_frames['CONFLICT'])}")
    st = defaultdict(int)
    for f in range(n):
        st[new[f].get("state", "?")] += 1
    say("new states: " + ", ".join(f"{k} {v / fps:.1f}s" for k, v in sorted(st.items())))

    picks = []
    for cat in ["DISAGREE", "CONFLICT", "BODY_ONLY", "NEW_GAIN", "NEW_LOST", "JUMP_NEW", "JUMP_OLD"]:
        rs = [r for r in runs(sorted(set(cat_frames[cat])))]
        min_len = 1 if cat.startswith("JUMP") else int(0.3 * fps)
        rs = [r for r in rs if r[1] - r[0] + 1 >= min_len]
        rs.sort(key=lambda r: -(r[1] - r[0]))
        say("")
        say(f"[{cat}] {len(rs)} interval(s), {sum(r[1] - r[0] + 1 for r in rs) / fps:.1f}s total")
        for r in rs[:15]:
            say(f"    {T(r[0])} - {T(r[1])}  ({(r[1] - r[0] + 1) / fps:5.1f}s)   "
                f"new={new[(r[0] + r[1]) // 2].get('state', '?')}")
        for r in rs[:args.per_category]:
            picks.append((cat, (r[0] + r[1]) // 2, r))

    # ---------------- contact sheet ----------------
    picks.sort(key=lambda p: p[1])
    want = sorted({p[1] for p in picks})
    frames = {}
    cap = cv2.VideoCapture(args.clip)
    for f in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            frames[f] = fr
    cap.release()

    TW, TH = 480, 270
    tiles = []
    for cat, f, r in picks:
        if f not in frames:
            continue
        fr = frames[f]
        sx, sy = TW / fr.shape[1], TH / fr.shape[0]
        img = cv2.resize(fr, (TW, TH))
        for b in yolo.get(f, []):
            cv2.rectangle(img, (int(b[1] * sx), int(b[2] * sy)), (int(b[3] * sx), int(b[4] * sy)), (170, 170, 170), 1)
        for tr, col in ((old, (0, 0, 255)), (new, (0, 220, 0))):
            p = tr[f]["pos"]
            cv2.circle(img, (int(p[0] * sx), int(p[1] * sy)), 14, col, -1 if not tr[f]["lost"] else 3)
        cv2.rectangle(img, (0, 0), (TW, 22), (0, 0, 0), -1)
        cv2.putText(img, f"{cat}  {f / fps:.1f}s  ({(r[1] - r[0] + 1) / fps:.1f}s)  new:{new[f].get('state', '?')}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    COLS = 4
    while len(tiles) % COLS:
        tiles.append(np.zeros((TH, TW, 3), np.uint8))
    if tiles:
        rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
        sheet = np.vstack(rows)
        cv2.imwrite(os.path.join(args.out_dir, "compare_sheet.png"), sheet)
        say(f"\nsheet: {len(picks)} frames -> {os.path.join(args.out_dir, 'compare_sheet.png')}")
    with open(os.path.join(args.out_dir, "compare_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
