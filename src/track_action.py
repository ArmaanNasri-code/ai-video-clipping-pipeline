#!/usr/bin/env python3
"""
track_action.py -- "follow the action" crop track, for clips where the streamer
is not the subject (boxing events, sports, watch-alongs, the streamer only in a
small facecam / commentary-desk strip).

Instead of following one known face, each frame it frames the people who matter
most on screen: the biggest people who are actually moving (e.g. two fighters),
ignoring small people (crowd, a commentary-desk strip, a facecam corner).

Output is the normal track file render_vertical_crop_v3.py already reads:
  {"frame": {"pos": [x, y], "lost": bool, "shot": i, "state": "action",
             "mode": "joint"|"single", "target_width": w}}
"joint" frames two people at once (the renderer widens the crop up to 43% of the
frame); "lost" frames (nobody worth framing) become the full-frame fallback.

Uses the YOLO+ByteTrack cache the streamer tracker already made, so it's fast.
Refuses to overwrite its output.

  python3 track_action.py --clip X.mp4 --tracks-cache X_yolo_tracks.json \
      --decisions X_track_decisions.json --output X_track_action.json
"""
import argparse
import bisect
import json
import os
from collections import defaultdict

import cv2
import numpy as np

MIN_HEIGHT_FRAC = 0.18   # people smaller than this share of the frame height are ignored
                         # (measured on the box's LONGER side, so a fighter lying on the
                         # canvas after a knockdown still counts)
                         # (crowd, commentary strip, facecam). Fighters in a wide
                         # ring shot are ~0.25-0.6 of the frame height.
SECOND_RATIO = 0.45      # frame a second person too if their score is >= 45% of the top one
MAX_JOINT_FRAC = 0.43    # renderer's widest joint crop (share of frame width)
MIN_TRACK_FRAMES = 6     # ignore flicker detections shorter than this
EMA_ALPHA = 0.08
HOLD_FRAMES = 10         # keep the last framing through short detection flicker
CLOSEUP_FRAC = 1.0       # framed width > the widest joint crop ...
CLOSEUP_HEIGHT = 0.90    # ... AND taller than 90% of the frame = a real camera close-up
                         # (head and shoulders filling the screen): show the whole frame.
                         # Wide but shorter (a clinch, a medium shot of two people) gets
                         # the widest crop centred on them instead of the whole frame.
EDGE_PENALTY = 0.35      # people cut off by the left/right frame edge (camera operators,
                         # someone walking past the lens) count much less
FOREGROUND_PENALTY = 0.5 # people cut off at the bottom AND taller than 75% of the frame
                         # (right in front of the camera) count less
BLUR_FRAC = 0.35         # frame sharpness below 35% of the clip's median = a transition
                         # wipe / motion-blur frame -> show the whole frame


def frame_sharpness(clip):
    """Per-frame sharpness (variance of the Laplacian on a small grey frame)."""
    cap = cv2.VideoCapture(clip)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        out.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    cap.release()
    return np.array(out)


def detect_cuts(clip, thresh=0.35):
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    d, prev = [], None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        d.append(0.0 if prev is None else float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)))
        prev = h
    cap.release()
    d = np.array(d)
    cuts = []
    for i in range(1, len(d)):
        neigh = np.concatenate([d[max(0, i - 15):i], d[i + 1:i + 16]])
        local = float(np.median(neigh)) if neigh.size else 0.0
        if d[i] > thresh and d[i] > 3 * local + 0.05 and (not cuts or i - cuts[-1] >= 30):
            cuts.append(i)
    return cuts, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracks-cache", required=True)
    ap.add_argument("--decisions", default=None, help="streamer tracker's decisions json (reuses its cuts)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"Refusing to overwrite {args.output}")

    cap = cv2.VideoCapture(args.clip)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    tracks = {int(k): v for k, v in json.load(open(args.tracks_cache)).items()}
    n = max(n_video, max(tracks) + 1 if tracks else 0)

    cuts = None
    if args.decisions and os.path.exists(args.decisions):
        cuts = [int(c) for c in json.load(open(args.decisions)).get("cuts", [])]
    if cuts is None:
        cuts, _ = detect_cuts(args.clip)
    shot_of = lambda fr: bisect.bisect_right(cuts, fr)

    sharp = frame_sharpness(args.clip)
    med = float(np.median(sharp)) if len(sharp) else 0.0
    blurry = set(int(i) for i in np.where(sharp < BLUR_FRAC * med)[0]) if med > 0 else set()

    # per-(id, shot) track stats: how long, how big, how much it moves
    boxes = defaultdict(dict)
    for fr, lst in tracks.items():
        for b in lst:
            tid = int(b[0])
            x1, y1, x2, y2 = (float(v) for v in b[1:5])
            boxes[(tid, shot_of(fr))][fr] = (x1, y1, x2, y2)
    motion = {}
    for k, bx in boxes.items():
        frs = sorted(bx)
        if len(frs) < MIN_TRACK_FRAMES:
            continue
        steps = []
        for a, b in zip(frs, frs[1:]):
            if b == a + 1:
                ca = ((bx[a][0] + bx[a][2]) / 2, (bx[a][1] + bx[a][3]) / 2)
                cb = ((bx[b][0] + bx[b][2]) / 2, (bx[b][1] + bx[b][3]) / 2)
                hgt = max(1.0, bx[a][3] - bx[a][1])
                steps.append(((cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2) ** 0.5 / hgt)
        motion[k] = float(np.median(steps)) if steps else 0.0

    raw = [None] * n          # (cx, cy, width, joint)
    for fr in range(n):
        if fr in blurry:
            raw[fr] = "transition"
            continue
        s = shot_of(fr)
        cands = []
        for b in tracks.get(fr, []):
            k = (int(b[0]), s)
            if k not in motion:
                continue
            x1, y1, x2, y2 = (float(v) for v in b[1:5])
            h = y2 - y1
            if max(h, x2 - x1) < MIN_HEIGHT_FRAC * H:
                continue
            area = (x2 - x1) * h / (W * H)
            score = area * (0.5 + min(motion[k], 0.05) / 0.05)   # moving people count up to 1.5x
            if x1 <= 0.01 * W or x2 >= 0.99 * W:
                score *= EDGE_PENALTY
            if y2 >= 0.98 * H and h >= 0.75 * H:
                score *= FOREGROUND_PENALTY
            cands.append((score, x1, y1, x2, y2))
        if not cands:
            continue
        cands.sort(reverse=True)
        top = cands[0]
        x1, y1, x2, y2 = top[1:]
        joint = False
        for c in cands[1:3]:
            if c[0] >= SECOND_RATIO * top[0]:
                ux1, ux2 = min(x1, c[1]), max(x2, c[3])
                if ux2 - ux1 <= MAX_JOINT_FRAC * W:
                    x1, x2 = ux1, ux2
                    y1, y2 = min(y1, c[2]), max(y2, c[4])
                    joint = True
                    break
        width = x2 - x1
        if width > CLOSEUP_FRAC * MAX_JOINT_FRAC * W and (y2 - y1) >= CLOSEUP_HEIGHT * H:
            raw[fr] = "closeup"
            continue
        width = min(width, MAX_JOINT_FRAC * W)   # clinch / medium shot: widest crop, centred
        # anyone wider than the 9:16 slice gets a wider crop that fits them
        joint = joint or width > H * 9 / 16
        raw[fr] = ((x1 + x2) / 2, (y1 + y2) / 2, width, joint)

    # fill short gaps (hold), smooth within shots, never across a cut
    bounds = [0] + cuts + [n]
    out, framed = {}, 0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1] - 1
        last, gap = None, 0
        seq = []
        for fr in range(a, b + 1):
            if raw[fr] in ("closeup", "transition"):
                last, gap = None, 0
                seq.append((None, True))
            elif raw[fr] is not None:
                last, gap = raw[fr], 0
                seq.append((raw[fr], False))
            elif last is not None and gap < HOLD_FRAMES:
                gap += 1
                seq.append((last, False))
            else:
                seq.append((None, True))
        # back-fill the start of the shot with the first framing
        first = next((v for v, lost in seq if v is not None), None)
        closeups = set(fr for fr in range(a, b + 1) if raw[fr] == "closeup")
        transitions = set(fr for fr in range(a, b + 1) if raw[fr] == "transition")
        sx = sw = None
        for j, fr in enumerate(range(a, b + 1)):
            v, lost = seq[j]
            if v is None:
                st = ("action_closeup" if fr in closeups else
                      "action_transition" if fr in transitions else "action_none")
                # force_full: render_vertical_crop_v3 shows the whole frame here even
                # for a short stretch, instead of holding the last tight crop
                out[str(fr)] = {"pos": [W / 2, H / 2], "lost": True, "shot": i,
                                "state": st, "force_full": True}
                continue
            if sx is None:
                sx, sw = (first or v)[0], (first or v)[2]
            elif j > 0 and seq[j - 1][0] is None:
                sx, sw = v[0], v[2]      # back after a full-frame gap: start fresh, don't slide
            sx = EMA_ALPHA * v[0] + (1 - EMA_ALPHA) * sx
            sw = EMA_ALPHA * v[2] + (1 - EMA_ALPHA) * sw
            framed += 1
            d = {"pos": [round(sx, 1), round(v[1], 1)], "lost": False, "shot": i, "state": "action"}
            if v[3]:
                d["mode"] = "joint"
                d["target_width"] = round(sw, 1)
            out[str(fr)] = d

    with open(args.output, "w") as f:
        json.dump(out, f)
    joint = sum(1 for d in out.values() if d.get("mode") == "joint")
    close = sum(1 for d in out.values() if d.get("state") == "action_closeup")
    trans = sum(1 for d in out.values() if d.get("state") == "action_transition")
    print(f"action track: {n} frames, {len(cuts) + 1} shots | framed {100 * framed / max(1, n):.0f}% "
          f"(widened {100 * joint / max(1, n):.0f}%), full-frame {100 * (n - framed) / max(1, n):.0f}% "
          f"(close-ups {100 * close / max(1, n):.0f}%, transitions/blur {100 * trans / max(1, n):.0f}%)")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
