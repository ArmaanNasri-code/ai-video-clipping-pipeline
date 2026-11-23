#!/usr/bin/env python3
"""
render_vertical_crop_v4.py -- steadier 9:16 crop. Same inputs/outputs as v3.

v3 (kept unchanged; v4 imports its helpers) re-targets position and zoom every frame
with per-frame smoothing, so:
  * when two people are framed together, their gap changes every frame -> the zoom
    "breathes" constantly (worse at 60 fps: the smoothing was tuned per frame);
  * when the tracker loses the target for 1.5 s it zooms out to the full frame, then
    back in 2 s later -> repeated big zoom-out / zoom-in in fast action (fights).
v4 plans the whole clip first (it is rendered offline, so it can look ahead):
  1. zoom target: rolling median over 1 s, then a dead-band -- the zoom only changes
     when the needed width differs by more than 12% -- then eased over ~0.6 s
  2. position: averaged over 0.5 s, ignores moves smaller than 6% of the frame width,
     eased over ~0.4 s (no jitter following every step of a boxer)
  3. full-frame stability: a shot that would be full-frame for 40%+ of its length, or
     would zoom out more than 3 times a minute (tracking unreliable), is shown
     full-frame for the whole shot; tight stretches under 4 s between full-frame
     stretches become full-frame -- no repeated zoom out/in
  4. all timings are in seconds, so 30 fps and 60 fps streams behave the same
  5. --wide: the whole clip full-frame (gameplay / screen clips: nothing gets cut off)
     --stack: gameplay (centre, full height) on the top 60%, the facecam enlarged on the
     bottom 40% -- what auto-wide now uses for a facecam over gameplay
     (clip_settings.json "gameplay_layout": "wide" switches back to the thin full-frame band)
     --auto-wide: picks --wide by itself when the tracked subject barely moves all clip
     (a facecam over gameplay) or is found under 25% of the time

    python3 render_vertical_crop_v4.py --clip X.mp4 --track T.json --output OUT.mp4 [--wide]
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_vertical_crop_v3 as v3   # noqa: E402  (helpers: plan_presentation, detect_cuts, renderers)

WIDTH_MEDIAN_SEC = 1.0
WIDTH_DEADBAND = 0.12
WIDTH_EASE_SEC = 0.6
CENTER_MEAN_SEC = 0.5
CENTER_DEADZONE_FRAC = 0.06      # of source width
CENTER_EASE_SEC = 0.4
SHOT_FULL_FRAC = 0.40
MAX_ZOOM_OUTS_PER_MIN = 3.0      # more zoom-outs than this in a shot = tracking unreliable -> stay wide
MIN_TIGHT_BETWEEN_FULL_SEC = 4.0
WIDE_RENDER_FACTOR = 1.02
STACK_TOP_FRAC = 0.60            # stacked layout: gameplay gets the top 60%, facecam the bottom 40%
STATIC_SUBJECT_FRAC = 0.03       # --auto-wide: subject moving less than this (std, share of frame) = facecam


def _rolling(x, win, fn):
    if win <= 1 or len(x) <= 1:
        return x.copy()
    h = win // 2
    p = np.pad(x, (h, win - 1 - h), mode="edge")
    if fn == "mean":
        c = np.cumsum(np.insert(p, 0, 0.0))
        return (c[win:] - c[:-win]) / win
    return np.array([np.median(p[i:i + win]) for i in range(len(x))])


def _deadband(x, rel=None, absolute=None):
    out = np.empty_like(x)
    held = x[0]
    for i, v in enumerate(x):
        lim = absolute if absolute is not None else rel * held
        if abs(v - held) > lim:
            held = v
        out[i] = held
    return out


def stabilise_plan(plan, bounds, fps):
    tight_min = int(round(MIN_TIGHT_BETWEEN_FULL_SEC * fps))
    for a, b in bounds:
        seg = plan[a:b + 1]
        if not seg or "full" not in seg:
            continue
        zoom_outs = len(v3.runs(plan, a, b, "full"))
        minutes = max(len(seg) / fps / 60.0, 0.25)
        if seg.count("full") >= SHOT_FULL_FRAC * len(seg) or zoom_outs / minutes > MAX_ZOOM_OUTS_PER_MIN:
            for f in range(a, b + 1):
                plan[f] = "full"
            continue
        for r0, r1 in v3.runs(plan, a, b, "tight"):
            left_full = r0 > a and plan[r0 - 1] == "full"
            right_full = r1 < b and plan[r1 + 1] == "full"
            if left_full and right_full and r1 - r0 + 1 < tight_min:
                for f in range(r0, r1 + 1):
                    plan[f] = "full"
    return plan


def smooth_targets(tx, tw, bounds, fps, src_w, base_w):
    sx, sw = np.array(tx, float), np.array(tw, float)
    win = lambda sec: max(1, int(round(sec * fps)))
    for a, b in bounds:
        w = sw[a:b + 1]
        w = _rolling(w, win(WIDTH_MEDIAN_SEC), "median")
        w = _deadband(w, rel=WIDTH_DEADBAND)
        w = _rolling(w, win(WIDTH_EASE_SEC), "mean")
        sw[a:b + 1] = np.clip(w, base_w, src_w)
        x = sx[a:b + 1]
        x = _rolling(x, win(CENTER_MEAN_SEC), "mean")
        x = _deadband(x, absolute=CENTER_DEADZONE_FRAC * src_w)
        x = _rolling(x, win(CENTER_EASE_SEC), "mean")
        sx[a:b + 1] = x
    return sx, sw


def _cover(img, w, h):
    """resize img to fill exactly w x h, cropping the excess (no stretching)."""
    ih, iw = img.shape[:2]
    sc = max(w / iw, h / ih)
    rw, rh = max(w, int(round(iw * sc))), max(h, int(round(ih * sc)))
    r = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    x, y = (rw - w) // 2, (rh - h) // 2
    return r[y:y + h, x:x + w]


def stack_boxes(src_w, src_h, out_w, out_h, face_x, face_y, face_h_frac):
    """fixed source boxes for the stacked layout: (game_box, face_box) as x1,y1,x2,y2."""
    top_h = int(round(out_h * STACK_TOP_FRAC))
    bot_h = out_h - top_h
    gw = min(src_w, int(round(src_h * out_w / top_h)))           # gameplay: full height, centred
    gx = (src_w - gw) // 2
    fh = int(round(min(0.65, max(0.25, face_h_frac * 1.4)) * src_h))  # facecam: a bit more than the person
    fw = min(src_w, int(round(fh * out_w / bot_h)))
    fx = int(min(max(face_x - fw / 2, 0), src_w - fw))
    fy = int(min(max(face_y - fh / 2, 0), src_h - fh))
    return (gx, 0, gx + gw, src_h), (fx, fy, fx + fw, fy + fh), top_h, bot_h


def render_stack(frame, game_box, face_box, out_w, top_h, bot_h):
    g = frame[game_box[1]:game_box[3], game_box[0]:game_box[2]]
    fc = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
    out = np.empty((top_h + bot_h, out_w, 3), dtype=frame.dtype)
    out[:top_h] = _cover(g, out_w, top_h)
    out[top_h:] = _cover(fc, out_w, bot_h)
    out[top_h - 2:top_h + 2] = 0                                   # thin divider
    return out


def gameplay_layout_setting():
    try:
        v = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_settings.json")))
        return str(v.get("gameplay_layout", "stack")).lower()
    except Exception:
        return "stack"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cuts-json", default=None)
    ap.add_argument("--wide", action="store_true", help="show the whole frame for the entire clip")
    ap.add_argument("--stack", action="store_true",
                    help="stacked layout: gameplay on top, facecam (tracked subject) enlarged below")
    ap.add_argument("--face-height", type=float, default=None,
                    help="tracked person's height as a share of the frame (from the presence check)")
    ap.add_argument("--auto-wide", action="store_true",
                    help="go --wide by itself when the tracked subject barely moves (a facecam over gameplay)")
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"Refusing to overwrite {args.output} -- pick a new name.")

    with open(args.track) as f:
        track = {int(k): v for k, v in json.load(f).items()}
    if not track:
        raise SystemExit(f"ERROR: {args.track} has no tracked positions.")
    cap = cv2.VideoCapture(args.clip)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max(track) + 1)

    keys, data, ki = sorted(track), [], 0
    for f in range(n):
        while ki < len(keys) - 1 and keys[ki + 1] <= f:
            ki += 1
        data.append(track[keys[ki]])
    lost = [bool(d.get("lost", False)) for d in data]

    if all("shot" in d for d in data):
        shot = [int(d["shot"]) for d in data]
    else:
        if args.cuts_json and os.path.exists(args.cuts_json):
            cuts = [int(c) for c in json.load(open(args.cuts_json))["cuts"]]
        else:
            cuts = v3.detect_cuts(args.clip)
        cset, shot, s = set(cuts), [], 0
        for f in range(n):
            s += f in cset
            shot.append(s)

    if args.auto_wide and not args.wide:
        xs = np.array([float(d["pos"][0]) for d, l in zip(data, lost) if not l])
        ys = np.array([float(d["pos"][1]) for d, l in zip(data, lost) if not l and len(d["pos"]) > 1])
        found = len(xs) / max(1, n)
        sx_ = float(np.std(xs)) / src_w if len(xs) else 0.0
        sy_ = float(np.std(ys)) / src_h if len(ys) else 0.0
        static = found < 0.25 or (sx_ < STATIC_SUBJECT_FRAC and sy_ < STATIC_SUBJECT_FRAC)
        print(f"auto-wide: subject found {100 * found:.0f}% of frames, movement x {100 * sx_:.1f}% "
              f"y {100 * sy_:.1f}% -> {'WIDE (static subject / facecam)' if static else 'follow'}")
        args.wide = static
        if static and found >= 0.25 and gameplay_layout_setting() == "stack":
            args.stack = True
            print("auto-wide: using STACKED layout (gameplay top, facecam bottom)")

    plan, bounds = v3.plan_presentation(n, lost, shot, fps)
    for f in range(n):
        if data[f].get("force_full"):
            plan[f] = "full"
    stack = None
    if args.stack:
        pts = [d["pos"] for d, l in zip(data, lost) if not l] or [d["pos"] for d in data]
        fx = float(np.median([p[0] for p in pts]))
        fy = float(np.median([p[1] for p in pts])) if len(pts[0]) > 1 else src_h / 2
        fhf = args.face_height if args.face_height else 0.40
        base_w0 = min(int(round(src_h * v3.TARGET_ASPECT)), src_w)
        stack = stack_boxes(src_w, src_h, base_w0, src_h, fx, fy, fhf)
        print(f"stacked layout: gameplay box {stack[0]}, facecam box {stack[1]}")
        args.wide = True
    if args.wide:
        plan = ["full"] * n
    else:
        plan = stabilise_plan(plan, bounds, fps)

    base_w = min(int(round(src_h * v3.TARGET_ASPECT)), src_w)
    out_w, out_h = base_w, src_h
    max_joint = src_w * v3.MAX_JOINT_WIDTH_FRAC
    tx, tw = [], []
    for f in range(n):
        d = data[f]
        if plan[f] == "full":
            tx.append(src_w / 2)
            tw.append(float(src_w))
        else:
            x, w = float(d["pos"][0]), float(base_w)
            if d.get("mode", "single") == "joint" and d.get("target_width") is not None:
                w = min(max(base_w, float(d["target_width"]) * v3.JOINT_EXTRA_PADDING), max_joint)
            tx.append(x)
            tw.append(w)
    sx, sw = smooth_targets(tx, tw, bounds, fps, src_w, base_w)

    zoom_moves = int(np.sum(np.abs(np.diff(sw)) > 0.5))
    print(f"Source: {src_w}x{src_h} @ {fps:.1f} fps   output {out_w}x{out_h}   shots {len(bounds)}"
          f"   mode {'WIDE' if args.wide else 'follow'}")
    print(f"  full-frame {100 * plan.count('full') / n:.1f}%   tight<->full changes "
          f"{sum(1 for f in range(1, n) if plan[f] != plan[f - 1] and shot[f] == shot[f - 1])}"
          f"   frames with zoom moving {100 * zoom_moves / max(1, n - 1):.1f}%")

    temp_video = args.output.rsplit(".", 1)[0] + "_noaudio.mp4"
    writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f = min(i, n - 1)
        if stack is not None:
            out = render_stack(frame, stack[0], stack[1], out_w, stack[2], stack[3])
        elif sw[f] >= base_w * WIDE_RENDER_FACTOR:
            out = v3.render_wide_joint(frame, sx[f], sw[f], out_w, out_h)
        else:
            out = v3.render_normal_crop(frame, sx[f], base_w)
        writer.write(out)
        i += 1
    cap.release()
    writer.release()
    print(f"Frames rendered: {i}. Muxing audio...")
    r = subprocess.run(["ffmpeg", "-y", "-i", temp_video, "-i", args.clip, "-c:v", "copy", "-c:a", "aac",
                        "-map", "0:v:0", "-map", "1:a:0?", "-shortest", args.output],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit("ffmpeg mux failed")
    os.remove(temp_video)
    print(f"Done. {args.output}")


if __name__ == "__main__":
    main()
