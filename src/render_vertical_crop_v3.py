#!/usr/bin/env python3
"""
render_vertical_crop_v2.py

Same output as render_vertical_crop.py (608x1080 portrait, original audio),
plus three behaviours the original lacks:

  1. Full-frame fallback. When the target is lost for a sustained stretch,
     the whole landscape frame is shown, centred over a blurred 9:16
     background, instead of a tight crop on whoever happens to be standing
     where the target used to be.

  2. No flicker. Because rendering is offline, the presentation for the
     whole clip is planned before any frame is drawn:
       - a loss shorter than FULL_MIN_LOST_SEC stays a tight crop (hold)
       - a tight stretch shorter than MIN_TIGHT_SEC that sits between
         full-frame stretches (or between one and a cut) becomes full-frame
         too, so the view never pops in and out.
     Transitions inside a shot are a smooth zoom.

  3. Hard cuts snap. At the first frame of every shot, position and zoom
     jump straight to their new values; smoothing only happens within a
     shot.

Track file: the usual {"frame": {"pos": [x, y], "lost": bool}}. Optional
keys used when present: "shot" (shot index, written by
track_by_tracklets.py), "mode": "joint" + "target_width" (two-person
framing, behaves exactly as in the original renderer).
If the track has no "shot" keys, pass --cuts-json, or cuts are detected
from the video.

Works for any target: it only reads the track file.
"""

import argparse
import json
import os
import subprocess

import cv2
import numpy as np


TARGET_ASPECT = 9 / 16

# --- unchanged from render_vertical_crop.py ---
CENTER_SMOOTH_ALPHA = 0.14
ZOOM_SMOOTH_ALPHA = 0.08
MAX_JOINT_WIDTH_FRAC = 0.43
MIN_WIDE_FACTOR = 1.08
JOINT_EXTRA_PADDING = 1.08

# --- new ---
FULL_MIN_LOST_SEC = 1.5     # losses shorter than this stay a tight crop (hold). Measured on Kai +
                            # Deen: 0.6 s gave zoom-out blips during fast camera pans (Deen 16
                            # transitions, Kai 3); 1.5 s -> Deen 8, Kai 1, no identity change
MIN_TIGHT_SEC = 2.0         # tight stretches shorter than this, next to full-frame, become full-frame
                            # (stops wide/tight/wide flips when the camera whips between people)
FULL_ZOOM_ALPHA = 0.12      # zoom speed for tight <-> full transitions inside a shot (~0.4 s)
CUT_THRESH = 0.35


def lerp(current, target, alpha):
    return current + (target - current) * alpha


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def make_blurred_background(frame, output_width, output_height):
    src_h, src_w = frame.shape[:2]
    scale = max(output_width / src_w, output_height / src_h)
    rw, rh = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    x1, y1 = max(0, (rw - output_width) // 2), max(0, (rh - output_height) // 2)
    bg = resized[y1:y1 + output_height, x1:x1 + output_width]
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=28, sigmaY=28)
    return (bg.astype(np.float32) * 0.72).astype(np.uint8)


def render_normal_crop(frame, center_x, crop_width):
    src_h, src_w = frame.shape[:2]
    x1 = int(clamp(int(round(center_x - crop_width / 2)), 0, src_w - crop_width))
    return frame[0:src_h, x1:x1 + crop_width]


def render_wide_joint(frame, center_x, virtual_width, output_width, output_height):
    """Wider landscape slice (up to the full frame), aspect preserved,
    centred over a blurred portrait background."""
    src_h, src_w = frame.shape[:2]
    virtual_width = int(round(clamp(virtual_width, output_width, src_w)))
    x1 = int(clamp(int(round(center_x - virtual_width / 2)), 0, src_w - virtual_width))
    wide_crop = frame[0:src_h, x1:x1 + virtual_width]
    scale = output_width / virtual_width
    fg_h = min(int(round(src_h * scale)), output_height)
    foreground = cv2.resize(wide_crop, (output_width, fg_h), interpolation=cv2.INTER_AREA)
    canvas = make_blurred_background(frame, output_width, output_height)
    y1 = (output_height - fg_h) // 2
    canvas[y1:y1 + fg_h, 0:output_width] = foreground
    return canvas


def detect_cuts(clip):
    cap = cv2.VideoCapture(clip)
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
        if d[i] > CUT_THRESH and d[i] > 3 * local + 0.05 and (not cuts or i - cuts[-1] >= 30):
            cuts.append(i)
    return cuts


def runs(flags, a, b, value):
    """[start, end] runs inside [a, b] where flags[f] == value."""
    out, cur = [], None
    for f in range(a, b + 1):
        if flags[f] == value:
            if cur is None:
                cur = [f, f]
            else:
                cur[1] = f
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def plan_presentation(n, lost, shot, fps):
    """Return per-frame 'tight' / 'full' for the whole clip, decided offline."""
    full_min = int(round(FULL_MIN_LOST_SEC * fps))
    tight_min = int(round(MIN_TIGHT_SEC * fps))
    plan = ["tight"] * n
    starts = [0] + [f for f in range(1, n) if shot[f] != shot[f - 1]]
    bounds = [(s, (starts[i + 1] - 1) if i + 1 < len(starts) else n - 1) for i, s in enumerate(starts)]
    for a, b in bounds:
        # 1. sustained losses become full-frame
        for r0, r1 in runs(lost, a, b, True):
            if r1 - r0 + 1 >= full_min:
                for f in range(r0, r1 + 1):
                    plan[f] = "full"
        if "full" not in plan[a:b + 1]:
            continue
        # 2. absorb short tight stretches next to full-frame (or a cut), so
        #    the view doesn't pop in and out
        for r0, r1 in runs(plan, a, b, "tight"):
            left_ok = r0 == a or plan[r0 - 1] == "full"
            right_ok = r1 == b or plan[r1 + 1] == "full"
            if r1 - r0 + 1 < tight_min and left_ok and right_ok:
                for f in range(r0, r1 + 1):
                    plan[f] = "full"
    return plan, bounds


def main():
    ap = argparse.ArgumentParser(description="9:16 crop renderer with full-frame fallback and hard-cut snapping.")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cuts-json", default=None, help="only needed if the track file has no 'shot' keys")
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
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = max(n_video, max(track) + 1)

    # per-frame data, carrying the last known entry forward like the original renderer
    keys = sorted(track)
    data, ki = [], 0
    for f in range(n):
        while ki < len(keys) - 1 and keys[ki + 1] <= f:
            ki += 1
        data.append(track[keys[ki]])
    lost = [bool(d.get("lost", False)) for d in data]

    if all("shot" in d for d in data):
        shot = [int(d["shot"]) for d in data]
        cut_source = "track file"
    else:
        if args.cuts_json and os.path.exists(args.cuts_json):
            cuts = [int(c) for c in json.load(open(args.cuts_json))["cuts"]]
            cut_source = args.cuts_json
        else:
            print("Track has no shot info and no --cuts-json: detecting cuts from the video...")
            cuts = detect_cuts(args.clip)
            cut_source = "detected from video"
        shot, s = [], 0
        cset = set(cuts)
        for f in range(n):
            if f in cset:
                s += 1
            shot.append(s)

    plan, bounds = plan_presentation(n, lost, shot, fps)
    forced = 0
    for f in range(n):
        if data[f].get("force_full") and plan[f] != "full":
            plan[f] = "full"
            forced += 1
    if forced:
        print(f"force_full frames (transitions/close-ups/no subject): {forced}")
    shot_start = {a for a, _ in bounds}

    base_w = min(int(round(src_h * TARGET_ASPECT)), src_w)
    out_w, out_h = base_w, src_h
    max_joint = src_w * MAX_JOINT_WIDTH_FRAC
    print(f"Source: {src_w}x{src_h}   Portrait output: {out_w}x{out_h}")
    print(f"Shots: {len(bounds)} (cuts from {cut_source})")

    temp_video = args.output.rsplit(".", 1)[0] + "_noaudio.mp4"
    writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    sx, swidth = float(data[0]["pos"][0]), float(base_w)
    counts = {"tight": 0, "joint": 0, "full": 0}
    transitions, frame_idx = 0, 0
    print("Rendering...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f = min(frame_idx, n - 1)
        d = data[f]

        if plan[f] == "full":
            tx, tw = src_w / 2, float(src_w)
        else:
            tx, tw = float(d["pos"][0]), float(base_w)
            if d.get("mode", "single") == "joint" and d.get("target_width") is not None:
                tw = min(max(base_w, float(d["target_width"]) * JOINT_EXTRA_PADDING), max_joint)
                counts["joint"] += 1

        if f in shot_start:
            # hard cut: no smoothing across it
            sx, swidth = tx, tw
        else:
            if plan[f] != plan[f - 1]:
                transitions += 1
            sx = lerp(sx, tx, CENTER_SMOOTH_ALPHA)
            zoom_alpha = FULL_ZOOM_ALPHA if (plan[f] == "full" or plan[max(0, f - 30)] == "full") else ZOOM_SMOOTH_ALPHA
            swidth = lerp(swidth, tw, zoom_alpha)

        if swidth >= base_w * MIN_WIDE_FACTOR:
            rendered = render_wide_joint(frame, sx, swidth, out_w, out_h)
        else:
            rendered = render_normal_crop(frame, sx, base_w)
        if plan[f] == "full":
            counts["full"] += 1
        else:
            counts["tight"] += 1
        writer.write(rendered)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Frames rendered: {frame_idx}")
    if frame_idx:
        print(f"  tight crop:  {100 * counts['tight'] / frame_idx:5.1f}%  (joint framing on {counts['joint']} of these)")
        print(f"  full frame:  {100 * counts['full'] / frame_idx:5.1f}%")
        print(f"  tight<->full transitions inside shots: {transitions}")
        print(f"  hard cuts snapped: {len(bounds) - 1}")
        for i, (a, b) in enumerate(bounds):
            fl = sum(1 for x in range(a, min(b, frame_idx - 1) + 1) if plan[x] == "full")
            print(f"    shot {i}: {a / fps:6.2f}s-{b / fps:6.2f}s  full-frame {100 * fl / (b - a + 1):5.1f}%")

    print("Muxing original audio back in...")
    cmd = ["ffmpeg", "-y", "-i", temp_video, "-i", args.clip, "-c:v", "copy", "-c:a", "aac",
           "-map", "0:v:0", "-map", "1:a:0?", "-shortest", args.output]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg mux failed:")
        print(result.stderr[-1500:])
        return
    os.remove(temp_video)
    print(f"Done. {args.output}")


if __name__ == "__main__":
    main()
