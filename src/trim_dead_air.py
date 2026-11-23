#!/usr/bin/env python3
"""
trim_dead_air.py -- remove dead air from a clip before captions.

Dead air = either
  * black screen (ffmpeg blackdetect), or
  * a frozen picture with silent audio at the same time (freezedetect +
    silencedetect) -- e.g. a "technical difficulties" test card or a stalled
    stream. A still picture with people talking is NOT dead air.
Then:
  * leading black   -> cut, keeping LEAD_KEEP seconds
  * trailing black  -> cut, keeping TAIL_KEEP seconds
  * black mid-clip longer than MID_MIN -> squeezed to MID_KEEP seconds
    (half kept from its start, half from its end, so the transition still reads)
Short black flashes (camera cuts, fades) are left alone.

Never overwrites: writes a new file plus <output>.json listing what was cut
(in original-clip seconds), so later stages can map timestamps if they need to.
If nothing needs cutting, the clip is copied unchanged (stream copy, no quality loss).

Usage:
  python3 trim_dead_air.py --input clip.mp4 --output clip_trimmed.mp4
"""

import argparse
import json
import os
import re
import subprocess
import sys

BLACK_MIN = 0.5      # seconds: shorter black runs are ignored entirely
PIX_TH = 0.10        # a pixel this dark (0-1) counts as black
PIC_TH = 0.97        # this share of the frame must be black
LEAD_KEEP = 0.2      # leading black kept
TAIL_KEEP = 0.3      # trailing black kept
MID_MIN = 1.5        # mid-clip black longer than this is squeezed...
MID_KEEP = 1.0       # ...down to this
FREEZE_NOISE = 0.003 # freezedetect noise tolerance
FREEZE_MIN = 1.0     # seconds: a frozen+silent stretch at least this long is dead air
SILENCE_DB = -50     # audio quieter than this counts as silent


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(out.stdout.strip())


def black_runs(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an", "-vf",
                        f"blackdetect=d={BLACK_MIN}:pix_th={PIX_TH}:pic_th={PIC_TH}",
                        "-f", "null", "-"], capture_output=True, text=True)
    runs = []
    for m in re.finditer(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)", r.stderr):
        runs.append((float(m.group(1)), float(m.group(2))))
    return runs


def _spans(stderr, key, dur):
    """parse ffmpeg *_start / *_end pairs; an open start runs to the end of the clip"""
    out, start = [], None
    for m in re.finditer(key + r"_(start|end):\s*([\d.]+)", stderr):
        if m.group(1) == "start":
            start = float(m.group(2))
        elif start is not None:
            out.append((start, float(m.group(2))))
            start = None
    if start is not None:
        out.append((start, dur))
    return out


def frozen_silent_runs(path, dur, audio):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an", "-vf",
                        f"freezedetect=n={FREEZE_NOISE}:d={FREEZE_MIN}", "-f", "null", "-"],
                       capture_output=True, text=True)
    frozen = _spans(r.stderr, "freeze", dur)
    if not audio:
        return [(a, b) for a, b in frozen if b - a >= 2 * FREEZE_MIN]
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af",
                        f"silencedetect=n={SILENCE_DB}dB:d={FREEZE_MIN}", "-f", "null", "-"],
                       capture_output=True, text=True)
    silent = _spans(r.stderr, "silence", dur)
    out = []
    for a, b in frozen:
        for c, d in silent:
            lo, hi = max(a, c), min(b, d)
            if hi - lo >= FREEZE_MIN:
                out.append((lo, hi))
    return out


def merge(runs, gap=0.1):
    out = []
    for a, b in sorted(runs):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def mean_volume(path, a, b):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
                        "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", r.stderr)
    return m.group(1) if m else "?"


def plan_cuts(runs, dur):
    """-> list of (start, end) spans to REMOVE, in original seconds."""
    cuts = []
    eps = 0.05
    for a, b in runs:
        lead = a <= eps
        tail = b >= dur - eps
        if lead and tail:
            continue                      # the whole clip is black: leave it for a human
        if lead:
            if b - LEAD_KEEP > 0.05:
                cuts.append((0.0, b - LEAD_KEEP))
        elif tail:
            if dur - (a + TAIL_KEEP) > 0.05:
                cuts.append((a + TAIL_KEEP, dur))
        elif b - a > MID_MIN:
            half = MID_KEEP / 2
            cuts.append((a + half, b - half))
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"Refusing to overwrite existing output {args.output} -- pick a new name.")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        sys.exit("Output must differ from input.")

    dur = duration(args.input)
    audio = has_audio(args.input)
    black = black_runs(args.input)
    still = frozen_silent_runs(args.input, dur, audio)
    runs = merge(black + still)
    cuts = plan_cuts(runs, dur)

    print(f"{os.path.basename(args.input)}: {dur:.2f}s, {len(black)} black run(s), "
          f"{len(still)} frozen+silent run(s) -> {len(runs)} dead-air stretch(es)")
    for a, b in black:
        print(f"  black          {a:7.2f}s - {b:7.2f}s  ({b - a:5.2f}s)" +
              (f"   audio mean {mean_volume(args.input, a, b)} dB" if audio else ""))
    for a, b in still:
        print(f"  frozen+silent  {a:7.2f}s - {b:7.2f}s  ({b - a:5.2f}s)")

    report = {"input": os.path.abspath(args.input), "duration_in": dur,
              "black_runs": black, "frozen_silent_runs": still,
              "dead_air": runs, "removed": cuts}

    if not cuts:
        print("nothing to trim -- copying unchanged")
        subprocess.run(["ffmpeg", "-v", "error", "-i", args.input, "-c", "copy",
                        "-movflags", "+faststart", args.output], check=True)
        report["duration_out"] = dur
    else:
        keep, t = [], 0.0
        for a, b in sorted(cuts):
            if a > t:
                keep.append((t, a))
            t = max(t, b)
        if t < dur:
            keep.append((t, dur))
        parts, labels = [], []
        for i, (a, b) in enumerate(keep):
            parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
            lab = f"[v{i}]"
            if audio:
                parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
                lab += f"[a{i}]"
            labels.append(lab)
        parts.append("".join(labels) + f"concat=n={len(keep)}:v=1:a={1 if audio else 0}"
                     + ("[v][a]" if audio else "[v]"))
        cmd = ["ffmpeg", "-v", "error", "-i", args.input, "-filter_complex", ";".join(parts),
               "-map", "[v]"]
        if audio:
            cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output]
        subprocess.run(cmd, check=True)
        report["duration_out"] = duration(args.output)
        removed = sum(b - a for a, b in cuts)
        for a, b in cuts:
            print(f"  removed {a:7.2f}s - {b:7.2f}s  ({b - a:5.2f}s)")
        print(f"trimmed {removed:.2f}s: {dur:.2f}s -> {report['duration_out']:.2f}s")

    with open(args.output + ".json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
