#!/usr/bin/env python3
"""
analyze_retention.py -- read-only first look at pacing in finished clips.

For each clip it measures, every half second:
  - loudness (crowd / streamer yelling)
  - on-screen motion (how much the picture changes)
and combines them into one "energy" curve. From that it reports:
  - where the peak moment is
  - how long the clip runs before things heat up (slow setup)
  - how long it keeps going after the last hot moment (weak ending)
  - a suggested tighter window (first hot moment - 2s ... last hot moment + 3s)
It also saves a chart per clip so you can compare it with what you see.

Nothing is changed or cut. Writes only into --out (default retention_report/).

  python3 analyze_retention.py 2902737688_final_review/production_ready/*_final.mp4
"""
import argparse
import json
import os
import subprocess

import numpy as np

STEP = 0.5          # seconds per measurement
SMOOTH = 2.0        # seconds of smoothing
HOT_LEVEL = 0.6     # "hot" = above 60% of the way from the median to the peak
LEAD_IN = 2.0       # suggested window starts this long before the first hot moment
TAIL = 3.0          # ... and ends this long after the last one


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(r.stdout.strip())


def loudness(path, n):
    sr = 16000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * STEP)
    out = np.zeros(n)
    for i in range(n):
        seg = a[i * hop:(i + 1) * hop]
        out[i] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-6) if seg.size else -120
    return out


def motion(path, n):
    w, h = 90, 160
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf",
                          f"fps={1 / STEP},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).astype(np.float32)
    out = np.zeros(n)
    for i in range(1, min(n, len(frames))):
        out[i] = np.mean(np.abs(frames[i] - frames[i - 1]))
    out[0] = out[1] if n > 1 else 0
    return out


def z(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 1e-6 else x * 0


def smooth(x):
    k = max(1, int(SMOOTH / STEP))
    return np.convolve(x, np.ones(k) / k, mode="same")


def fmt(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="retention_report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    rows = []
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        dur = duration(v)
        n = int(dur / STEP)
        loud, mot = loudness(v, n), motion(v, n)
        energy = smooth(0.6 * z(loud) + 0.4 * z(mot))
        t = np.arange(n) * STEP
        peak_i = int(np.argmax(energy))
        med = float(np.median(energy))
        thr = med + HOT_LEVEL * (energy[peak_i] - med)
        hot = np.where(energy >= thr)[0]
        first, last = t[hot[0]], t[hot[-1]]
        win0, win1 = max(0.0, first - LEAD_IN), min(dur, last + TAIL)
        row = {"clip": name, "duration": round(dur, 1), "peak_at": round(float(t[peak_i]), 1),
               "slow_setup_s": round(first, 1), "after_last_hot_s": round(dur - last, 1),
               "suggested_window": [round(win0, 1), round(win1, 1)],
               "suggested_length": round(win1 - win0, 1)}
        rows.append(row)
        if plt:
            fig, ax = plt.subplots(figsize=(10, 2.6))
            ax.plot(t, energy, color="#2b6cb0", lw=1.6, label="energy (loudness + motion)")
            ax.axhline(thr, color="#999", ls="--", lw=1, label="hot level")
            ax.axvspan(win0, win1, color="#38a169", alpha=0.12, label="suggested window")
            ax.axvline(t[peak_i], color="#e53e3e", lw=1.2, label=f"peak {fmt(t[peak_i])}")
            ax.set_xlabel("seconds")
            ax.set_yticks([])
            ax.set_title(name, fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"{name}_energy.png"), dpi=110)
            plt.close(fig)

    print(f"\n{'clip':42s} {'len':>5s} {'peak':>7s} {'setup':>6s} {'tail':>6s}  suggested")
    for r in rows:
        print(f"{r['clip'][:42]:42s} {r['duration']:5.1f} {fmt(r['peak_at']):>7s} "
              f"{r['slow_setup_s']:5.1f}s {r['after_last_hot_s']:5.1f}s  "
              f"{fmt(r['suggested_window'][0])}-{fmt(r['suggested_window'][1])} "
              f"({r['suggested_length']:.0f}s)")
    json.dump(rows, open(os.path.join(args.out, "_retention.json"), "w"), indent=2)
    print(f"\ncharts and _retention.json saved in {args.out}/ (nothing was cut or changed)")


if __name__ == "__main__":
    main()
