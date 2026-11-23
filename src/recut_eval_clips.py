#!/usr/bin/env python3
"""
recut_eval_clips.py -- rebuild the eval-set clips whose videos were cleaned up.

Your labelled eval set (newest eval_set_labeled*.csv) points at candidate videos that
free_space.py later deleted. This re-cuts each one from the Twitch VOD with the SAME
window the pipeline used, so the judge test sees what you labelled:

  final_*       merged_start..merged_end from <vod>_final.json (same as the refetch stage)
  chat_* / visualdisc_*
                centred on the time in the name, same length as the original candidate
                (the length is read from its old judgment file)

Writes to eval_set/recut/<vod>/<name>.mp4 (720p max). Changes nothing else, uses no
Anthropic credits. Already-cut clips are skipped, so it is safe to re-run.

    python3 recut_eval_clips.py            # do it
    python3 recut_eval_clips.py --dry-run  # just show the windows
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys


def labelled_csv():
    files = sorted(glob.glob("eval_set_labeled*.csv"), key=os.path.getmtime, reverse=True)
    if not files:
        sys.exit("ERROR: no eval_set_labeled*.csv in this folder -- run from ~/Downloads.")
    return files[0]


def final_windows(vod):
    try:
        data = json.load(open(f"{vod}_final.json"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return [d for d in data if isinstance(d, dict) and "merged_start" in d and "merged_end" in d]


def old_duration(run, name):
    try:
        return float(json.load(open(os.path.join(run, "visual_judgments", f"{name}_visual.json")))["duration"])
    except Exception:
        return None


def window(vod, run, name):
    m = re.search(r"_(\d+)m(\d{2})s", name)
    if not m:
        return None, "no time in name"
    t = int(m.group(1)) * 60 + int(m.group(2))
    if name.startswith("final_"):
        for d in final_windows(vod):
            if abs(int(d["merged_start"]) - t) <= 1:
                return (float(d["merged_start"]), float(d["merged_end"])), "from final.json"
    dur = old_duration(run, name)
    if dur:
        return (max(0.0, t - dur / 2), t + dur / 2), f"centred, {dur:.0f}s like the original"
    return (max(0.0, t - 35), t + 35), "centred, 70s (original length unknown)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = labelled_csv()
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    todo = []
    for r in rows:
        cp = (r.get("clip_path") or "").strip()
        if not cp or os.path.exists(cp):
            continue
        name = os.path.splitext(os.path.basename(cp))[0]
        vod = (r.get("vod") or "").strip() or (re.match(r"(\d{9,})", r.get("id", "")) or [None, None])[1]
        if not vod:
            print(f"  skip {name}: can't tell which VOD")
            continue
        out = os.path.join("eval_set", "recut", vod, f"{name}.mp4")
        win, how = window(vod, os.path.dirname(cp), name)
        if not win:
            print(f"  skip {name}: {how}")
            continue
        todo.append((vod, name, out, win, how))

    print(f"{path}: {len(todo)} clip(s) need re-cutting\n")
    done = failed = 0
    for i, (vod, name, out, (a, b), how) in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {vod} {name}  {a:.0f}-{b:.0f}s ({how})"
        if os.path.exists(out):
            print(f"{tag}  already cut")
            done += 1
            continue
        print(tag, flush=True)
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-S", "res:720",
               "--download-sections", f"*{a:.1f}-{b:.1f}", "--force-keyframes-at-cuts",
               "-o", out, f"https://www.twitch.tv/videos/{vod}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            done += 1
        else:
            failed += 1
            print(f"    failed: {(r.stderr or r.stdout).strip()[-300:]}")
    print(f"\n{done} ready, {failed} failed. Clips are in eval_set/recut/. Nothing else was changed.")


if __name__ == "__main__":
    main()
