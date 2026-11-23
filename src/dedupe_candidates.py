#!/usr/bin/env python3
"""
dedupe_candidates.py -- make sure the same VOD moment is only clipped once.

Candidates come from several places (transcript, chat spikes, visual discovery)
and from re-runs, so the same moment can show up twice under different names
(e.g. visualdisc_06_226m39s and visualdisc_07_226m39s, or a chat clip and a
visual clip of the same knockout). Names carry the VOD time ("226m39s").

Candidates whose VOD times are within WINDOW seconds are one moment. We keep:
  1. the one with the highest visual-judge score (if judged), then
  2. the oldest file (the one earlier runs already worked on)
and MOVE every other copy -- landscape clip, judgment, approved copy, vertical
files, finished folder, production_ready final, review links -- into
<run folder>/_duplicates/<name>/. Nothing is deleted.

Used by run_pipeline.py before the visual judge; can also be run by hand:
  python3 dedupe_candidates.py --output-dir 2902737688_final_review [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import shutil

WINDOW = 30.0


def vod_time(base):
    m = re.search(r"(\d+)m(\d{2})s", base)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def judge_score(output_dir, base):
    try:
        return float(json.load(open(os.path.join(output_dir, "visual_judgments",
                                                 f"{base}_visual.json"))).get("score"))
    except Exception:
        return None


def related_files(output_dir, base):
    va = os.path.join(output_dir, "visual_approved")
    vd = os.path.join(va, "vertical")
    pats = [
        os.path.join(output_dir, f"{base}.mp4"),
        os.path.join(output_dir, "visual_judgments", f"{base}_*.json"),
        os.path.join(va, f"{base}.mp4"),
        os.path.join(vd, f"{base}_*"),
        os.path.join(vd, "finished", base),
        os.path.join(vd, "smart_trim", f"{base}_*"),
        os.path.join(output_dir, "production_ready", f"*_{base}_final.*"),
        os.path.join(output_dir, "needs_review", base),
    ]
    out = []
    for p in pats:
        out += glob.glob(p)
    return out


def dedupe(output_dir, log_fn=None, dry_run=False, window=WINDOW):
    say = (lambda m: log_fn("dedupe", m)) if log_fn else (lambda m: print("[dedupe]", m))
    clips = [p for p in glob.glob(os.path.join(output_dir, "*.mp4")) if "proxy" not in p]
    items = []
    for p in clips:
        b = os.path.splitext(os.path.basename(p))[0]
        t = vod_time(b)
        if t is not None:
            items.append({"base": b, "t": t, "score": judge_score(output_dir, b),
                          "mtime": os.path.getmtime(p)})
    items.sort(key=lambda x: x["t"])
    groups, cur = [], []
    for it in items:
        if cur and it["t"] - cur[-1]["t"] > window:
            groups.append(cur)
            cur = []
        cur.append(it)
    if cur:
        groups.append(cur)

    moved = []
    dup_root = os.path.join(output_dir, "_duplicates")
    for g in groups:
        if len(g) < 2:
            continue
        keep = sorted(g, key=lambda x: (-(x["score"] if x["score"] is not None else -1), x["mtime"]))[0]
        for it in g:
            if it is keep:
                continue
            say(f"{it['base']} is the same moment as {keep['base']} "
                f"({it['t'] // 60}m{it['t'] % 60:02d}s vs {keep['t'] // 60}m{keep['t'] % 60:02d}s)"
                f" -> moved to _duplicates/")
            if dry_run:
                continue
            dest = os.path.join(dup_root, it["base"])
            os.makedirs(dest, exist_ok=True)
            for f in related_files(output_dir, it["base"]):
                rel = os.path.relpath(f, output_dir).replace(os.sep, "__")
                shutil.move(f, os.path.join(dest, rel))
            moved.append(it["base"])
    if not moved and not dry_run:
        say("no duplicate moments")
    return moved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dedupe(a.output_dir, dry_run=a.dry_run)
