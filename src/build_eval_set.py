#!/usr/bin/env python3
"""
build_eval_set.py -- gather every clip the visual judge has scored (plus your
keepers) into one labelling sheet for the judge-calibration experiment.

Read-only for the pipeline: it never changes, moves or deletes pipeline files.
It only writes into --out (default eval_set/):
  eval_set.csv     one row per clip: judge score/reason + empty columns for your labels
  clips/           links to each clip, named  s6__<vod>__<clip>.mp4  so they sort by
                   judge score and open from one folder

Duplicate runs of the same VOD (e.g. two folders for 2837447163) are merged:
one row per (vod, clip).

Usage:
  python3 build_eval_set.py --runs 2837447163_tracklets_run 2837447163_final_review \
      2918060463_final_review --keepers keepers
"""
import argparse
import csv
import glob
import json
import os
import re

LABEL_COLS = [
    "your_label (strong/borderline/weak)",
    "hook 1-5 (stops scrolling in 2s?)",
    "payoff 1-5 (funny/surprising/satisfying?)",
    "context 1-5 (new viewer gets it?)",
    "edit_potential (y/n)",
    "best_trim (e.g. 4-33)",
    "final_call (auto/edit/reject)",
    "notes",
]


def first(d, *keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    for sub in ("result", "judgment", "verdict"):
        if isinstance(d.get(sub), dict):
            v = first(d[sub], *keys)
            if v not in (None, ""):
                return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="pipeline output folders")
    ap.add_argument("--keepers", default=None, help="folder of clips you consider good")
    ap.add_argument("--out", default="eval_set")
    args = ap.parse_args()

    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    rows, seen, skipped = [], set(), []

    for run in args.runs:
        m = re.match(r"(\d{6,})", os.path.basename(run.rstrip("/")))
        vod = m.group(1) if m else os.path.basename(run.rstrip("/"))
        mp4s = {}
        for p in glob.glob(os.path.join(run, "*.mp4")) + \
                glob.glob(os.path.join(run, "visual_approved", "*.mp4")):
            b = os.path.splitext(os.path.basename(p))[0]
            if "proxy" not in b:
                mp4s.setdefault(b, p)
        for jf in sorted(glob.glob(os.path.join(run, "visual_judgments", "*.json"))):
            if jf.endswith("_transcript.json"):
                continue
            jbase = os.path.splitext(os.path.basename(jf))[0]
            try:
                j = json.load(open(jf))
            except Exception as e:
                skipped.append((jf, f"unreadable json: {e}"))
                continue
            if not isinstance(j, dict):
                skipped.append((jf, f"json is a {type(j).__name__}, not an object"))
                continue
            score = first(j, "score", "visual_score", "overall_score")
            if score is None:
                skipped.append((jf, "no score key; keys: " + ",".join(list(j.keys())[:8])))
                continue
            # judgment file may be named <clip>.json or <clip>_<suffix>.json:
            # pick the longest clip name that the judgment name starts with
            matches = [b for b in mp4s if jbase == b or jbase.startswith(b + "_") or jbase.startswith(b + ".")]
            if not matches:
                skipped.append((jf, "no matching clip .mp4 in the run folder"))
                continue
            base = max(matches, key=len)
            if (vod, base) in seen:
                continue
            seen.add((vod, base))
            reason = str(first(j, "reason", "summary", "explanation", "rationale") or "")
            rows.append({
                "id": f"{vod}__{base}", "source": "pipeline", "vod": vod,
                "clip_path": os.path.abspath(mp4s[base]), "judge_score": score,
                "judge_postable": first(j, "postable", "approved", "decision"),
                "judge_reason": reason.replace("\n", " ")[:400],
            })

    if args.keepers and os.path.isdir(args.keepers):
        for clip in sorted(glob.glob(os.path.join(args.keepers, "*.mp4"))):
            base = os.path.splitext(os.path.basename(clip))[0]
            rows.append({"id": f"keeper__{base}", "source": "keepers", "vod": "",
                         "clip_path": os.path.abspath(clip), "judge_score": "",
                         "judge_postable": "", "judge_reason": "(not judged yet)"})

    def sort_key(r):
        s = r["judge_score"]
        return (0 if r["source"] == "keepers" else 1, -(float(s) if s != "" else 99))
    rows.sort(key=sort_key)

    for r in rows:
        tag = f"s{r['judge_score']}" if r["judge_score"] != "" else "keeper"
        link = os.path.join(clips_dir, f"{tag}__{r['id']}.mp4")
        if not os.path.lexists(link):
            os.symlink(r["clip_path"], link)
        r["watch_file"] = os.path.abspath(link)
        for c in LABEL_COLS:
            r[c] = ""

    fields = ["id", "source", "vod", "judge_score", "judge_postable", "judge_reason"] + LABEL_COLS + \
             ["watch_file", "clip_path"]
    out_csv = os.path.join(args.out, "eval_set.csv")
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as fh:
            labelled = any(any((r.get(c) or "").strip() for c in LABEL_COLS) for r in csv.DictReader(fh))
        if labelled:
            raise SystemExit(f"Refusing to overwrite {out_csv}: it already has your labels. "
                             f"Rename it or pass a different --out.")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    sc = Counter(str(r["judge_score"]) for r in rows if r["source"] == "pipeline")
    print(f"clips in sheet: {len(rows)}  "
          f"(pipeline-judged {sum(1 for r in rows if r['source'] == 'pipeline')}, "
          f"keepers {sum(1 for r in rows if r['source'] == 'keepers')})")
    print("judge scores: " + ", ".join(f"{k}:{v}" for k, v in sorted(sc.items(), key=lambda x: -float(x[0]))))
    print(f"saved {out_csv}")
    if skipped:
        print(f"judgment files NOT included: {len(skipped)}")
        for jf, why in skipped[:8]:
            print(f"  {jf}  ->  {why}")
    print(f"clips to watch (sorted by judge score): {clips_dir}/")


if __name__ == "__main__":
    main()
