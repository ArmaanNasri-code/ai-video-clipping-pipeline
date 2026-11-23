#!/usr/bin/env python3
"""
apply_layout.py -- layout step for a finished run (split-screen vs full-screen).

Runs AFTER the normal pipeline on a run folder. It never changes or replaces the
normal finished clips.

    python3 apply_layout.py --run-dir <vod>_final_review --mode auto
    python3 apply_layout.py --run-dir <vod>_final_review --mode split
    python3 apply_layout.py --run-dir <vod>_final_review --mode full      (records only)

For every APPROVED clip (production_ready/*_final.mp4):
  full  -> nothing rendered; recommended = the normal clip
  auto  -> Claude classifies (layout_decide.py); PERSON_FOCUSED + safety checks -> split
  split -> no Claude call; split if the safety checks pass
When split is chosen, split_preview.py renders the split clip on the SAME timeline as
the final (smart trims / internal cuts kept), with the clip's own captions (moved to the
split line), hook, watermark and audio; gameplay is muted, picked at random from
backgrounds/ avoiding recently used stretches. Saved as production_ready/<clip>_final_split.mp4.
If classification or rendering fails, or the tracker can't keep the person in a fill
crop, the recommendation falls back to the normal (full-screen) clip.

Report: <run>/_layout_report.json -- per clip: mode, layout chosen, category, confidence,
reasoning, normal + split paths, and "post_file" = the recommended file. A future
auto-poster only needs to read "post_file".
Re-runs reuse decisions and existing split renders (--force redoes both).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import layout_decide as ld

PY = sys.executable


def render_split(final, force=False):
    """Returns (path or None, note)."""
    base = os.path.basename(final)[: -len("_final.mp4")]
    dst = os.path.join(os.path.dirname(final), f"{base}_final_split.mp4")
    if os.path.exists(dst) and not force:
        return dst, "existing split render reused"
    r = subprocess.run([PY, "split_preview.py", "--final", final, "--layout", "smart"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    smart = next((l for l in out.splitlines() if l.startswith("smart:")), "")
    produced = next((l.split()[1] for l in out.splitlines()
                     if l.startswith("saved ") and "_finished.mp4" in l), None)
    if r.returncode != 0 or not produced or not os.path.exists(produced):
        return None, "split render failed: " + (out.strip().splitlines() or ["?"])[-1][:200]
    if "FULL" in smart.split("--")[0]:
        shutil.rmtree(os.path.dirname(produced), ignore_errors=True)
        return None, "tracker couldn't keep the person in a fill crop -> " + smart.split("--", 1)[-1].strip()
    shutil.move(produced, dst)
    shutil.rmtree(os.path.dirname(produced), ignore_errors=True)   # preview working folder
    return dst, smart.replace("smart:", "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["full", "auto", "split"], default="full")
    ap.add_argument("--force", action="store_true", help="re-decide and re-render")
    args = ap.parse_args()

    prod = os.path.join(args.run_dir, "production_ready")
    finals = sorted(f for f in glob.glob(os.path.join(prod, "*_final.mp4")))
    rp = os.path.join(args.run_dir, "_layout_report.json")
    try:
        rep = json.load(open(rp))
    except Exception:
        rep = {}
    decisions = ld.load_report()
    print(f"[layout] {len(finals)} approved clip(s), mode={args.mode}")

    for final in finals:
        key = os.path.basename(final)
        rec = {"mode": args.mode, "normal_file": final, "split_file": None, "post_file": final,
               "layout": "full", "why": "mode full (default): normal clip only"}
        if args.mode != "full":
            try:
                d = ld.decide_clip(final, decisions, force=args.force, ask=(args.mode == "auto"))
            except SystemExit:
                raise
            except Exception as e:
                d = {"layout": "full", "why": f"decision failed ({e.__class__.__name__}) -> full-screen"}
            if d is None:
                d = {"layout": "full", "why": "landscape original not found -> full-screen"}
            rec.update({k: d.get(k) for k in ("category", "confidence", "what_viewer_watches",
                                               "claude_reason", "signals")})
            rec["layout"], rec["why"] = d["layout"], d["why"]
            if d["layout"] == "split":
                path, note = render_split(final, force=args.force)
                rec["render_note"] = note
                if path:
                    rec["split_file"] = rec["post_file"] = path
                else:
                    rec["layout"], rec["why"] = "full", f"split chosen but fell back: {note}"
        rep[key] = rec
        json.dump(rep, open(rp, "w"), indent=2)
        tag = "SPLIT" if rec["post_file"] != final else "FULL "
        print(f"[layout] {tag} {key}: {rec['why']}")
    print(f"[layout] report: {rp}")


if __name__ == "__main__":
    main()
