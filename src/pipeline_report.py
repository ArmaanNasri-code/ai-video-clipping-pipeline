#!/usr/bin/env python3
"""
pipeline_report.py -- end-of-run report for one run_pipeline.py output folder.

Read-only with respect to the pipeline: it never changes, moves or deletes any
pipeline file. It only writes:
  <output-dir>/_pipeline_report.txt / .json   the report
  <output-dir>/needs_review/                  links to every clip held for review
                                              (review video + report), one folder
                                              to look in instead of many

Every clip is put in exactly one bucket, so nothing can be skipped silently:
  APPROVED            final clip in production_ready
  EDIT_REVIEW         finished, but internal cuts removed over half of it: held in
                      needs_edit_review/ for a human check
  CAPTION_REVIEW      finished, but the caption gate held it for a human
  LOW_PRESENCE        held back: target absent for too much of the clip
  CROP_FAILED         passed the visual check but no vertical video was produced
  FINISH_FAILED       vertical exists, finishing did not complete
  NOT_FINISHED        vertical exists, finishing never ran (e.g. run interrupted)
  NOT_REACHED         the run stopped before this clip got to cropping
  VISUAL_REJECTED     candidate clip the visual judge did not approve
and errors found in the run log are listed at the end.

Usage:
  python3 pipeline_report.py --output-dir <run folder> [--log run.log]
"""
import argparse
import glob
import json
import os
import re
import subprocess
from collections import OrderedDict


def load(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def duration(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def base_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--log", default=None, help="the run's log file (for the error list)")
    args = ap.parse_args()
    out = args.output_dir.rstrip("/")
    if not os.path.isdir(out):
        raise SystemExit(f"ERROR: {out} is not a folder")

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---------- discovered candidates ----------
    candidates = sorted(p for p in glob.glob(os.path.join(out, "*.mp4"))
                        if "proxy" not in os.path.basename(p))
    cand_bases = [base_of(p) for p in candidates]

    approved_dir = os.path.join(out, "visual_approved")
    visual_ran = os.path.isdir(approved_dir)
    approved = sorted(glob.glob(os.path.join(approved_dir, "*.mp4"))) if visual_ran else candidates
    appr_bases = [base_of(p) for p in approved]
    judgments = {base_of(p): load(p, {}) for p in glob.glob(os.path.join(out, "visual_judgments", "*.json"))}

    vdir = os.path.join(approved_dir if visual_ran else out, "vertical")
    presence = load(os.path.join(vdir, "_presence_summary.json"), {}) or {}
    skipped = load(os.path.join(vdir, "_skipped_low_presence.json"), {}) or {}
    skipped_names = set(skipped.keys()) if isinstance(skipped, dict) else \
        {(s.get("source_base") or s.get("clip") or str(s)) if isinstance(s, dict) else str(s) for s in skipped}
    prod_dir = os.path.join(out, "production_ready")
    finished_root = os.path.join(vdir, "finished")

    log_text = open(args.log, errors="replace").read() if args.log and os.path.exists(args.log) else ""

    clips = OrderedDict()
    for b in cand_bases:
        if visual_ran and b not in appr_bases:
            j = judgments.get(b, {})
            clips[b] = {"bucket": "VISUAL_REJECTED",
                        "note": str(j.get("verdict") or j.get("decision") or j.get("reason") or "")[:90]}
    for b in appr_bases:
        info = {"bucket": None, "note": ""}
        vert = os.path.join(vdir, f"{b}_vertical.mp4")
        trim_json = load(os.path.join(vdir, f"{b}_vertical_trimmed.mp4.json"), None)
        fin = os.path.join(finished_root, b)
        marker = os.path.join(fin, "_full_clip_pipeline_complete.json")
        gate = load(os.path.join(fin, "finish", "captions", "caption_gate.json"), None)
        hook = load(os.path.join(fin, "finish", "hook_generated.json"), {}) or {}
        finals = glob.glob(os.path.join(prod_dir, f"*{b}*_final.mp4"))
        reviews = glob.glob(os.path.join(fin, "needs_caption_review", "*", "manifest.json"))
        p = presence.get(b, {}) or {}
        info.update({
            "tracked": p.get("tracked_frac"), "uncertain": p.get("uncertain_frac"),
            "crop_mode": p.get("crop_mode"),
            "absent": p.get("absent_frac"), "empty": p.get("empty_frac"),
            "dead_air_cut_s": (round(trim_json["duration_in"] - trim_json["duration_out"], 1)
                               if trim_json and "duration_out" in trim_json else None),
            "gate": gate.get("status") if gate else None,
            "medium_frac": gate.get("review_fraction") if gate else None,
            "low": gate.get("low_confidence") if gate else None,
            "hook": hook.get("hook") or hook.get("text") or "",
        })
        edit_rev = load(os.path.join(fin, "_edit_review.json"), None)
        held = glob.glob(os.path.join(out, "needs_edit_review", f"*{b}*_final.mp4"))
        if finals:
            info["bucket"] = "APPROVED"
            info["final"] = finals[0]
            info["final_s"] = duration(finals[0])
        elif edit_rev and held:
            info["bucket"] = "EDIT_REVIEW"
            info["final"] = held[0]
            info["final_s"] = duration(held[0])
            info["note"] = f"big edit: {edit_rev.get('removed_pct')}% removed by internal cuts -- check before posting"
        elif reviews:
            info["bucket"] = "CAPTION_REVIEW"
            info["review_dir"] = os.path.dirname(reviews[0])
        elif b in skipped_names or any(b in s for s in skipped_names):
            info["bucket"] = "LOW_PRESENCE"
        elif not os.path.exists(vert) and (not visual_ran or not os.path.isdir(vdir)):
            info["bucket"] = "NOT_REACHED"
            info["note"] = "run stopped before cropping" if "Traceback" in log_text else "cropping never ran"
        elif not os.path.exists(vert):
            info["bucket"] = "CROP_FAILED"
        elif f"FAILED downstream pipeline for {b}" in log_text:
            info["bucket"] = "FINISH_FAILED"
            info["note"] = "downstream caption/finish pipeline exited with an error (see log)"
        elif os.path.exists(marker):
            info["bucket"] = "FINISH_FAILED"
            info["note"] = "marked complete but no final clip and no review folder"
        elif os.path.isdir(fin):
            info["bucket"] = "FINISH_FAILED"
            info["note"] = "finishing started but did not complete"
        else:
            info["bucket"] = "NOT_FINISHED"
        clips[b] = info

    counts = OrderedDict((k, 0) for k in ["APPROVED", "EDIT_REVIEW", "CAPTION_REVIEW", "LOW_PRESENCE", "CROP_FAILED",
                                          "FINISH_FAILED", "NOT_FINISHED", "NOT_REACHED",
                                          "VISUAL_REJECTED"])
    for c in clips.values():
        counts[c["bucket"]] += 1

    # ---------- review folder (links only) ----------
    review_links = os.path.join(out, "needs_review")
    rv = [(b, c) for b, c in clips.items() if c["bucket"] == "CAPTION_REVIEW"]
    if rv:
        for b, c in rv:
            sub = os.path.join(review_links, b)
            os.makedirs(sub, exist_ok=True)
            for src in glob.glob(os.path.join(c["review_dir"], "*")):
                dst = os.path.join(sub, os.path.basename(src))
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(src), dst)

    # ---------- errors from the log ----------
    errors = []
    if args.log and os.path.exists(args.log):
        pat = re.compile(r"Traceback|Error|ERROR|FAILED|failed|could not|PARSE FAILURE|Killed|No space")
        ignore = re.compile(r"cabac decode|error while decoding MB|concealing|warnings\.warn|"
                            r"0 failed|failed\)$|PARSE RETRY|retry parsed|non-existing PPS")
        seen = set()
        for ln in open(args.log, errors="replace"):
            ln = ln.rstrip()
            if pat.search(ln) and not ignore.search(ln):
                key = re.sub(r"\d+", "#", ln)[:160]
                if key not in seen:
                    seen.add(key)
                    errors.append(ln[:240])

    # ---------- print ----------
    say("=" * 72)
    say(f"PIPELINE REPORT  {out}")
    say("=" * 72)
    say(f"candidates discovered:      {len(cand_bases)}")
    say(f"passed visual check:        " + (f"{len(appr_bases)}" if visual_ran else "-  (visual check not reached)"))
    if "Traceback" in log_text:
        say("RUN STOPPED WITH AN ERROR -- see the error list at the bottom")
    say("")
    for k, v in counts.items():
        say(f"  {k:16s} {v}")
    say("")
    say("per clip:")
    for b, c in clips.items():
        if c["bucket"] == "VISUAL_REJECTED":
            say(f"  [{c['bucket']}] {b}  {c['note']}")
            continue
        tr = f"{100 * c['tracked']:.0f}%" if c.get("tracked") is not None else "?"
        unc = f"{100 * c['uncertain']:.0f}%" if c.get("uncertain") is not None else "?"
        mf = f"{100 * c['medium_frac']:.0f}%" if c.get("medium_frac") is not None else "-"
        da = f"{c['dead_air_cut_s']}s" if c.get("dead_air_cut_s") is not None else "-"
        say(f"  [{c['bucket']}] {b}")
        say(f"      crop: {c.get('crop_mode') or 'streamer'} | target tracked {tr}, uncertain {unc} | dead air cut {da} | "
            f"captions: {c.get('gate') or '-'} (medium {mf}, low {c.get('low') if c.get('low') is not None else '-'})")
        if c.get("hook"):
            say(f"      hook: {c['hook']}")
        if c.get("final"):
            say(f"      final: {c['final']}  ({c['final_s']:.1f}s)" if c.get("final_s") else f"      final: {c['final']}")
        if c.get("note"):
            say(f"      note: {c['note']}")
    if rv:
        say("")
        say(f"clips needing review are linked in: {review_links}/")
    say("")
    say(f"errors in log: {len(errors)}" + ("" if args.log else "  (no --log given)"))
    for e in errors[:40]:
        say(f"  {e}")
    say("")
    say("Nothing was uploaded or posted.")

    with open(os.path.join(out, "_pipeline_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "_pipeline_report.json"), "w") as fh:
        json.dump({"counts": counts, "clips": clips, "errors": errors}, fh, indent=2, default=str)
    print(f"\nSaved {os.path.join(out, '_pipeline_report.txt')}")


if __name__ == "__main__":
    main()
