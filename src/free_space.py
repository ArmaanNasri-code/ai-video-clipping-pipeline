#!/usr/bin/env python3
"""
free_space.py -- free disk space by removing clipper working files that are no longer needed.

    python3 free_space.py          # SHOW what would be removed and how much space (changes nothing)
    python3 free_space.py --go     # actually remove them (permanently -- not via the Trash)

Only touches VODs whose run is FINISHED ("PIPELINE COMPLETE" in <vod>_run.log):
  1. rejected landscape clips: the <run>/*.mp4 candidate clips the judge turned down
     (approved ones live in visual_approved/ and are kept, as are their judgments;
     clips with no saved verdict -- e.g. the judge failed on them -- are kept too)
  2. the low-res scan copy <vod>_visual_proxy.mp4
  3. the full VOD audio <vod>_full_audio.mp4 once its transcript is saved (only used for
     transcription; run_pipeline no longer re-downloads it when the transcript exists)
  4. <run>/_duplicates -- second copies of the same moment that dedupe set aside
Plus:
  5. streamer_refs/*/snip_*.mp4 -- video snippets used to BUILD face references; the
     references themselves (target_reference_*.json) and face pictures are kept.
     Only needed again for add_streamer.py --append (it would re-download them).
Never touched: production_ready, needs_review, needs_edit_review, visual_approved,
finished clips, judgments, reports, logs, unfinished runs, and anything of yours.
Run from ~/Downloads.
"""
import glob
import os
import shutil
import sys

GO = "--go" in sys.argv


def finished_vods():
    out = []
    for lp in glob.glob("*_run.log"):
        vid = lp[:-len("_run.log")]
        if not vid.isdigit():
            continue
        with open(lp, errors="ignore") as f:
            if f"PIPELINE COMPLETE for VOD {vid}" in f.read():
                out.append(vid)
    return sorted(out)


def main():
    if not os.path.exists("run_pipeline.py"):
        sys.exit("ERROR: run from ~/Downloads.")
    victims = []  # (path, bytes, why)
    vods = finished_vods()
    for vid in vods:
        run = f"{vid}_final_review"
        if os.path.isdir(run):
            approved = set(os.listdir(os.path.join(run, "visual_approved"))) \
                if os.path.isdir(os.path.join(run, "visual_approved")) else set()
            for fp in glob.glob(os.path.join(run, "*.mp4")):
                stem = os.path.basename(fp)[:-4]
                judged = os.path.exists(os.path.join(run, "visual_judgments", f"{stem}_visual.json"))
                # only clips the judge actually turned down; never-judged ones (e.g. a judge
                # failure) are kept so a re-run can still judge them
                if os.path.basename(fp) not in approved and judged:
                    victims.append((fp, os.path.getsize(fp), f"{vid} rejected clip"))
        px = f"{vid}_visual_proxy.mp4"
        if os.path.exists(px):
            victims.append((px, os.path.getsize(px), f"{vid} scan copy"))
        au, tr = f"{vid}_full_audio.mp4", f"{vid}_transcript.json"
        if os.path.exists(au) and os.path.exists(tr) and os.path.getsize(tr) > 100:
            victims.append((au, os.path.getsize(au), f"{vid} full audio"))
        dup = os.path.join(run, "_duplicates")
        if os.path.isdir(dup):
            for fp in glob.glob(os.path.join(dup, "**", "*"), recursive=True):
                if os.path.isfile(fp):
                    victims.append((fp, os.path.getsize(fp), f"{vid} duplicate copy"))
    for fp in glob.glob(os.path.join("streamer_refs", "*", "snip_*.mp4")):
        victims.append((fp, os.path.getsize(fp), "reference snippet"))

    groups = {}
    for _, b, why in victims:
        key = why.split(" ", 1)[1] if why[0].isdigit() else why
        groups[key] = groups.get(key, 0) + b
    total = sum(b for _, b, _ in victims)
    print(f"finished VODs: {', '.join(vods) or 'none'}")
    for k, b in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {b / 1e9:5.1f} GB")
    print(f"TOTAL: {len(victims)} files, {total / 1e9:.1f} GB")
    if not GO:
        print("\nNothing removed (preview). Run again with --go to remove them.")
        return
    for fp, _, _ in victims:
        try:
            os.remove(fp)
        except OSError as e:
            print(f"  could not remove {fp}: {e}")
    print(f"removed. Free disk now: {shutil.disk_usage('.').free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
