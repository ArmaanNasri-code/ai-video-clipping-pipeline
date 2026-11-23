#!/usr/bin/env python3
"""
refinish_clips.py -- re-make your queued clips with the current crop + finishing.

For every VOD that has a clip in the post queue you haven't skipped or posted, it
re-runs run_pipeline.py with  --force-crop --force-finish  and every discovery step
switched off (no new candidates, no re-judging, no VOD download), then redoes the
layout choice (apply_layout.py --force). Result: the same clips, same files in the
queue, but with the steady v4 crop, stacked gameplay layout, colour fix and your
clip_settings. Your edited captions and approve/skip choices in the queue are kept.

Costs: a new hook title per clip and a layout check per clip (a few cents in total).

    python3 refinish_clips.py              # show what it would do (changes nothing)
    python3 refinish_clips.py --go         # do it
    python3 refinish_clips.py --go --vod 2933861646    # just one VOD
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

PY = sys.executable


def stream_start(run):
    try:
        return json.load(open(os.path.join(run, "_stream_start.json")))["stream_start"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--vod", action="append", help="only these VOD ids")
    args = ap.parse_args()
    try:
        queue = json.load(open("post_queue/queue.json"))["queue"]
    except Exception:
        sys.exit("ERROR: post_queue/queue.json not found -- run post_queue.py first (from ~/Downloads).")

    vods = {}
    for e in queue:
        if e["status"] in ("skipped", "posted"):
            continue
        vods.setdefault(e["vod"], {"channel": e.get("channel"), "clips": []})["clips"].append(e["clip"])
    if args.vod:
        vods = {v: d for v, d in vods.items() if v in args.vod}
    if not vods:
        sys.exit("Nothing to re-finish (every queued clip is skipped or posted).")

    plan = []
    for vod, d in sorted(vods.items()):
        run = f"{vod}_final_review"
        ss = stream_start(run)
        if not os.path.exists(f"{vod}_transcript.json"):
            print(f"  skip {vod}: no saved transcript (would need the full audio again)")
            continue
        if not os.path.isdir(run) or not ss or not d["channel"]:
            print(f"  skip {vod}: missing run folder / stream start / channel")
            continue
        cmd = [PY, "-u", "run_pipeline.py", "--vod-id", vod, "--stream-start", ss, "--output-dir", run,
               "--channel", d["channel"], "--tracker", "tracklets", "--face-backend", "insight",
               "--finish-verticals", "--internal-cuts", "--smart-trim",
               "--skip-chat-spikes", "--skip-visual-discovery", "--skip-viewer-clips",
               "--force-crop", "--force-finish"]
        plan.append((vod, d, run, cmd))
        print(f"{vod} ({d['channel']}): {len(d['clips'])} queued clip(s) -> " + ", ".join(d["clips"]))

    if not args.go:
        print(f"\n{len(plan)} VOD(s) would be re-finished. Run again with --go to do it. Nothing changed.")
        return

    os.makedirs("refinish_logs", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    for i, (vod, d, run, cmd) in enumerate(plan, 1):
        log = os.path.join("refinish_logs", f"{vod}_{stamp}.log")
        print(f"\n[{i}/{len(plan)}] {vod} ({d['channel']}) -- re-cropping + finishing (log: {log})", flush=True)
        # the transcript is saved, so the full VOD audio isn't needed: an empty stand-in
        # file stops run_pipeline from re-downloading hours of audio
        audio, placeholder = f"{vod}_full_audio.mp4", False
        if not os.path.exists(audio) and os.path.exists(f"{vod}_transcript.json"):
            open(audio, "w").close()
            placeholder = True
        t0 = datetime.datetime.now()
        try:
            with open(log, "w") as fh:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        finally:
            if placeholder and os.path.exists(audio) and os.path.getsize(audio) == 0:
                os.remove(audio)
        mins = (datetime.datetime.now() - t0).seconds // 60
        if r.returncode != 0:
            print(f"  run_pipeline exited {r.returncode} after {mins} min -- see {log}")
            continue
        print(f"  done in {mins} min")
        for ln in open(log, errors="ignore"):
            if "auto-wide:" in ln or "STACKED" in ln or ("[APPROVED]" in ln) or "hook:" in ln:
                print("   ", ln.rstrip()[:150])
        if os.path.exists("apply_layout.py"):
            subprocess.run([PY, "apply_layout.py", "--run-dir", run, "--mode", "auto", "--force"])
    print("\nRe-finished. Refresh the queue with:  python3 post_queue.py   (nothing was uploaded)")


if __name__ == "__main__":
    main()
