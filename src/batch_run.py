#!/usr/bin/env python3
"""
batch_run.py -- run the clipper on a list of Twitch streamers' latest VODs.

    caffeinate -i ~/Downloads/clipper_v8_env/bin/python3 -u batch_run.py streamers.txt
    ~/Downloads/clipper_v8_env/bin/python3 batch_run.py streamers.txt --dry-run   # just show the plan

streamers.txt: one Twitch channel login per line (lines starting with # are ignored).

For each streamer:
  - finds their newest finished VODs (skips the stream that is live right now,
    VODs shorter than --min-minutes or longer than --max-hours)
  - skips VODs already fully processed (re-running is safe; half-done VODs resume)
  - new streamers: at the START it runs add_streamer.py for each one and you confirm
    their face (y/n) -- after that it runs unattended; also makes
    streamer_names/<channel>.json with their display name if missing
  - runs run_pipeline.py with the same settings as the Speed/Adin runs, then
    pipeline_report.py
Stops the whole batch when: Anthropic credits run out, disk space drops below
--min-free-gb, or you press Ctrl-C (the current VOD can be resumed later).

At the end: batch_runs/<date_time>/batch_report.txt plus folders of shortcuts
(approved/, needs_review/) pointing at the finished clips, so you can review
everything in one place. Nothing is uploaded or posted. Nothing is deleted.
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

PY = sys.executable
CREDIT_RE = re.compile(r"credit balance is too low|credit balance", re.I)


def log(msg):
    print(f"[batch {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def ytdlp(args, timeout=120):
    return subprocess.run([PY, "-m", "yt_dlp", *args], capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL)


def is_live(channel):
    try:
        r = ytdlp(["--skip-download", "--print", "%(id)s", f"https://www.twitch.tv/{channel}"], 60)
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


def still_recording(v):
    """True if this VOD's end (start + current length) is within 30 min of now."""
    try:
        r = ytdlp(["--skip-download", "--print", "%(timestamp)s",
                   f"https://www.twitch.tv/videos/{v['id']}"], 60)
        ts = int(r.stdout.strip().splitlines()[-1])
        return ts + (v["duration"] or 0) > time.time() - 1800
    except Exception:
        return True


def list_vods(channel, n=15):
    r = ytdlp(["--flat-playlist", "--playlist-end", str(n), "--print",
               "%(id)s|%(duration)s|%(title)s",
               f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time"])
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1:] or "yt-dlp failed")
    vods = []
    for line in r.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        vid = parts[0].strip().lstrip("v")
        if not vid.isdigit():
            continue
        try:
            dur = float(parts[1])
        except ValueError:
            dur = None
        vods.append({"id": vid, "duration": dur, "title": parts[2].strip()})
    return vods


def display_name(channel):
    try:
        r = ytdlp(["--flat-playlist", "--playlist-end", "1", "--print", "%(uploader)s",
                   f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time"], 60)
        name = r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
        return name if name and name != "NA" else channel
    except Exception:
        return channel


def has_reference(channel):
    if os.path.exists(f"target_reference_{channel}.json"):
        return True
    try:
        with open("reference_map.json") as f:
            ref = json.load(f).get(channel)
        return bool(ref) and os.path.exists(ref)
    except Exception:
        return False


def stream_start(vod_id):
    r = ytdlp(["--skip-download", "--print", "%(timestamp)s",
               f"https://www.twitch.tv/videos/{vod_id}"])
    ts = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if not ts.isdigit():
        raise RuntimeError("could not read the stream start time")
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc).isoformat()


def finished(vod_id):
    lp = f"{vod_id}_run.log"
    if not os.path.exists(lp):
        return False
    with open(lp, errors="ignore") as f:
        return f"PIPELINE COMPLETE for VOD {vod_id}" in f.read()


def credits_ok():
    """1-token test call; False only if Anthropic says the credit balance is too low."""
    try:
        from anthropic import Anthropic
        Anthropic().messages.create(model="claude-sonnet-4-6", max_tokens=1,
                                    messages=[{"role": "user", "content": "ok"}])
        return True
    except Exception as e:
        return "credit balance" not in str(e).lower()


def free_gb():
    return shutil.disk_usage(".").free / 1e9


def run_and_tee(cmd, log_path):
    """Run cmd, show its output live and append it to log_path. Returns (code, saw_credit_error)."""
    credit = False
    with open(log_path, "a") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
            if CREDIT_RE.search(line):
                credit = True
        p.wait()
    return p.returncode, credit


def best_quality_from_log(log_path):
    """If the run failed because 1080p doesn't exist, return the best quality the VOD has."""
    try:
        with open(log_path, errors="ignore") as f:
            txt = f.read()[-20000:]
    except OSError:
        return None
    i = txt.rfind("Could not find quality")
    if i < 0:
        return None
    # twitch_adapter only knows these exact resolutions (by name like "720p");
    # anything else -> "chunked" (= the first/highest variant in the manifest)
    known = {"1920x1080": "1080p", "1280x720": "720p", "852x480": "480p",
             "640x360": "360p", "284x160": "160p"}
    best = None
    for m in re.finditer(r'RESOLUTION=(\d+)x(\d+)', txt[i:]):
        h, res = int(m.group(2)), f"{m.group(1)}x{m.group(2)}"
        if best is None or h > best[0]:
            best = (h, res)
    if not best:
        return None
    return known.get(best[1], "chunked")


def parse_report(out_dir):
    rp = os.path.join(out_dir, "_pipeline_report.txt")
    res = {"counts": {}, "clips": []}
    if not os.path.exists(rp):
        return res
    cur = None
    with open(rp, errors="ignore") as f:
        for line in f:
            m = re.match(r"^\s{2}([A-Z_]+)\s+(\d+)\s*$", line)
            if m:
                res["counts"][m.group(1)] = int(m.group(2))
                continue
            m = re.match(r"^\s{2}\[([A-Z_]+)\]\s+(\S+)", line)
            if m:
                cur = {"status": m.group(1), "clip": m.group(2), "hook": "", "final": ""}
                res["clips"].append(cur)
                continue
            if cur:
                m = re.match(r"^\s+hook:\s*(.*)$", line)
                if m:
                    cur["hook"] = m.group(1).strip()
                m = re.match(r"^\s+final:\s*(\S+\.mp4)", line)
                if m:
                    cur["final"] = m.group(1)
    return res


def link(src, folder, name):
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, name)
    if os.path.exists(src) and not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("streamers", help="text file, one Twitch channel login per line")
    ap.add_argument("--vods-per-streamer", type=int, default=1)
    ap.add_argument("--min-minutes", type=float, default=30)
    ap.add_argument("--max-hours", type=float, default=12)
    ap.add_argument("--min-free-gb", type=float, default=5)
    ap.add_argument("--layout", choices=["full", "auto", "split"], default="full",
                    help="full (default) = normal clips only; auto = Claude picks split-screen for "
                         "person-focused clips; split = split wherever safe. Split versions are saved "
                         "NEXT TO the normal clips (*_final_split.mp4), never replacing them.")
    ap.add_argument("--keep-proxies", action="store_true",
                    help="keep each finished VOD's low-res scan copy (default: remove it to save disk)")
    ap.add_argument("--dry-run", action="store_true", help="only show what would run")
    args = ap.parse_args()

    if not os.path.exists("run_pipeline.py"):
        sys.exit("ERROR: run from ~/Downloads (run_pipeline.py not found).")
    with open(args.streamers) as f:
        channels = []
        for line in f:
            c = line.split("#")[0].strip().lower()
            if c and c not in channels:
                channels.append(c)
    if not channels:
        sys.exit("ERROR: no streamers in the file.")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    bdir = os.path.join("batch_runs", stamp)
    rows, notes, stop_reason = [], [], None
    log(f"{len(channels)} streamers: {', '.join(channels)}"
        + ("  (DRY RUN -- nothing is downloaded)" if args.dry_run else ""))

    # ---------- plan ----------
    plan = []
    for ch in channels:
        try:
            vods = list_vods(ch)
        except Exception as e:
            notes.append(f"{ch}: could not list VODs ({e}) -- skipped")
            log(notes[-1])
            continue
        if not vods:
            notes.append(f"{ch}: no VODs found (Kick/YouTube-only or VODs off) -- skipped")
            log(notes[-1])
            continue
        if is_live(ch) and vods and still_recording(vods[0]):
            log(f"{ch}: live right now -- skipping VOD {vods[0]['id']} (still recording)")
            vods = vods[1:]
        picked = 0
        for v in vods:
            if picked >= args.vods_per_streamer:
                break
            mins = (v["duration"] or 0) / 60
            if v["duration"] and mins < args.min_minutes:
                log(f"{ch}: skip {v['id']} ({mins:.0f} min, too short)")
                continue
            if v["duration"] and mins > args.max_hours * 60:
                log(f"{ch}: skip {v['id']} ({mins / 60:.1f} h, longer than --max-hours {args.max_hours:g})")
                continue
            picked += 1
            done = finished(v["id"])
            plan.append({"channel": ch, **v, "done": done})
            log(f"{ch}: {v['id']} ({mins / 60:.1f} h) {v['title'][:60]!r}"
                + ("  -> already processed, will just collect it" if done else ""))
        if picked == 0:
            notes.append(f"{ch}: no suitable VOD in the last 15")

    todo = [p for p in plan if not p["done"]]
    hrs = sum((p["duration"] or 0) for p in todo) / 3600
    log(f"plan: {len(todo)} VOD(s) to process ({hrs:.1f} h of stream), "
        f"{len(plan) - len(todo)} already done. Free disk: {free_gb():.0f} GB")
    if args.dry_run:
        new = sorted({p["channel"] for p in todo if not has_reference(p["channel"])})
        if new:
            log(f"new streamers (you'll be asked to confirm their face at the start): {', '.join(new)}")
        return

    if todo and not credits_ok():
        sys.exit("STOPPED: Anthropic API credits are used up. Top up at console.anthropic.com "
                 "(Plans & Billing), then run this again. Nothing was started.")

    # ---------- setup new streamers FIRST, while you're at the keyboard ----------
    new = []
    for p in todo:
        if p["channel"] not in new and not has_reference(p["channel"]):
            new.append(p["channel"])
    if new:
        log(f"SETUP: {len(new)} new streamer(s): {', '.join(new)}. For each, a picture of a face opens --")
        log("answer y if it's the streamer, n for the next person. After this you can walk away.")
        for ch in new:
            try:
                subprocess.run([PY, "-u", "add_streamer.py", "--channel", ch])
            except KeyboardInterrupt:
                log(f"{ch}: setup skipped")
            if has_reference(ch):
                notes.append(f"{ch}: NEW face reference -- check its crops closely")
            else:
                notes.append(f"{ch}: no face reference (not confirmed) -- skipped")
        log("SETUP done -- the rest runs on its own. Nothing else will ask you anything.")

    # ---------- run ----------
    try:
        for p in plan:
            ch, vid = p["channel"], p["id"]
            out = f"{vid}_final_review"
            if not p["done"]:
                if free_gb() < args.min_free_gb:
                    stop_reason = (f"disk space low ({free_gb():.1f} GB free, need {args.min_free_gb:g}). "
                                   "Empty the Trash or move old run folders to an external drive.")
                    break
                if not has_reference(ch):
                    continue
                if not credits_ok():
                    stop_reason = ("Anthropic API credits are used up (checked before starting the next VOD). "
                                   "Top up at console.anthropic.com (Plans & Billing), then run the same "
                                   "batch command again.")
                    break
                nf = os.path.join("streamer_names", f"{ch}.json")
                if not os.path.exists(nf):
                    os.makedirs("streamer_names", exist_ok=True)
                    dn = display_name(ch)
                    names = [dn] if dn.lower() == ch else [dn, ch]
                    with open(nf, "w") as f:
                        json.dump({"names": names, "fix": {}}, f)
                    log(f"{ch}: created {nf} with names {names}")
                try:
                    start = stream_start(vid)
                except Exception as e:
                    notes.append(f"{ch} {vid}: {e} -- skipped")
                    log(notes[-1])
                    continue
                log(f"=== {ch} VOD {vid} (stream start {start}) ===")
                t0 = time.time()
                base = [PY, "-u", "run_pipeline.py", "--vod-id", vid, "--stream-start", start,
                        "--output-dir", out, "--channel", ch, "--tracker", "tracklets",
                        "--face-backend", "insight", "--finish-verticals", "--internal-cuts", "--smart-trim"]
                code, credit = run_and_tee(base, f"{vid}_run.log")
                alt = best_quality_from_log(f"{vid}_run.log") if code != 0 and not credit else None
                if alt:
                    log(f"{ch} {vid}: no 1080p for this VOD -- retrying at {alt}")
                    code, credit = run_and_tee(base + ["--quality", alt], f"{vid}_run.log")
                log(f"{ch} {vid}: run_pipeline exit {code} after {(time.time() - t0) / 60:.0f} min")
                if os.path.isdir(out):
                    run_and_tee([PY, "pipeline_report.py", "--output-dir", out, "--log", f"{vid}_run.log"],
                                os.devnull)
                px = f"{vid}_visual_proxy.mp4"
                if args.keep_proxies is False and finished(vid) and os.path.exists(px):
                    gb = os.path.getsize(px) / 1e9
                    os.remove(px)
                    log(f"{ch} {vid}: removed {px} ({gb:.1f} GB low-res scan copy; VOD is finished)")
                if finished(vid) and os.path.exists("free_space.py"):
                    r = subprocess.run([PY, "free_space.py", "--go"], capture_output=True, text=True,
                                       stdin=subprocess.DEVNULL)
                    tail = (r.stdout.strip().splitlines() or [""])[-1]
                    log(f"cleanup of finished VODs' working files: {tail}")
                if credit:
                    stop_reason = ("Anthropic API credits ran out. Top up at console.anthropic.com "
                                   "(Plans & Billing), then run the same batch command again -- "
                                   "finished work is kept and it continues where it stopped.")
                if code != 0 and not credit:
                    notes.append(f"{ch} {vid}: run_pipeline stopped with an error (see {vid}_run.log); "
                                 "re-running the batch resumes it")
            if args.layout != "full" and os.path.isdir(os.path.join(out, "production_ready")) \
                    and os.path.exists("apply_layout.py"):
                code_l, credit_l = run_and_tee([PY, "-u", "apply_layout.py", "--run-dir", out,
                                                "--mode", args.layout], f"{vid}_run.log")
                if credit_l:
                    stop_reason = ("Anthropic API credits ran out during the layout step. Top up, then run "
                                   "the same batch command again.")
            rep = parse_report(out)
            try:
                lrep = json.load(open(os.path.join(out, "_layout_report.json")))
            except Exception:
                lrep = {}
            for k, r in lrep.items():
                if r.get("split_file") and os.path.exists(r["split_file"]):
                    tag = "RECOMMENDED_" if r.get("post_file") == r["split_file"] else ""
                    link(r["split_file"], os.path.join(bdir, "approved"),
                         f"{ch}_{tag}{os.path.basename(r['split_file'])}")
            n_split = sum(1 for r in lrep.values() if r.get("post_file") == r.get("split_file") and r.get("split_file"))
            if lrep:
                notes.append(f"{ch} {vid}: layout {args.layout} -> {n_split} split-screen recommended, "
                             f"{len(lrep) - n_split} full-screen (see {out}/_layout_report.json)")
            rows.append({"channel": ch, "vod": vid, "title": p["title"], **rep})
            for c in rep["clips"]:
                if c["final"]:
                    name = f"{ch}_{os.path.basename(c['final'])}"
                    link(c["final"], os.path.join(bdir, "approved"), name)
            for sub in ("needs_review", "needs_edit_review"):
                for fp in glob.glob(os.path.join(out, sub, "**", "*.mp4"), recursive=True):
                    link(fp, os.path.join(bdir, "needs_review"), f"{ch}_{vid}_{os.path.basename(fp)}")
            if stop_reason:
                break
    except KeyboardInterrupt:
        stop_reason = "you pressed Ctrl-C. Run the same command again to continue."

    # ---------- report ----------
    os.makedirs(bdir, exist_ok=True)
    L = ["=" * 72, f"BATCH REPORT  {stamp}", "=" * 72]
    tot = {}
    for r in rows:
        for k, v in r["counts"].items():
            tot[k] = tot.get(k, 0) + v
        approved = [c for c in r["clips"] if c["status"] == "APPROVED"]
        c = r["counts"]
        L.append(f"\n{r['channel']}  VOD {r['vod']}  {r['title'][:50]!r}")
        L.append(f"  approved {c.get('APPROVED', 0)} | edit review {c.get('EDIT_REVIEW', 0)} | "
                 f"caption review {c.get('CAPTION_REVIEW', 0)} | rejected {c.get('VISUAL_REJECTED', 0)} | "
                 f"failed {c.get('FINISH_FAILED', 0) + c.get('CROP_FAILED', 0)}")
        for a in approved:
            L.append(f"    + {a['hook'] or a['clip']}")
    L.append("\nTOTAL: " + ", ".join(f"{k} {v}" for k, v in sorted(tot.items())) if tot else "\nTOTAL: nothing finished")
    if notes:
        L.append("\nNOTES:")
        L += [f"  - {n}" for n in notes]
    if stop_reason:
        L.append(f"\nSTOPPED EARLY: {stop_reason}")
    L.append(f"\nReview folder: {os.path.abspath(bdir)}  (approved/ and needs_review/ hold shortcuts)")
    L.append("Nothing was uploaded or posted.")
    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(bdir, "batch_report.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
