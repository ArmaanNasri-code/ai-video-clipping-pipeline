#!/usr/bin/env python3
"""
run_pipeline.py

Consolidated end-to-end chuzzzclips pipeline for a single Twitch VOD.
Replaces the manual relay of:
    fetch_full_vod_audio_kai.py -> transcribe_groq.py ->
    score_transcript_clips.py -> merge_candidates_kai_v5.py ->
    silence_check_kai.py -> refetch_final_clips_v2.py

Each stage writes a file named after the VOD ID, and each stage is
SKIPPED if its output file already exists -- so re-running this script
after an interruption (or to tweak a later stage) picks up where it
left off instead of redoing expensive work. Use --force-<stage> to
force a specific stage to re-run, or --force-all to redo everything.

Requires (same as before): ANTHROPIC_API_KEY, GROQ_API_KEY env vars,
and TwitchAdapter's helix_client_id/helix_client_secret if you want
get_stream_start_time() to auto-fetch the stream start instead of you
passing --stream-start manually.

Usage:
    python3 run_pipeline.py --vod-id 2837447163 \\
        --stream-start "2026-08-04T23:39:01.405+00:00"

    # or, with Helix creds set as env vars TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET:
    python3 run_pipeline.py --vod-id 2837447163

Files produced (relative to cwd), all namespaced by --vod-id:
    {vod}_full_audio.mp4
    {vod}_transcript.json
    {vod}_transcript_scores.json   (written incrementally by score_transcript_clips.py)
    {vod}_merged.json
    {vod}_final.json               (post silence-check)
    {vod}_final_review/            (final re-fetched clips, ready for visual review)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.parse
import shutil
from datetime import datetime, timedelta, timezone

from twitch_adapter import TwitchAdapter
from visual_discovery_stage import stage_visual_discovery

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def log(stage, msg):
    print(f"[{stage}] {msg}")


def cleanup_leftover_segments():
    for f in glob.glob("twitch_seg_*.ts"):
        try:
            os.remove(f)
        except OSError:
            pass


def rewrite_manifest_with_absolute_urls(sub_manifest_path, base_url, output_path):
    """ffmpeg needs absolute URLs when reading a local .m3u8 -- Twitch
    sub-manifests use relative segment paths ('0.ts'), so resolve them
    against the CDN base_url before handing the file to ffmpeg."""
    with open(sub_manifest_path) as f:
        lines = f.readlines()
    rewritten = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("http"):
            rewritten.append(urllib.parse.urljoin(base_url, stripped) + "\n")
        else:
            rewritten.append(line)
    with open(output_path, "w") as f:
        f.writelines(rewritten)


def get_manifest(adapter, vod_id, quality="1080p"):
    """Fetch master + quality sub-manifest once; reused across every
    stage that needs to pull segments (silence-check, refetch)."""
    master_path = adapter.get_vod_manifest(vod_id)
    sub_path = adapter.get_quality_submanifest(master_path, quality=quality)
    base_url = getattr(adapter, "_last_manifest_base_url", None)
    if not base_url:
        raise RuntimeError(
            "TwitchAdapter._last_manifest_base_url not set after "
            "get_quality_submanifest() -- check the adapter before proceeding."
        )
    return sub_path, base_url


def cut_local_video_window(
    local_video,
    start_sec,
    duration_sec,
    output_path,
    audio_only=False,
):
    cmd = [
        "ffmpeg",
        "-ss", str(max(0, float(start_sec))),
        "-i", local_video,
        "-t", str(max(0.1, float(duration_sec))),
    ]

    if audio_only:
        cmd += [
            "-vn",
            "-acodec", "aac",
        ]
    else:
        cmd += [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-c:a", "aac",
        ]

    cmd += [
        "-y",
        output_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(
            f"ffmpeg local-video cut failed: {output_path}"
        )

    return output_path


# --------------------------------------------------------------------------
# Automatic target reference (tracklets mode)
# --------------------------------------------------------------------------

def fetch_vod_channel(adapter, vod_id):
    """Channel login that owns a Twitch VOD, from the same Helix /videos
    endpoint (and credentials) get_stream_start_time() already uses."""
    import requests
    token = adapter._get_helix_app_token()
    r = requests.get(
        "https://api.twitch.tv/helix/videos",
        params={"id": vod_id},
        headers={"Client-Id": adapter._helix_client_id, "Authorization": f"Bearer {token}"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data:
        raise RuntimeError(f"Helix returned no video for id {vod_id}")
    return str(data[0]["user_login"]).lower()


def resolve_reference(args, adapter, vod_id, output_dir, local_video):
    """--reference-file always wins. Otherwise, in tracklets mode only, find the
    channel (--channel, a per-VOD cache, or Helix) and map it to a reference via
    reference_map.json, then target_reference_<channel>.json. Returns a path or
    None (crop is skipped, exactly as when no --reference-file is given)."""
    if args.reference_file:
        return args.reference_file
    if args.tracker != "tracklets":
        return None

    channel = (args.channel or "").strip().lower() or None
    cache = os.path.join(output_dir, "_channel.json")
    if not channel and os.path.exists(cache):
        try:
            with open(cache) as f:
                cached = json.load(f)
            if str(cached.get("vod_id")) == str(vod_id):
                channel = cached.get("channel")
        except Exception:
            pass
    if not channel and not local_video:
        try:
            channel = fetch_vod_channel(adapter, vod_id)
            os.makedirs(output_dir, exist_ok=True)
            with open(cache, "w") as f:
                json.dump({"vod_id": str(vod_id), "channel": channel}, f, indent=2)
        except Exception as e:
            log("reference", f"could not look up the VOD's channel: {e}")
    if not channel:
        log("reference", "no channel known (pass --channel for --local-video runs) -- "
                         "auto-crop will be skipped")
        return None

    candidates = []
    if os.path.exists(args.reference_map):
        try:
            with open(args.reference_map) as f:
                mapping = {str(k).lower(): v for k, v in json.load(f).items()}
            if channel in mapping:
                candidates.append(mapping[channel])
        except Exception as e:
            log("reference", f"could not read {args.reference_map}: {e}")
    candidates.append(f"target_reference_{channel}.json")

    for c in candidates:
        if os.path.exists(c):
            log("reference", f"channel '{channel}' -> {c}")
            return c
    log("reference", f"no reference for channel '{channel}' (looked for: {', '.join(candidates)}). "
                     f"Build one with build_target_reference.py and add it to {args.reference_map}; "
                     f"auto-crop will be skipped")
    return None


# --------------------------------------------------------------------------
# Stage 1: fetch full VOD audio
# --------------------------------------------------------------------------

def stage_fetch_audio(
    adapter,
    vod_id,
    quality,
    force,
    local_video=None,
):
    audio_path = f"{vod_id}_full_audio.mp4"
    _tr = f"{vod_id}_transcript.json"
    if not force and os.path.exists(_tr) and os.path.getsize(_tr) > 100:
        log("fetch", f"{_tr} exists -- full audio not needed, skipping download")
        return audio_path
    if os.path.exists(audio_path) and not force:
        _ok = False
        try:
            _r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", audio_path], capture_output=True, text=True)
            _ok = float(_r.stdout.strip() or 0) > 60
        except Exception:
            _ok = False
        if _ok:
            log("fetch", f"{audio_path} already exists, skipping (--force-fetch to redo)")
            return audio_path
        log("fetch", f"{audio_path} is incomplete or unreadable -- deleting and downloading again")
        os.remove(audio_path)

    if local_video:
        log("fetch", f"extracting full audio from local video: {local_video}")
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", local_video,
                "-vn",
                "-acodec", "aac",
                "-y",
                audio_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError(
                "ffmpeg local audio extraction failed -- see stderr above."
            )
        log("fetch", f"saved {audio_path}")
        return audio_path

    log("fetch", "fetching master + quality sub-manifest...")
    sub_path, base_url = get_manifest(adapter, vod_id, quality)

    rewritten_manifest = f"{vod_id}_absolute.m3u8"
    rewrite_manifest_with_absolute_urls(sub_path, base_url, rewritten_manifest)

    log("fetch", "downloading + extracting full VOD audio via ffmpeg (long-running)...")
    cmd = [
        "ffmpeg",
        "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
        "-i", rewritten_manifest,
        "-vn",
        "-acodec", "aac",
        "-y",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("ffmpeg audio fetch failed -- see stderr above.")

    log("fetch", f"saved {audio_path}")
    return audio_path


# --------------------------------------------------------------------------
# Stage 2: transcribe (delegates to transcribe_groq.py, reused as-is)
# --------------------------------------------------------------------------

def stage_transcribe(vod_id, audio_path, force):
    transcript_path = f"{vod_id}_transcript.json"
    if os.path.exists(transcript_path) and not force:
        log("transcribe", f"{transcript_path} already exists, skipping (--force-transcribe to redo)")
        return transcript_path

    log("transcribe", "running transcribe_groq.py...")
    r = subprocess.run(
        [sys.executable, "transcribe_groq.py", audio_path, transcript_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError("transcription failed -- see stderr above.")
    log("transcribe", f"saved {transcript_path}")
    return transcript_path


# --------------------------------------------------------------------------
# Stage 3: score (delegates to score_transcript_clips.py, which is already
# self-resuming based on window_start values already present in its output)
# --------------------------------------------------------------------------

def stage_score(transcript_path, window_seconds, step_seconds, force):
    scores_path = transcript_path.replace(".json", "_scores.json")
    if os.path.exists(scores_path):
        if not force:
            log("score", f"{scores_path} already exists, skipping (--force-score to redo/resume)")
            return scores_path
        # score_transcript_clips.py has its OWN internal resume logic that
        # just checks whether this file already exists on disk -- our force
        # flag means "redo scoring," which requires actually removing the
        # stale file, or it'll silently "resume" by skipping every window
        # already in it (e.g. all-error windows from an invalid API key).
        log("score", f"--force-score set: removing stale {scores_path} before rescoring")
        os.remove(scores_path)

    log("score", "running score_transcript_clips.py (resumable if interrupted)...")
    r = subprocess.run(
        [sys.executable, "score_transcript_clips.py", transcript_path,
         str(window_seconds), str(step_seconds)],
        capture_output=True, text=True
    )
    # score_transcript_clips.py streams progress to stdout as it scores each
    # window and writes incrementally, so even a non-zero exit here may have
    # left useful partial output -- surface stderr but don't discard the file.
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        if not os.path.exists(scores_path):
            raise RuntimeError("scoring failed with no output file produced.")
        log("score", "scoring exited non-zero but a scores file exists -- continuing with partial results.")
    log("score", f"saved {scores_path}")
    return scores_path


# --------------------------------------------------------------------------
# Stage 4: merge (logic ported from merge_candidates_kai_v5.py, generalized)
# --------------------------------------------------------------------------

def stage_merge(vod_id, scores_path, min_score, padding, max_clip_duration, force):
    merged_path = f"{vod_id}_merged.json"
    if os.path.exists(merged_path) and not force:
        log("merge", f"{merged_path} already exists, skipping (--force-merge to redo)")
        return merged_path

    with open(scores_path) as f:
        data = json.load(f)

    qualifying = sorted(
        [
            d
            for d in data
            if (
                d["score"] >= min_score
                and d.get("postable", True)
            )
        ],
        key=lambda d: d["window_start"]
    )
    log("merge", f"windows scoring >= {min_score}: {len(qualifying)}")

    intervals = []
    for d in qualifying:
        intervals.append({
            "start": max(0, d["window_start"] - padding),
            "end": d["window_end"] + padding,
            "windows": [d],
        })
    intervals.sort(key=lambda x: x["start"])

    merged = []
    for iv in intervals:
        if merged and iv["start"] <= merged[-1]["end"]:
            candidate_end = max(merged[-1]["end"], iv["end"])
            candidate_duration = candidate_end - merged[-1]["start"]
            if candidate_duration <= max_clip_duration:
                merged[-1]["end"] = candidate_end
                merged[-1]["windows"].extend(iv["windows"])
                continue
        merged.append(iv)

    # Hard trim pass: guarantee zero overlaps even where the duration cap
    # blocked a full merge above.
    merged.sort(key=lambda x: x["start"])
    for i in range(len(merged) - 1):
        if merged[i]["end"] > merged[i + 1]["start"]:
            trimmed_end = merged[i + 1]["start"]
            log("merge", f"trimming clip at {merged[i]['start']:.0f}s to eliminate overlap")
            merged[i]["end"] = trimmed_end

    final_candidates = []
    for m in merged:
        if m["end"] <= m["start"]:
            continue
        best = max(m["windows"], key=lambda d: d["score"])
        reasons = " | ".join(f"[{d['score']}] {d['reason']}" for d in m["windows"])
        final_candidates.append({
            "merged_start": m["start"],
            "merged_end": m["end"],
            "duration": m["end"] - m["start"],
            "top_score": best["score"],
            "top_reason": best["reason"],
            "all_reasons": reasons if len(m["windows"]) > 1 else best["reason"],
            "num_windows_merged": len(m["windows"]),
        })

    final_candidates.sort(key=lambda c: -c["top_score"])

    with open(merged_path, "w") as f:
        json.dump(final_candidates, f, indent=2)

    log("merge", f"saved {len(final_candidates)} candidates to {merged_path}")
    return merged_path


# --------------------------------------------------------------------------
# Stage 5: silence check (logic ported from silence_check_kai.py, generalized)
# --------------------------------------------------------------------------

def check_audio_energy(clip_path):
    cmd = [
        "ffmpeg", "-i", clip_path,
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return None
    import numpy as np
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64)
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def stage_silence_check(
    vod_id,
    adapter,
    merged_path,
    stream_start,
    min_score,
    rms_threshold,
    quality,
    force,
    local_video=None,
):
    final_path = f"{vod_id}_final.json"
    if os.path.exists(final_path) and not force:
        log("silence", f"{final_path} already exists, skipping (--force-silence to redo)")
        return final_path

    with open(merged_path) as f:
        merged = json.load(f)

    candidates = [m for m in merged if m["top_score"] >= min_score]
    log("silence", f"checking audio energy for {len(candidates)} candidates (score >= {min_score})...")

    sub_path = None
    if not local_video:
        sub_path, _ = get_manifest(adapter, vod_id, quality)

    results = []
    for i, cand in enumerate(candidates):
        start_sec = int(cand["merged_start"])
        m, s = divmod(start_sec, 60)
        log("silence", f"[{i+1}/{len(candidates)}] {m:02d}m{s:02d}s (score={cand['top_score']}) -- checking...")

        cleanup_leftover_segments()
        window_seconds = max(30, int(cand["duration"]))

        try:
            if local_video:
                clip_path = f"_local_silence_{i:03d}.mp4"
                cut_local_video_window(
                    local_video,
                    start_sec,
                    window_seconds,
                    clip_path,
                )
            else:
                target = stream_start + timedelta(seconds=start_sec)
                clip_path = adapter.fetch_segment_window(
                    sub_path,
                    target,
                    window_seconds=window_seconds,
                )
        except Exception as e:
            log("silence", f"  SKIP (fetch failed): {e}")
            continue

        rms = check_audio_energy(clip_path)
        flagged_silent = rms is not None and rms < rms_threshold

        cand["audio_rms"] = rms
        cand["flagged_silent"] = bool(flagged_silent)
        cand["clip_path"] = clip_path

        if flagged_silent:
            log("silence", f"  FLAGGED silent (RMS={rms:.1f}) -- excluding")
        else:
            log("silence", f"  OK (RMS={rms:.1f})" if rms is not None else "  WARNING: could not measure audio")

        results.append(cand)

    passed = [r for r in results if not r.get("flagged_silent")]
    silenced = [r for r in results if r.get("flagged_silent")]

    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)

    log("silence", f"{len(passed)} passed, {len(silenced)} flagged silent and excluded")
    return final_path


# --------------------------------------------------------------------------
# Stage 6: refetch final clips at corrected midpoint/half-width math
# (ported from refetch_final_clips_v2.py, generalized)
# --------------------------------------------------------------------------

def stage_auto_crop(output_dir, reference_path, force, tracker="legacy", face_backend="insight",
                    crop_mode="auto"):
    """tracker="legacy"    -> auto_track_target.py + render_vertical_crop.py (unchanged)
    tracker="tracklets" -> track_by_tracklets.py + render_vertical_crop_v3.py
    Both write the same {clip}_track.json / {clip}_vertical.mp4 names, so the
    finishing stage works with either.

    Legacy behaviour: Runs the automated single-target reframing (auto_track_target.py +
    render_vertical_crop.py) against every clip in output_dir, producing
    a 9:16 vertical version of each. Skips clips that already have a
    vertical output unless force is set. A clip the tracker can't
    confidently seed (no match to the reference anywhere in its first
    ~150 frames) is left as-is and flagged -- this is the manual escape
    hatch: build_target_reference.py's per-clip seeding tools still work
    standalone for anything that lands here."""
    if not reference_path:
        log("crop", "no --reference-file given, skipping auto-crop stage entirely")
        return []

    if not os.path.exists(reference_path):
        raise RuntimeError(f"reference file {reference_path} not found -- run build_target_reference.py first.")

    vertical_dir = os.path.join(output_dir, "vertical")
    os.makedirs(vertical_dir, exist_ok=True)

    if tracker == "tracklets":
        return stage_auto_crop_tracklets(output_dir, vertical_dir, reference_path, force, face_backend,
                                         crop_mode)

    clips = sorted(glob.glob(os.path.join(output_dir, "*.mp4")))
    failed = []

    for clip_path in clips:
        base = os.path.splitext(os.path.basename(clip_path))[0]
        track_path = os.path.join(vertical_dir, f"{base}_track.json")
        vertical_path = os.path.join(vertical_dir, f"{base}_vertical.mp4")

        if os.path.exists(vertical_path) and not force:
            log("crop", f"{vertical_path} already exists, skipping (--force-crop to redo)")
            continue

        log("crop", f"tracking {base}...")
        r = subprocess.run(
            [sys.executable, "auto_track_target.py", "--clip", clip_path,
             "--reference", reference_path, "--output", track_path],
            capture_output=True, text=True
        )
        print(r.stdout[-1500:])
        if r.returncode != 0 or not os.path.exists(track_path):
            print(r.stderr[-1500:])
            log("crop", f"  FAILED to track {base} -- likely no confident match to reference in this clip. "
                         f"Seed it manually with pick_face_id_manual.py / build_target_reference.py if needed.")
            failed.append(base)
            continue

        log("crop", f"rendering vertical crop for {base}...")
        r = subprocess.run(
            [sys.executable, "render_vertical_crop.py", "--clip", clip_path,
             "--track", track_path, "--output", vertical_path],
            capture_output=True, text=True
        )
        print(r.stdout[-800:])
        if r.returncode != 0:
            print(r.stderr[-1500:])
            log("crop", f"  FAILED to render {base}")
            failed.append(base)
            continue

        log("crop", f"  saved {vertical_path}")

    if failed:
        log("crop", f"{len(failed)} clip(s) could not be auto-cropped and need manual review: {', '.join(failed)}")
    log("crop", f"done -- vertical clips in ./{vertical_dir}/")
    return failed



def target_height_frac(decisions_path, yolo_cache, clip_path):
    """Median height of the streamer's person box as a share of the frame height,
    from the tracklets the streamer tracker accepted. None if never found."""
    import cv2
    import numpy as np
    try:
        dec = json.load(open(decisions_path))
        tracks = json.load(open(yolo_cache))
        cap = cv2.VideoCapture(clip_path)
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
        cap.release()
    except Exception:
        return None
    fps = float(dec.get("fps") or 30)
    hs = []
    for t in dec.get("tracklets", []):
        if not str(t.get("decision", "")).startswith("TARGET"):
            continue
        a, b = int(round(t["start_s"] * fps)), int(round(t["end_s"] * fps))
        for fr in range(a, b + 1, 3):
            for bx in tracks.get(str(fr), []):
                if int(bx[0]) == int(t["tid"]):
                    hs.append((float(bx[4]) - float(bx[2])) / H)
    return float(np.median(hs)) if hs else None


def stage_auto_crop_tracklets(output_dir, vertical_dir, reference_path, force, face_backend="insight",
                              crop_mode="auto"):
    """New crop path (--tracker tracklets).

    Per clip, in {output_dir}/vertical/:
      {base}_yolo_tracks.json   YOLO+ByteTrack cache (reused on re-runs)
      {base}_track_evidence.npz face/body evidence cache (reused on re-runs)
      {base}_track.json         per-frame pos/lost/shot/state (same name as legacy)
      {base}_track_decisions.json  why each tracklet was accepted/rejected + presence
      {base}_vertical.mp4       final 9:16 render (same name as legacy)
    plus _presence_summary.json for every clip, for the later clip-selection
    presence penalty. The detection caches are kept on --force-crop because
    they depend only on the clip, not on tracker settings.
    """
    clips = sorted(glob.glob(os.path.join(output_dir, "*.mp4")))
    failed = []
    summary_path = os.path.join(vertical_dir, "_presence_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    tracker_module = os.path.abspath("auto_track_target.py")

    for clip_path in clips:
        base = os.path.splitext(os.path.basename(clip_path))[0]
        track_path = os.path.join(vertical_dir, f"{base}_track.json")
        decisions_path = os.path.join(vertical_dir, f"{base}_track_decisions.json")
        yolo_cache = os.path.join(vertical_dir, f"{base}_yolo_tracks.json")
        vertical_path = os.path.join(vertical_dir, f"{base}_vertical.mp4")

        if os.path.exists(vertical_path) and not force:
            log("crop", f"{vertical_path} already exists, skipping (--force-crop to redo)")
            continue

        # both new scripts refuse to overwrite, so clear old outputs on a re-run
        action_track = os.path.join(vertical_dir, f"{base}_track_action.json")
        trimmed = os.path.join(vertical_dir, f"{base}_vertical_trimmed.mp4")
        for stale in (track_path, decisions_path, vertical_path, action_track, trimmed, trimmed + ".json"):
            if os.path.exists(stale):
                os.remove(stale)

        log("crop", f"tracking {base} (tracklets, {face_backend} faces)...")
        r = subprocess.run(
            [sys.executable, "track_by_tracklets.py", "--clip", clip_path,
             "--reference", reference_path, "--tracks-cache", yolo_cache,
             "--tracker", tracker_module, "--face-backend", face_backend,
             "--output", track_path],
            capture_output=True, text=True
        )
        print(r.stdout[-2500:])
        if r.returncode != 0 or not os.path.exists(track_path):
            print(r.stderr[-1500:])
            log("crop", f"  FAILED to track {base}")
            failed.append(base)
            continue

        presence = None
        try:
            with open(decisions_path) as f:
                presence = json.load(f).get("presence")
        except Exception as e:
            log("crop", f"  warning: could not read {decisions_path}: {e}")
        if presence:
            summary[base] = {k: presence.get(k, 0.0) for k in
                             ("tracked_frac", "sole_subject_frac", "uncertain_frac", "absent_frac", "empty_frac")}
            log("crop", f"  target tracked {100 * presence['tracked_frac']:.0f}%, "
                        f"uncertain {100 * presence['uncertain_frac']:.0f}%, "
                        f"absent {100 * presence['absent_frac']:.0f}%, "
                        f"nobody on screen {100 * presence.get('empty_frac', 0):.0f}%")
            if presence["tracked_frac"] < 0.2:
                log("crop", f"  NOTE: target on screen for under 20% of {base} -- "
                            f"will render mostly full-frame")

        mode = crop_mode
        th = target_height_frac(decisions_path, yolo_cache, clip_path)
        if mode == "auto":
            tf = presence["tracked_frac"] if presence else 0.0
            mode = "action" if (th is None or tf < 0.25 or th < 0.18) else "streamer"
        render_track = track_path
        if mode == "action":
            why = ("streamer not found" if th is None else
                   f"streamer small ({100 * th:.0f}% of frame height)" if th < 0.18 else
                   f"streamer tracked only {100 * (presence or {}).get('tracked_frac', 0):.0f}%")
            log("crop", f"  crop mode: ACTION ({why}) -- framing the main people on screen instead")
            r = subprocess.run(
                [sys.executable, "track_action.py", "--clip", clip_path, "--tracks-cache", yolo_cache,
                 "--decisions", decisions_path, "--output", action_track],
                capture_output=True, text=True
            )
            print(r.stdout[-800:])
            if r.returncode == 0 and os.path.exists(action_track):
                render_track = action_track
            else:
                print(r.stderr[-1500:])
                log("crop", "  action track FAILED -- falling back to the streamer crop")
                mode = "streamer"
        else:
            log("crop", f"  crop mode: streamer"
                        + (f" ({100 * th:.0f}% of frame height)" if th is not None else ""))
        summary.setdefault(base, {})
        summary[base]["crop_mode"] = mode
        summary[base]["target_height_frac"] = None if th is None else round(th, 3)

        try:
            _steady = bool(json.load(open("clip_settings.json")).get("steady_crop", True))
        except Exception:
            _steady = True
        r = None
        _auto_wide = mode == "action" or (th is not None and th < 0.5)
        if _steady and os.path.exists("render_vertical_crop_v4.py"):
            log("crop", f"rendering vertical crop for {base} (v4 steady renderer"
                        + (", auto-wide" if _auto_wide else "") + ")...")
            r = subprocess.run(
                [sys.executable, "render_vertical_crop_v4.py", "--clip", clip_path,
                 "--track", render_track, "--output", vertical_path]
                + (["--auto-wide"] if _auto_wide else [])
                + (["--face-height", f"{th:.3f}"] if (_auto_wide and th) else []),
                capture_output=True, text=True
            )
            if r.returncode != 0 or not os.path.exists(vertical_path):
                print(r.stderr[-800:])
                log("crop", "  v4 renderer failed -- falling back to v3")
                for _p in (vertical_path, vertical_path.rsplit(".", 1)[0] + "_noaudio.mp4"):
                    if os.path.exists(_p):
                        os.remove(_p)
                r = None
            else:
                for _ln in r.stdout.splitlines():
                    if _ln.startswith("auto-wide:"):
                        log("crop", "  " + _ln)
                        summary.setdefault(base, {})["auto_wide"] = "WIDE" in _ln
        if r is None:
            log("crop", f"rendering vertical crop for {base} (v3 renderer)...")
            r = subprocess.run(
                [sys.executable, "render_vertical_crop_v3.py", "--clip", clip_path,
                 "--track", render_track, "--output", vertical_path],
                capture_output=True, text=True
            )
        print(r.stdout[-1200:])
        if r.returncode != 0 or not os.path.exists(vertical_path):
            print(r.stderr[-1500:])
            log("crop", f"  FAILED to render {base}")
            failed.append(base)
            continue

        log("crop", f"  saved {vertical_path}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if failed:
        log("crop", f"{len(failed)} clip(s) could not be auto-cropped and need manual review: {', '.join(failed)}")
    log("crop", f"done -- vertical clips in ./{vertical_dir}/ (presence summary: {summary_path})")
    return failed


# --------------------------------------------------------------------------
# Stage 9: frozen caption/recovery/finish pipeline
# --------------------------------------------------------------------------

_CREDIT_MSG = (
    "ERROR: STOPPED -- Anthropic API credit balance is too low. Top up at "
    "console.anthropic.com (Plans & Billing), then re-run the same command: "
    "finished clips are kept and the run continues where it stopped."
)


def api_credits_ok():
    """1-token test call. False only when Anthropic reports low credit balance."""
    try:
        from anthropic import Anthropic
        Anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=1,
            messages=[{"role": "user", "content": "ok"}],
        )
        return True
    except Exception as e:
        return "credit balance" not in str(e).lower()


def stage_finish_verticals(
    vod_id,
    crop_input_dir,
    force=False,
    max_target_absent=None,
    trim_dead_air=True,
    channel=None,
    smart_trim=False,
    internal_cuts=False,
):
    """
    Send successfully rendered vertical clips into the already-frozen
    run_full_clip_pipeline.py downstream pipeline.

    Metadata sources:
      - timestamp: candidate filename
      - score/context: visual judgment JSON when available
      - fallback score: original filename score

    This stage does NOT modify caption, resolver, recovery, render,
    quality-gate, hook, watermark, or routing internals.
    """

    import re

    vertical_dir = os.path.join(
        crop_input_dir,
        "vertical",
    )

    if not os.path.isdir(vertical_dir):
        log(
            "finish",
            f"no vertical directory found: {vertical_dir}",
        )
        return [], []

    vertical_clips = sorted(
        glob.glob(
            os.path.join(
                vertical_dir,
                "*_vertical.mp4",
            )
        )
    )

    if not vertical_clips:
        log(
            "finish",
            "no successfully rendered vertical clips found",
        )
        return [], []

    # visual_approved lives inside the main VOD output directory.
    if os.path.basename(
        os.path.normpath(crop_input_dir)
    ) == "visual_approved":

        vod_output_dir = os.path.dirname(
            os.path.normpath(crop_input_dir)
        )

    else:
        vod_output_dir = os.path.normpath(
            crop_input_dir
        )

    judgments_dir = os.path.join(
        vod_output_dir,
        "visual_judgments",
    )

    finished_root = os.path.join(
        vertical_dir,
        "finished",
    )

    os.makedirs(
        finished_root,
        exist_ok=True,
    )

    completed = []
    failed = []

    # ----------------------------------------------------
    # Target-presence gate (only when the tracklet tracker
    # wrote _presence_summary.json; legacy runs are unaffected).
    # A clip is held back only when the target is CONFIRMED
    # absent (everyone on screen identified as someone else)
    # for more than max_target_absent of it. "uncertain" and
    # "sole_subject" frames never count against a clip.
    # ----------------------------------------------------
    presence_summary = {}
    summary_path = os.path.join(vertical_dir, "_presence_summary.json")
    if max_target_absent is not None and os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                presence_summary = json.load(f)
        except Exception as e:
            log("finish", f"warning: could not read {summary_path}: {e}")
    skipped_low_presence = {}

    for i, video_path in enumerate(vertical_clips):
        if not api_credits_ok():
            log("finish", _CREDIT_MSG)
            raise SystemExit(_CREDIT_MSG)

        vertical_base = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        if vertical_base.endswith("_vertical"):
            source_base = vertical_base[:-len("_vertical")]
        else:
            source_base = vertical_base

        output_dir = os.path.join(
            finished_root,
            source_base,
        )

        pres = presence_summary.get(source_base)
        if (pres is not None and pres.get("crop_mode") != "action"
                and pres.get("absent_frac", 0.0) > max_target_absent):
            log(
                "finish",
                f"[{i+1}/{len(vertical_clips)}] {source_base} held back: target confirmed "
                f"absent {100 * pres['absent_frac']:.0f}% (> {100 * max_target_absent:.0f}%), "
                f"tracked {100 * pres.get('tracked_frac', 0):.0f}%",
            )
            skipped_low_presence[source_base] = pres
            continue

        complete_marker = os.path.join(
            output_dir,
            "_full_clip_pipeline_complete.json",
        )

        if (
            os.path.exists(complete_marker)
            and not force
        ):
            log(
                "finish",
                f"[{i+1}/{len(vertical_clips)}] "
                f"{source_base} already finished, skipping",
            )

            completed.append(output_dir)
            continue

        if force and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # ----------------------------------------------------
        # Read visual editorial metadata if available.
        # ----------------------------------------------------

        judgment_path = os.path.join(
            judgments_dir,
            f"{source_base}_visual.json",
        )

        judgment = {}

        if os.path.exists(judgment_path):
            try:
                with open(judgment_path) as f:
                    judgment = json.load(f)
            except Exception as e:
                log(
                    "finish",
                    f"warning: could not read {judgment_path}: {e}",
                )

        # ----------------------------------------------------
        # Timestamp from filename.
        #
        # Supports:
        #   final_06_94m45s_score7
        #   chat_07_92m15s_1.1x
        #   visualdisc_03_197m33s
        # ----------------------------------------------------

        tm = re.search(
            r"_(\d+)m(\d+)s(?:_|$)",
            source_base,
        )

        if tm:
            timestamp = (
                f"{int(tm.group(1))}m"
                f"{int(tm.group(2)):02d}s"
            )
        else:
            timestamp = "unknown"

        # ----------------------------------------------------
        # Prefer visual-editor score because it is the final
        # editorial decision. Fall back to transcript score
        # embedded in final_* filenames.
        # ----------------------------------------------------

        score = judgment.get(
            "score",
            0,
        )

        try:
            score = int(score)
        except Exception:
            score = 0

        if score <= 0:
            sm = re.search(
                r"_score(\d+)(?:_|$)",
                source_base,
            )

            if sm:
                score = int(
                    sm.group(1)
                )

        # ----------------------------------------------------
        # Build real scene context for downstream hook creation.
        #
        # These are descriptions produced by the visual editor,
        # not guessed/hardcoded context.
        # ----------------------------------------------------

        context_parts = []

        for key in (
            "visual_hook",
            "setup",
            "progression",
            "payoff",
            "why_keep_watching",
        ):
            value = judgment.get(
                key
            )

            if isinstance(value, str):
                value = value.strip()

                if value:
                    context_parts.append(
                        value
                    )

        if context_parts:
            context = " ".join(
                context_parts
            )
        else:
            context = (
                "A livestream clip selected by the automated "
                "moment-discovery pipeline."
            )

        if channel:
            context = f"Streamer: {channel} (Twitch channel name). " + context

        # name list for the caption resolver (read via environment by
        # resolve_caption_text_v2.py, inherited through the finish scripts)
        import re as _re
        try:
            _COMMON_WORDS = {w.strip() for w in open("/usr/share/dict/words") if w[:1].islower()}
        except Exception:
            _COMMON_WORDS = set()
        _nf = os.path.join("streamer_names", f"{channel}.json") if channel else ""
        os.environ["CLIP_NAMES_FILE"] = _nf if _nf and os.path.exists(_nf) else ""
        _stop = {"The", "A", "An", "He", "She", "They", "His", "Her", "Their", "This", "That",
                 "It", "In", "On", "At", "As", "After", "Before", "When", "While", "Then",
                 "Streamer", "Twitch", "TikTok", "Chat", "No", "Not", "But", "And", "Or",
                 "With", "Without", "A livestream"}
        _found = []
        for _mm in _re.finditer(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*", context):
            _m = _mm.group(0)
            for _lead in ("The ", "A ", "An "):
                if _m.startswith(_lead):
                    _m = _m[len(_lead):]
            _parts = _m.split()
            while len(_parts) > 1 and _parts[0] in ("Then", "When", "After", "Before", "While", "So",
                                                    "But", "And", "Now", "Here", "There", "Meanwhile"):
                _parts.pop(0)
            while len(_parts) > 1 and _parts[-1] in ("Someone", "Everyone", "Nobody", "Chat"):
                _parts.pop()
            _m = " ".join(_parts)
            _before = context[:_mm.start()].rstrip()
            _sentence_start = (not _before) or _before[-1] in ".!?:;-\u2014\""
            if " " not in _m and not _m.isupper() and _sentence_start:
                continue
            if " " not in _m and not _m.isupper() and _m.lower() in _COMMON_WORDS:
                continue
            if _m not in _stop and len(_m) >= 3 and _m not in _found:
                _found.append(_m)
        os.environ["CLIP_EXTRA_NAMES"] = ", ".join(_found[:15])

        clip_id = (
            f"{vod_id}_{source_base}"
        )

        log(
            "finish",
            f"[{i+1}/{len(vertical_clips)}] "
            f"{source_base} "
            f"(timestamp={timestamp}, score={score})",
        )

        # ----------------------------------------------------
        # Dead-air trim (black screen, or a frozen picture with
        # silent audio) BEFORE captions, so the transcript and
        # captions are built on the trimmed clip. Writes
        # {source_base}_vertical_trimmed.mp4 next to the vertical
        # (reused on re-runs); the vertical itself is untouched.
        # If trimming fails the untrimmed vertical is used.
        # ----------------------------------------------------
        finish_input = video_path
        if trim_dead_air:
            trimmed_path = os.path.join(vertical_dir, f"{source_base}_vertical_trimmed.mp4")
            if force and os.path.exists(trimmed_path):
                os.remove(trimmed_path)
                if os.path.exists(trimmed_path + ".json"):
                    os.remove(trimmed_path + ".json")
            if not os.path.exists(trimmed_path):
                tr = subprocess.run(
                    [sys.executable, "trim_dead_air.py", "--input", video_path,
                     "--output", trimmed_path],
                    capture_output=True, text=True,
                )
                print(tr.stdout[-1200:])
                if tr.returncode != 0:
                    print(tr.stderr[-800:])
            if os.path.exists(trimmed_path):
                finish_input = trimmed_path
                try:
                    with open(trimmed_path + ".json") as f:
                        tj = json.load(f)
                    cut_s = tj.get("duration_in", 0) - tj.get("duration_out", 0)
                    if cut_s > 0.05:
                        log("finish", f"  dead air trimmed: {cut_s:.1f}s removed "
                                      f"({tj['duration_in']:.1f}s -> {tj['duration_out']:.1f}s)")
                except Exception:
                    pass
            else:
                log("finish", f"  warning: dead-air trim failed for {source_base}; using untrimmed clip")

        # ----------------------------------------------------
        # Optional smart trim (--smart-trim): an editor-style pass
        # that trims slow setup and dragging endings, BEFORE
        # captions so everything downstream uses the trimmed clip.
        # Keeps the clip as is when unsure or on any failure.
        # ----------------------------------------------------
        # ----------------------------------------------------
        # Optional internal cuts (--internal-cuts): the editor may
        # also remove dull parts from the middle. BEFORE captions.
        # Big edits (> half the clip removed) are held for review.
        # ----------------------------------------------------
        ic_applied = False
        edit_review = None
        if internal_cuts:
            ic_dir = os.path.join(vertical_dir, "internal_cuts")
            ic_name = os.path.splitext(os.path.basename(finish_input))[0]
            ic_out = os.path.join(ic_dir, f"{ic_name}_cuts.mp4")
            ic_json = os.path.join(ic_dir, f"{ic_name}_cuts.json")
            if force:
                for stale in (ic_out, ic_json):
                    if os.path.exists(stale):
                        os.remove(stale)
            if not os.path.exists(ic_json):
                os.makedirs(ic_dir, exist_ok=True)
                icr = subprocess.run(
                    [sys.executable, "internal_cuts.py", finish_input, "--out", ic_dir],
                    capture_output=True, text=True,
                )
                print(icr.stdout[-1500:])
                if icr.returncode != 0:
                    print(icr.stderr[-800:])
            try:
                with open(ic_json) as f:
                    icj = json.load(f)
                if icj.get("applied") and os.path.exists(ic_out):
                    removed = 1.0 - float(icj["kept_seconds"]) / max(1e-6, float(icj["duration"]))
                    parts = " + ".join(f"{a:.1f}-{b:.1f}s" for a, b in icj.get("parts", []))
                    log("finish", f"  internal cuts: kept {parts} ({icj['kept_seconds']:.0f}s of "
                                  f"{icj['duration']:.0f}s, {100 * removed:.0f}% removed)")
                    finish_input = ic_out
                    ic_applied = True
                    if removed > 0.5:
                        edit_review = {"reason": "internal cuts removed more than half the clip",
                                       "removed_pct": round(100 * removed),
                                       "parts": icj.get("parts"), "decision": ic_json}
                else:
                    log("finish", f"  internal cuts: none ({icj.get('kept_original_because')})")
            except Exception as e:
                log("finish", f"  internal cuts: no decision ({e.__class__.__name__}); clip kept as is")

        if smart_trim and not ic_applied:
            st_dir = os.path.join(vertical_dir, "smart_trim")
            st_name = os.path.splitext(os.path.basename(finish_input))[0]
            st_out = os.path.join(st_dir, f"{st_name}_trim.mp4")
            st_json = os.path.join(st_dir, f"{st_name}_trim.json")
            if force:
                for stale in (st_out, st_json):
                    if os.path.exists(stale):
                        os.remove(stale)
            if not os.path.exists(st_json):
                os.makedirs(st_dir, exist_ok=True)
                st = subprocess.run(
                    [sys.executable, "smart_trim.py", finish_input, "--out", st_dir],
                    capture_output=True, text=True,
                )
                print(st.stdout[-1500:])
                if st.returncode != 0:
                    print(st.stderr[-800:])
            try:
                with open(st_json) as f:
                    sj = json.load(f)
                if sj.get("applied") and os.path.exists(st_out):
                    log("finish", f"  smart trim: {sj['final_start']:.1f}s -> {sj['final_end']:.1f}s "
                                  f"({sj['final_end'] - sj['final_start']:.0f}s of {sj['duration']:.0f}s)")
                    finish_input = st_out
                else:
                    log("finish", f"  smart trim: kept as is ({sj.get('kept_original_because')})")
            except Exception as e:
                log("finish", f"  smart trim: no decision ({e.__class__.__name__}); clip kept as is")

        cmd = [
            sys.executable,
            "run_full_clip_pipeline.py",
            "--video",
            finish_input,
            "--context",
            context,
            "--clip-id",
            clip_id,
            "--vod",
            str(vod_id),
            "--timestamp",
            timestamp,
            "--score",
            str(score),
            "--output-dir",
            output_dir,
            "--production-dir",
            os.path.join(vod_output_dir, "production_ready"),
        ]

        print()
        print("=" * 72)
        print(
            "$",
            " ".join(cmd),
        )
        print("=" * 72)

        r = subprocess.run(
            cmd
        )

        if r.returncode != 0:

            log(
                "finish",
                f"FAILED downstream pipeline for {source_base}",
            )

            failed.append(
                source_base
            )

            continue

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        if edit_review:
            prod = os.path.join(vod_output_dir, "production_ready")
            hold = os.path.join(vod_output_dir, "needs_edit_review")
            moved_any = False
            for ext in (".mp4", ".json"):
                fp = os.path.join(prod, f"{clip_id}_final{ext}")
                if os.path.exists(fp):
                    os.makedirs(hold, exist_ok=True)
                    shutil.move(fp, os.path.join(hold, os.path.basename(fp)))
                    moved_any = True
            edit_review["held_final"] = moved_any
            with open(os.path.join(output_dir, "_edit_review.json"), "w") as f:
                json.dump(edit_review, f, indent=2)
            log("finish", f"  big edit ({edit_review['removed_pct']}% removed) -> "
                          + ("held in needs_edit_review/ for a human check" if moved_any
                             else "noted for review (it wasn't approved anyway)"))

        marker_data = {
            "vod": str(vod_id),
            "source_base": source_base,
            "vertical_video": video_path,
            "judgment": (
                judgment_path
                if os.path.exists(judgment_path)
                else None
            ),
            "timestamp": timestamp,
            "score": score,
            "context": context,
        }

        with open(
            complete_marker,
            "w",
        ) as f:
            json.dump(
                marker_data,
                f,
                indent=2,
            )

        completed.append(
            output_dir
        )

        log(
            "finish",
            f"completed {source_base}",
        )

    if presence_summary:
        skipped_path = os.path.join(vertical_dir, "_skipped_low_presence.json")
        with open(skipped_path, "w") as f:
            json.dump(skipped_low_presence, f, indent=2)
        if skipped_low_presence:
            log("finish", f"{len(skipped_low_presence)} clip(s) held back for low target presence "
                          f"-- see {skipped_path}")

    log(
        "finish",
        (
            f"done -- {len(completed)} completed, "
            f"{len(failed)} failed"
        ),
    )

    return completed, failed


def stage_refetch(
    vod_id,
    adapter,
    final_path,
    stream_start,
    quality,
    output_dir,
    force,
    local_video=None,
):
    if os.path.exists(output_dir) and not force:
        existing = glob.glob(os.path.join(output_dir, "*.mp4"))
        if existing:
            log("refetch", f"{output_dir} already has {len(existing)} clips, skipping (--force-refetch to redo)")
            return output_dir

    with open(final_path) as f:
        results = json.load(f)

    passed = [r for r in results if not r.get("flagged_silent")]
    passed.sort(key=lambda r: -r["top_score"])

    os.makedirs(output_dir, exist_ok=True)

    sub_path = None
    if not local_video:
        sub_path, _ = get_manifest(adapter, vod_id, quality)

    for i, r in enumerate(passed):
        start_sec = r["merged_start"]
        end_sec = r["merged_end"]
        midpoint_sec = (start_sec + end_sec) / 2
        half_width = (end_sec - start_sec) / 2

        m, s = divmod(int(start_sec), 60)
        log("refetch", f"[{i+1}/{len(passed)}] {m:02d}m{s:02d}s (score={r['top_score']})")

        cleanup_leftover_segments()

        dest = os.path.join(
            output_dir,
            f"final_{i+1:02d}_{m:02d}m{s:02d}s_score{r['top_score']}.mp4",
        )

        try:
            if local_video:
                cut_local_video_window(
                    local_video,
                    start_sec,
                    end_sec - start_sec,
                    dest,
                )
                clip_path = dest
            else:
                target = stream_start + timedelta(seconds=midpoint_sec)
                clip_path = adapter.fetch_segment_window(
                    sub_path,
                    target,
                    window_seconds=half_width,
                )
                os.rename(clip_path, dest)
        except Exception as e:
            log("refetch", f"  SKIP: fetch failed: {e}")
            continue
        log("refetch", f"  saved {dest}")

    log("refetch", f"done -- clips in ./{output_dir}/")
    return output_dir




# --------------------------------------------------------------------------
# Stage 6.5: independent chat-spike candidate discovery
# --------------------------------------------------------------------------

def stage_chat_spike_candidates(
    vod_id,
    adapter,
    transcript_path,
    final_path,
    stream_start,
    quality,
    output_dir,
    spike_threshold,
    top_n,
    window_seconds,
    force,
):
    """
    Independent candidate discovery from Twitch chat.

    Important:
    chat is NOT a hard gate for transcript candidates.

    A moment enters the visual-review queue if it was found by:
      - transcript scoring
      OR
      - chat spike detection

    Spike math:
      - 15 second buckets
      - count UNIQUE chatters
      - local median baseline over +/- 20 buckets (~300 sec)
      - keep ratio >= spike_threshold
      - dedupe spikes within 60 seconds
      - skip moments already covered by transcript candidate intervals

    Resulting MP4s are added directly to output_dir so the existing
    visual judge reviews transcript and chat candidates together.
    """

    import statistics

    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Determine VOD duration from transcript.
    # --------------------------------------------------------

    with open(transcript_path) as f:
        transcript_data = json.load(f)

    segments = transcript_data.get(
        "segments",
        [],
    )

    if not segments:
        log(
            "chat",
            "transcript contains no segments; skipping chat discovery",
        )
        return []

    total_duration = float(
        segments[-1]["end"]
    )

    stream_end = (
        stream_start
        + timedelta(
            seconds=total_duration
        )
    )


    # --------------------------------------------------------
    # Existing transcript candidate intervals.
    #
    # We use these to avoid fetching a duplicate chat candidate
    # when the transcript route already found the same moment.
    # --------------------------------------------------------

    transcript_intervals = []

    if os.path.exists(final_path):
        try:
            with open(final_path) as f:
                final_data = json.load(f)

            for cand in final_data:
                if cand.get(
                    "flagged_silent",
                    False,
                ):
                    continue

                start = float(
                    cand.get(
                        "merged_start",
                        0,
                    )
                )

                end = float(
                    cand.get(
                        "merged_end",
                        start,
                    )
                )

                transcript_intervals.append(
                    (
                        start,
                        end,
                    )
                )

        except Exception as e:
            log(
                "chat",
                f"warning: could not read transcript intervals: {e}",
            )


    # --------------------------------------------------------
    # Pull full Twitch chat log.
    # Expected tuples:
    #   (elapsed_seconds, username, message)
    # --------------------------------------------------------

    log(
        "chat",
        "loading Twitch chat log...",
    )

    chat_cache = f"twitch_chat_{vod_id}_full.json"

    if os.path.exists(chat_cache):
        try:
            with open(chat_cache) as f:
                cached = json.load(f)

            chat_tuples = [
                tuple(x)
                for x in cached
            ]

            log(
                "chat",
                f"loaded cached chat: {len(chat_tuples)} messages",
            )

        except Exception as e:
            log(
                "chat",
                f"cache read failed, refetching: {e}",
            )
            chat_tuples = None

    else:
        chat_tuples = None


    if chat_tuples is None:
        try:
            chat_tuples = adapter.get_chat_log(
                vod_id,
                stream_start,
                stream_end,
            )

            with open(
                chat_cache,
                "w",
            ) as f:
                json.dump(
                    chat_tuples,
                    f,
                )

            log(
                "chat",
                f"saved chat cache: {chat_cache}",
            )

        except Exception as e:
            log(
                "chat",
                f"chat retrieval failed: {e}",
            )
            return []


    if not chat_tuples:
        log(
            "chat",
            "no chat messages returned",
        )
        return []


    log(
        "chat",
        f"loaded {len(chat_tuples)} chat messages",
    )


    # --------------------------------------------------------
    # 15-second UNIQUE chatter buckets.
    # --------------------------------------------------------

    CHAT_BUCKET_SECONDS = 15
    LOCAL_RADIUS_BUCKETS = 20

    buckets = {}

    for item in chat_tuples:

        if len(item) < 2:
            continue

        offset = item[0]
        username = item[1]

        try:
            offset = float(
                offset
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if offset < 0:
            continue

        bucket_idx = int(
            offset
            // CHAT_BUCKET_SECONDS
        )

        buckets.setdefault(
            bucket_idx,
            set(),
        ).add(
            str(username)
        )


    if not buckets:
        log(
            "chat",
            "chat log produced no valid buckets",
        )
        return []


    max_bucket = max(
        buckets.keys()
    )

    chatter_counts = [
        len(
            buckets.get(
                i,
                set(),
            )
        )
        for i in range(
            max_bucket + 1
        )
    ]


    # --------------------------------------------------------
    # Local baseline / relative spike ratio.
    #
    # This matches twitch_hybrid_score_full.py's proven math:
    # +/- 20 buckets = roughly +/- 300 seconds.
    # --------------------------------------------------------

    ratios = []

    for i in range(
        len(chatter_counts)
    ):

        lo = max(
            0,
            i - LOCAL_RADIUS_BUCKETS,
        )

        hi = min(
            len(chatter_counts),
            i + LOCAL_RADIUS_BUCKETS + 1,
        )

        neighborhood = chatter_counts[
            lo:hi
        ]

        if len(neighborhood) >= 10:
            baseline = statistics.median(
                neighborhood
            )
        else:
            baseline = statistics.median(
                chatter_counts
            )

        baseline = max(
            baseline,
            1e-6,
        )

        ratio = (
            chatter_counts[i]
            / baseline
        )

        ratios.append(
            ratio
        )


    # --------------------------------------------------------
    # Smart chat-event ranking.
    #
    # Instead of relying only on raw chatter ratio, combine:
    #   - local chatter spike
    #   - acceleration versus previous minute
    #   - reaction-message density
    #   - absolute unique chatter count
    #
    # This proved much better in smoke testing than ratio alone.
    # --------------------------------------------------------

    import re

    REACTION_RE = re.compile(
        r"""
        \b(
            wtf|
            what\s+the|
            no\s+way|
            nooo+|
            yoo+|
            omg|
            oh\s+shit|
            holy|
            nah+|
            lmao+|
            lmfao+|
            lol+|
            dead|
            crying|
            got\s*em|
            crazy|
            insane
        )\b
        |😂|🤣|😭|💀|😱|🤯
        """,
        re.I | re.X,
    )


    # Rebuild per-bucket message lists so we can calculate
    # reaction density.
    message_buckets = {}

    for item in chat_tuples:

        if len(item) < 3:
            continue

        offset = item[0]
        message = item[2]

        try:
            offset = float(
                offset
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if offset < 0:
            continue

        bucket_idx = int(
            offset
            // CHAT_BUCKET_SECONDS
        )

        message_buckets.setdefault(
            bucket_idx,
            [],
        ).append(
            str(message)
        )


    raw_candidates = []


    for i, (
        chatter_count,
        ratio,
    ) in enumerate(
        zip(
            chatter_counts,
            ratios,
        )
    ):

        bucket_sec = (
            i
            * CHAT_BUCKET_SECONDS
        )


        # Startup guard:
        # ignore first 5 minutes because acceleration is unstable
        # before enough prior history exists.
        if bucket_sec < 300:
            continue


        # -----------------------------------
        # Acceleration versus prior minute.
        # -----------------------------------

        previous = [
            chatter_counts[j]
            for j in range(
                max(0, i - 4),
                i,
            )
        ]

        if previous:
            previous_base = statistics.median(
                previous
            )
        else:
            previous_base = chatter_count

        previous_base = max(
            previous_base,
            1,
        )

        acceleration = (
            chatter_count
            / previous_base
        )


        # -----------------------------------
        # Reaction density.
        # -----------------------------------

        messages = message_buckets.get(
            i,
            [],
        )

        reaction_count = sum(
            1
            for m in messages
            if REACTION_RE.search(m)
        )

        reaction_density = (
            reaction_count
            / max(
                len(messages),
                1,
            )
        )


        # -----------------------------------
        # Combined event score.
        # -----------------------------------

        event_score = (
            ratio
            * acceleration
            * (
                1.0
                + 2.5
                * reaction_density
            )
            * (
                1.0
                + min(
                    chatter_count,
                    200,
                )
                / 500
            )
        )


        raw_candidates.append(
            {
                "bucket_sec": bucket_sec,
                "ratio": float(
                    ratio
                ),
                "chatters": int(
                    chatter_count
                ),
                "acceleration": float(
                    acceleration
                ),
                "reaction_density": float(
                    reaction_density
                ),
                "event_score": float(
                    event_score
                ),
            }
        )


    raw_candidates.sort(
        key=lambda x: -x[
            "event_score"
        ]
    )


    log(
        "chat",
        "TOP SMART CHAT EVENTS:",
    )

    for d in raw_candidates[:15]:

        mins = int(
            d["bucket_sec"] // 60
        )

        secs = int(
            d["bucket_sec"] % 60
        )

        log(
            "chat",
            (
                f"  {mins:03d}:{secs:02d} "
                f"event={d['event_score']:.3f} "
                f"ratio={d['ratio']:.3f}x "
                f"accel={d['acceleration']:.3f}x "
                f"reaction={d['reaction_density']:.1%} "
                f"chatters={d['chatters']}"
            ),
        )

    ranked_debug = sorted(
        [
            {
                "bucket_sec": i * CHAT_BUCKET_SECONDS,
                "ratio": ratios[i],
                "chatters": chatter_counts[i],
            }
            for i in range(len(ratios))
        ],
        key=lambda x: -x["ratio"],
    )

    log(
        "chat",
        "TOP CHAT RATIOS:",
    )

    for d in ranked_debug[:15]:
        mins = int(
            d["bucket_sec"] // 60
        )
        secs = int(
            d["bucket_sec"] % 60
        )

        log(
            "chat",
            (
                f"  {mins:03d}:{secs:02d} "
                f"ratio={d['ratio']:.3f}x "
                f"chatters={d['chatters']}"
            ),
        )


    if not raw_candidates:
        return []


    # --------------------------------------------------------
    # Dedupe nearby spike buckets.
    #
    # Candidates are ranked strongest-first by event_score.
    # Keep a candidate only when no stronger selected event is
    # already within +/- 60 seconds.
    # --------------------------------------------------------

    deduped = []

    for cand in raw_candidates:

        too_close = any(
            abs(
                cand["bucket_sec"]
                - existing["bucket_sec"]
            )
            <= 60
            for existing in deduped
        )

        if too_close:
            continue

        deduped.append(
            cand
        )


    # --------------------------------------------------------
    # Skip chat moments already covered by transcript clips.
    # --------------------------------------------------------

    independent = []

    for cand in deduped:

        t = cand[
            "bucket_sec"
        ]

        covered = any(
            (
                start - 15
                <= t
                <= end + 15
            )
            for start, end
            in transcript_intervals
        )

        if covered:
            mins = int(t // 60)
            secs = int(t % 60)

            log(
                "chat",
                (
                    f"  overlap skip {mins:03d}:{secs:02d} "
                    f"event={cand['event_score']:.3f} "
                    f"(already covered by transcript candidate)"
                ),
            )
            continue

        independent.append(
            cand
        )


    # Strongest spikes first.
    independent.sort(
        key=lambda x: -x[
            "event_score"
        ]
    )

    independent = independent[
        :top_n
    ]


    log(
        "chat",
        (
            f"{len(independent)} independent "
            f"chat candidates after dedupe/"
            f"transcript-overlap removal"
        ),
    )


    if not independent:
        return []


    # --------------------------------------------------------
    # Fetch actual candidate videos.
    # --------------------------------------------------------

    sub_path, _ = get_manifest(
        adapter,
        vod_id,
        quality,
    )

    created = []


    for rank, cand in enumerate(
        independent,
        start=1,
    ):

        bucket_sec = cand[
            "bucket_sec"
        ]

        ratio = cand[
            "ratio"
        ]

        event_score = cand[
            "event_score"
        ]

        chatters = cand[
            "chatters"
        ]


        mins = int(
            bucket_sec
            // 60
        )

        secs = int(
            bucket_sec
            % 60
        )


        dest = os.path.join(
            output_dir,
            (
                f"chat_{rank:02d}_"
                f"{mins:02d}m{secs:02d}s_"
                f"{ratio:.1f}x.mp4"
            ),
        )


        if (
            os.path.exists(
                dest
            )
            and not force
        ):
            log(
                "chat",
                f"exists, skipping: {dest}",
            )

            created.append(
                dest
            )

            continue


        cleanup_leftover_segments()


        target = (
            stream_start
            + timedelta(
                seconds=bucket_sec
            )
        )


        log(
            "chat",
            (
                f"[{rank}/{len(independent)}] "
                f"{mins:02d}:{secs:02d} "
                f"event={event_score:.2f} "
                f"ratio={ratio:.2f}x "
                f"chatters={chatters}"
            ),
        )


        try:
            clip_path = (
                adapter.fetch_segment_window(
                    sub_path,
                    target,
                    window_seconds=window_seconds,
                )
            )

        except Exception as e:
            log(
                "chat",
                f"  fetch failed: {e}",
            )
            continue


        try:
            os.replace(
                clip_path,
                dest,
            )
        except Exception as e:
            log(
                "chat",
                f"  rename failed: {e}",
            )
            continue


        created.append(
            dest
        )


    log(
        "chat",
        (
            f"added {len(created)} "
            f"chat-discovered candidate clips"
        ),
    )

    return created


# --------------------------------------------------------------------------
# Stage 7: multimodal visual editorial judge
# --------------------------------------------------------------------------

def stage_visual_judge(
    output_dir,
    min_visual_score,
    force,
):
    """
    Final editorial gate.

    For every landscape candidate:
      1. transcribe the actual candidate clip locally via Groq
      2. sample video frames + transcript through visual_clip_judge.py
      3. keep only score >= min_visual_score AND postable=true

    Approved MP4s are copied into:
        {output_dir}/visual_approved/

    Per-clip judgments are saved in:
        {output_dir}/visual_judgments/
    """

    judgments_dir = os.path.join(
        output_dir,
        "visual_judgments",
    )

    approved_dir = os.path.join(
        output_dir,
        "visual_approved",
    )

    os.makedirs(
        judgments_dir,
        exist_ok=True,
    )

    if force and os.path.exists(
        approved_dir
    ):
        shutil.rmtree(
            approved_dir
        )

    os.makedirs(
        approved_dir,
        exist_ok=True,
    )


    # Only judge original landscape clips in output_dir itself.
    clips = sorted(
        glob.glob(
            os.path.join(
                output_dir,
                "*.mp4",
            )
        )
    )

    if not clips:
        log(
            "visual",
            "no landscape candidate clips found",
        )
        return approved_dir


    approved_count = 0
    rejected_count = 0
    failed_count = 0


    for i, clip_path in enumerate(
        clips
    ):
        if not api_credits_ok():
            log("visual", _CREDIT_MSG)
            raise SystemExit(_CREDIT_MSG)
        base = os.path.splitext(
            os.path.basename(
                clip_path
            )
        )[0]

        transcript_path = os.path.join(
            judgments_dir,
            f"{base}_transcript.json",
        )

        judgment_path = os.path.join(
            judgments_dir,
            f"{base}_visual.json",
        )

        log(
            "visual",
            f"[{i+1}/{len(clips)}] judging {base}",
        )


        # ----------------------------------------------------
        # Transcribe this exact candidate clip.
        # ----------------------------------------------------

        if (
            force
            or not os.path.exists(
                transcript_path
            )
        ):

            log(
                "visual",
                "  transcribing candidate...",
            )

            r = subprocess.run(
                [
                    sys.executable,
                    "transcribe_groq.py",
                    clip_path,
                    transcript_path,
                ],
                capture_output=True,
                text=True,
            )

            if r.returncode != 0:
                print(
                    r.stderr[-1500:]
                )

                log(
                    "visual",
                    "  FAILED transcription",
                )

                failed_count += 1
                continue


        # ----------------------------------------------------
        # Run multimodal visual judge.
        # ----------------------------------------------------

        if (
            force
            or not os.path.exists(
                judgment_path
            )
        ):

            log(
                "visual",
                "  running visual editor...",
            )

            r = subprocess.run(
                [
                    sys.executable,
                    "visual_clip_judge.py",
                    "--video",
                    clip_path,
                    "--transcript",
                    transcript_path,
                    "--output",
                    judgment_path,
                ],
                capture_output=True,
                text=True,
            )

            if r.returncode != 0:
                print(
                    r.stdout[-1000:]
                )
                print(
                    r.stderr[-1500:]
                )

                log(
                    "visual",
                    "  FAILED visual judgment",
                )

                failed_count += 1
                continue


        try:
            with open(
                judgment_path
            ) as f:
                judgment = json.load(
                    f
                )

        except Exception as e:
            log(
                "visual",
                f"  FAILED reading judgment: {e}",
            )

            failed_count += 1
            continue


        score = int(
            judgment.get(
                "score",
                0,
            )
        )

        postable = bool(
            judgment.get(
                "postable",
                False,
            )
        )

        reason = judgment.get(
            "reason",
            "",
        )


        if (
            score >= min_visual_score
            and postable
        ):

            dest = os.path.join(
                approved_dir,
                os.path.basename(
                    clip_path
                ),
            )

            shutil.copy2(
                clip_path,
                dest,
            )

            approved_count += 1

            log(
                "visual",
                f"  APPROVED score={score}: {reason}",
            )

        else:

            rejected_count += 1

            log(
                "visual",
                f"  REJECTED score={score} "
                f"postable={postable}: {reason}",
            )


    log(
        "visual",
        (
            f"complete -- "
            f"{approved_count} approved, "
            f"{rejected_count} rejected, "
            f"{failed_count} failed"
        ),
    )

    log(
        "visual",
        f"approved clips: {approved_dir}",
    )

    log(
        "visual",
        f"judgments: {judgments_dir}",
    )


    return approved_dir


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run the full chuzzzclips pipeline for one Twitch VOD.")
    p.add_argument("--vod-id", required=True, help="Twitch VOD ID (numeric, from the VOD URL)")
    p.add_argument("--stream-start", default=None,
                   help="ISO8601 UTC stream start time, e.g. 2026-08-04T23:39:01.405+00:00. "
                        "If omitted, fetched automatically via Helix (needs --helix-client-id/secret "
                        "or TWITCH_CLIENT_ID/TWITCH_CLIENT_SECRET env vars).")
    p.add_argument("--helix-client-id", default=os.environ.get("TWITCH_CLIENT_ID"))
    p.add_argument("--helix-client-secret", default=os.environ.get("TWITCH_CLIENT_SECRET"))
    p.add_argument("--quality", default="1080p")
    p.add_argument("--score-window-seconds", type=float, default=35)
    p.add_argument("--score-step-seconds", type=float, default=20)
    p.add_argument("--min-score", type=int, default=8)
    p.add_argument("--padding", type=float, default=6)
    p.add_argument("--max-clip-duration", type=float, default=75)
    p.add_argument("--silence-rms-threshold", type=float, default=300)
    p.add_argument("--output-dir", default=None, help="Defaults to {vod_id}_final_review/")
    p.add_argument("--reference-file", default=None,
                    help="Path to target_reference_<name>.json (from build_target_reference.py). "
                         "If omitted, the auto-crop stage is skipped entirely and clips are left "
                         "in their original landscape/scale+pad form.")

    p.add_argument(
        "--tracker",
        choices=["legacy", "tracklets"],
        default="legacy",
        help=(
            "Crop-stage tracker. legacy = auto_track_target.py + render_vertical_crop.py "
            "(default, unchanged). tracklets = track_by_tracklets.py + "
            "render_vertical_crop_v3.py (cut-aware identity tracking, full-frame fallback)."
        ),
    )

    p.add_argument(
        "--face-backend",
        choices=["insight", "dlib"],
        default="insight",
        help=(
            "Face recognition used by --tracker tracklets. insight (default) = InsightFace "
            "detection + ArcFace identity, cross-checked against dlib; finds turned and small "
            "faces. dlib = the original HOG + dlib path. Falls back to dlib automatically "
            "if InsightFace can't load."
        ),
    )

    p.add_argument(
        "--skip-chat-spikes",
        action="store_true",
        help="Disable independent chat-spike candidate discovery."
    )

    p.add_argument(
        "--chat-spike-threshold",
        type=float,
        default=1.8,
        help="Minimum local unique-chatter spike ratio."
    )

    p.add_argument(
        "--chat-top-n",
        type=int,
        default=10,
        help="Maximum independent chat-spike candidates to fetch."
    )

    p.add_argument(
        "--skip-viewer-clips",
        action="store_true",
        help="Do not add candidates from the clips Twitch viewers made.",
    )

    p.add_argument(
        "--viewer-top-n",
        type=int,
        default=8,
        help="Viewer-clip moments to add as candidates per VOD (default 8).",
    )

    p.add_argument(
        "--viewer-min-views",
        type=int,
        default=200,
        help="Ignore viewer clips with fewer views than this (default 200).",
    )

    p.add_argument(
        "--chat-window-seconds",
        type=float,
        default=30,
        help="Window size passed to Twitch candidate fetch around each chat spike."
    )

    p.add_argument(
        "--skip-visual-discovery",
        action="store_true",
        help=(
            "Disable independent visual-only "
            "candidate discovery."
        ),
    )

    p.add_argument(
        "--visual-proxy-quality",
        default="360p",
        help=(
            "Twitch quality used for cheap "
            "full-VOD visual scanning."
        ),
    )

    p.add_argument(
        "--visual-discovery-maybe-confidence",
        type=int,
        default=0,
        help=(
            "Minimum confidence for MAYBE "
            "contact-sheet candidates."
        ),
    )

    p.add_argument(
        "--crop-mode",
        choices=["auto", "streamer", "action"],
        default="auto",
        help="auto (default): follow the streamer, but switch to following the action when the "
             "streamer is small (facecam/desk strip) or barely on screen. streamer / action: force one.",
    )

    p.add_argument(
        "--visual-min-score",
        type=int,
        default=7,
        help="Minimum multimodal visual score required for final approval."
    )

    p.add_argument(
        "--skip-visual-judge",
        action="store_true",
        help="Skip multimodal editorial review and send all fetched candidates onward."
    )

    p.add_argument(
        "--finish-verticals",
        action="store_true",
        help=(
            "After auto-cropping, run each successful vertical clip "
            "through the frozen caption/recovery/finish pipeline."
        ),
    )

    p.add_argument(
        "--channel",
        default=None,
        help="Streamer's channel login, for picking the target reference automatically "
             "(only needed with --local-video; Twitch VODs are looked up via Helix).",
    )

    p.add_argument(
        "--reference-map",
        default="reference_map.json",
        help="JSON mapping channel login -> target_reference file (default reference_map.json).",
    )

    p.add_argument(
        "--max-target-absent",
        type=float,
        default=0.4,
        help=(
            "With --tracker tracklets: don't finish a clip whose target is confirmed absent "
            "for more than this fraction of it (default 0.4). Uncertain frames never count. "
            "Set to 1 to disable."
        ),
    )

    p.add_argument(
        "--internal-cuts",
        action="store_true",
        help=(
            "Experimental, off by default: let the editor (internal_cuts.py) also remove "
            "dull parts from the middle, before captions. Edits removing more than half a "
            "clip are held in needs_edit_review/. Falls back to --smart-trim / untouched."
        ),
    )

    p.add_argument(
        "--smart-trim",
        action="store_true",
        help=(
            "Experimental, off by default: trim slow setup and dragging endings with "
            "smart_trim.py before captions. Keeps the clip as is when unsure."
        ),
    )

    p.add_argument(
        "--no-dead-air-trim",
        action="store_true",
        help=(
            "Don't trim dead air (black screen, or a frozen picture with silent audio) "
            "from vertical clips before captions. Trimming is on by default; the "
            "untrimmed vertical is always kept."
        ),
    )

    p.add_argument(
        "--force-finish",
        action="store_true",
        help=(
            "Re-run downstream finishing even when a completed marker "
            "already exists."
        ),
    )

    p.add_argument("--force-all", action="store_true")
    p.add_argument("--force-fetch", action="store_true")
    p.add_argument("--force-transcribe", action="store_true")
    p.add_argument("--force-score", action="store_true")
    p.add_argument("--force-merge", action="store_true")
    p.add_argument("--force-silence", action="store_true")
    p.add_argument("--force-refetch", action="store_true")
    p.add_argument("--force-crop", action="store_true")
    p.add_argument("--force-visual", action="store_true")
    p.add_argument("--force-chat", action="store_true")

    p.add_argument(
        "--local-video",
        default=None,
        help=(
            "Optional local archived VOD/video file. When supplied, "
            "Twitch fetching, stream-start lookup, chat-spike discovery, "
            "and visual-only full-VOD discovery are bypassed."
        ),
    )
    p.add_argument(
        "--force-visual-discovery",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()
    # ignore the keyboard: child ffmpeg processes inherit this, so a stray
    # keypress can't drop them into their interactive "Enter command:" prompt
    try:
        _devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(_devnull, 0)
    except OSError:
        pass
    vod_id = args.vod_id
    output_dir = args.output_dir or f"{vod_id}_final_review"

    local_video = None
    if args.local_video:
        local_video = os.path.abspath(args.local_video)
        if not os.path.isfile(local_video):
            raise FileNotFoundError(
                f"local video not found: {local_video}"
            )
        log("start", f"local-video mode: {local_video}")

    adapter = TwitchAdapter(
        helix_client_id=args.helix_client_id,
        helix_client_secret=args.helix_client_secret,
    )

    stream_start_cache = os.path.join(
        output_dir,
        "_stream_start.json",
    )

    if local_video:
        stream_start = None
        log(
            "start",
            "local-video mode: Twitch stream-start lookup not required",
        )

    elif args.stream_start:
        stream_start = datetime.fromisoformat(args.stream_start)
        if stream_start.tzinfo is None:
            stream_start = stream_start.replace(tzinfo=timezone.utc)

        os.makedirs(output_dir, exist_ok=True)
        with open(stream_start_cache, "w") as f:
            json.dump(
                {
                    "vod_id": vod_id,
                    "stream_start": stream_start.isoformat(),
                },
                f,
                indent=2,
            )

        log("start", f"saved stream-start cache: {stream_start_cache}")

    elif os.path.exists(stream_start_cache):
        with open(stream_start_cache) as f:
            cached = json.load(f)

        cached_vod = str(cached.get("vod_id", ""))
        if cached_vod != str(vod_id):
            raise RuntimeError(
                f"stream-start cache VOD mismatch: "
                f"expected {vod_id}, found {cached_vod}"
            )

        cached_start = cached.get("stream_start")
        if not cached_start:
            raise RuntimeError(
                f"stream-start cache is missing stream_start: "
                f"{stream_start_cache}"
            )

        stream_start = datetime.fromisoformat(cached_start)
        if stream_start.tzinfo is None:
            stream_start = stream_start.replace(tzinfo=timezone.utc)

        log("start", f"using cached stream start: {stream_start_cache}")

    else:
        log("start", "no cached stream start; fetching via Helix...")
        stream_start = adapter.get_stream_start_time(vod_id)

        os.makedirs(output_dir, exist_ok=True)
        with open(stream_start_cache, "w") as f:
            json.dump(
                {
                    "vod_id": vod_id,
                    "stream_start": stream_start.isoformat(),
                },
                f,
                indent=2,
            )

        log("start", f"saved stream-start cache: {stream_start_cache}")

    if stream_start is not None:
        log("start", f"stream_start = {stream_start.isoformat()}")

    f = lambda flag: args.force_all or flag

    audio_path = stage_fetch_audio(
        adapter,
        vod_id,
        args.quality,
        f(args.force_fetch),
        local_video=local_video,
    )
    transcript_path = stage_transcribe(vod_id, audio_path, f(args.force_transcribe))
    scores_path = stage_score(transcript_path, args.score_window_seconds,
                               args.score_step_seconds, f(args.force_score))
    merged_path = stage_merge(vod_id, scores_path, args.min_score, args.padding,
                               args.max_clip_duration, f(args.force_merge))
    final_path = stage_silence_check(
        vod_id,
        adapter,
        merged_path,
        stream_start,
        args.min_score,
        args.silence_rms_threshold,
        args.quality,
        f(args.force_silence),
        local_video=local_video,
    )
    stage_refetch(
        vod_id,
        adapter,
        final_path,
        stream_start,
        args.quality,
        output_dir,
        f(args.force_refetch),
        local_video=local_video,
    )

    if local_video:
        log(
            "chat",
            "skipped in --local-video mode (Twitch chat/absolute stream time unavailable)",
        )
    elif args.skip_chat_spikes:
        log(
            "chat",
            "skipped by --skip-chat-spikes",
        )
    else:
        stage_chat_spike_candidates(
            vod_id,
            adapter,
            transcript_path,
            final_path,
            stream_start,
            args.quality,
            output_dir,
            args.chat_spike_threshold,
            args.chat_top_n,
            args.chat_window_seconds,
            f(args.force_chat),
        )

    if args.skip_visual_discovery:
        log(
            "visual-discovery",
            "skipped by --skip-visual-discovery",
        )
    else:
        stage_visual_discovery(
            vod_id=vod_id,
            adapter=adapter,
            stream_start=stream_start,
            quality=args.quality,
            output_dir=output_dir,
            proxy_quality=(
                args.visual_proxy_quality
            ),
            maybe_confidence=(
                args.visual_discovery_maybe_confidence
            ),
            force=f(
                args.force_visual_discovery
            ),
            log_fn=log,
            local_video=local_video,
        )

    if local_video:
        log("viewer", "skipped in --local-video mode")
    elif args.skip_viewer_clips:
        log("viewer", "skipped by --skip-viewer-clips")
    else:
        try:
            from viewer_candidates import stage_viewer_candidates
            stage_viewer_candidates(
                vod_id, adapter, stream_start, output_dir,
                lambda: get_manifest(adapter, vod_id, args.quality)[0],
                top_n=args.viewer_top_n, min_views=args.viewer_min_views,
                force=False, log_fn=log,
            )
        except Exception as e:
            log("viewer", f"skipped ({e!r})")

    if args.skip_visual_judge:
        log(
            "visual",
            "skipped by --skip-visual-judge",
        )
        crop_input_dir = output_dir
    else:
        try:
            from dedupe_candidates import dedupe
            dedupe(output_dir, log_fn=log)
        except Exception as e:
            log("dedupe", f"skipped ({e!r})")
        crop_input_dir = stage_visual_judge(
            output_dir,
            args.visual_min_score,
            f(args.force_visual),
        )

    reference_file = resolve_reference(args, adapter, vod_id, output_dir, local_video)

    failed_crops = stage_auto_crop(
        crop_input_dir,
        reference_file,
        f(args.force_crop),
        tracker=args.tracker,
        face_backend=args.face_backend,
        crop_mode=args.crop_mode,
    )

    finished_outputs = []
    failed_finishes = []

    if args.finish_verticals:

        if not reference_file:
            log(
                "finish",
                "skipped: --finish-verticals needs a target reference "
                "(--reference-file, or a mapped channel in tracklets mode) "
                "so vertical clips exist",
            )

        else:
            (
                finished_outputs,
                failed_finishes,
            ) = stage_finish_verticals(
                vod_id,
                crop_input_dir,
                f(args.force_finish),
                max_target_absent=(
                    args.max_target_absent
                    if args.tracker == "tracklets"
                    else None
                ),
                trim_dead_air=not args.no_dead_air_trim,
                channel=args.channel,
                smart_trim=args.smart_trim,
                internal_cuts=args.internal_cuts,
            )

    else:
        log(
            "finish",
            "skipped (use --finish-verticals to enable)",
        )

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE for VOD {vod_id}")
    print(f"Landscape clips: ./{output_dir}/")
    if not args.skip_visual_judge:
        print(f"Visual-approved clips: ./{output_dir}/visual_approved/")
        print(f"Visual judgments: ./{output_dir}/visual_judgments/")

    if reference_file:
        print(f"Target reference: {reference_file}")
        if args.skip_visual_judge:
            vertical_output = os.path.join(output_dir, "vertical")
        else:
            vertical_output = os.path.join(
                output_dir,
                "visual_approved",
                "vertical",
            )

        print(f"Vertical (auto-cropped) clips: ./{vertical_output}/")
        if failed_crops:
            print(f"  ({len(failed_crops)} clip(s) need manual crop review: {', '.join(failed_crops)})")
        if args.finish_verticals:
            finished_root = os.path.join(
                vertical_output,
                "finished",
            )
            print(
                f"Finished downstream clips: "
                f"./{finished_root}/"
            )

            if failed_finishes:
                print(
                    f"  ({len(failed_finishes)} downstream "
                    f"finish failure(s): "
                    f"{', '.join(failed_finishes)})"
                )
    print("=" * 70)


if __name__ == "__main__":
    main()
