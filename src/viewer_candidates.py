#!/usr/bin/env python3
"""
viewer_candidates.py -- use the clips Twitch VIEWERS made as extra candidates.

Called by run_pipeline.py after chat-spike and visual discovery, before dedupe and the
visual judge. For a finished VOD it fetches the most-viewed viewer clips (with their
position in the VOD), and cuts a ~70s candidate window around each one from the VOD,
the same way chat candidates are cut. Files are named

    viewer_<rank>_<mm>m<ss>s_<views>v.mp4

so dedupe (30s rule) merges them with our own candidates of the same moment, and they go
through the SAME judge, crop, captions etc. Nothing is approved just because viewers
clipped it. If the Twitch keys are missing or the API fails, it logs and does nothing.

Needs TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET (env or ~/.twitch_keys).
"""
import json
import os
from datetime import timedelta

HALF_WINDOW = 35   # seconds either side of the viewer clip's middle


def stage_viewer_candidates(vod_id, adapter, stream_start, output_dir, get_sub_path,
                            top_n=8, min_views=200, force=False, log_fn=print):
    import viewer_clip_check as vcc

    def log(msg):
        log_fn("viewer", msg)

    try:
        cid, sec = vcc.keys()
    except SystemExit:
        log("no Twitch API keys -- skipped")
        return []
    try:
        tok = vcc.token(cid, sec)
        _video, clips = vcc.vod_clips(vod_id, cid, tok)
    except Exception as e:
        log(f"Twitch API failed ({e.__class__.__name__}: {e}) -- skipped")
        return []
    usable = [c for c in clips if c.get("vod_offset") is not None and c.get("view_count", 0) >= min_views]
    usable.sort(key=lambda c: -c["view_count"])
    picked = []
    for c in usable:                       # one candidate per moment (viewers often clip the same thing)
        mid = c["vod_offset"] + c.get("duration", 30) / 2
        if all(abs(mid - p[0]) > 30 for p in picked):
            picked.append((mid, c))
        if len(picked) >= top_n:
            break
    log(f"{len(clips)} viewer clips; {len(usable)} with >= {min_views} views; using top {len(picked)} moments")
    if not picked:
        return []

    sub_path = None
    created = []
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "_viewer_candidates.json")
    try:
        meta = json.load(open(meta_path))
    except Exception:
        meta = {}
    for rank, (mid, c) in enumerate(picked, 1):
        m, s = int(mid // 60), int(mid % 60)
        name = f"viewer_{rank:02d}_{m:02d}m{s:02d}s_{c['view_count']}v.mp4"
        dest = os.path.join(output_dir, name)
        meta[name[:-4]] = {"views": c["view_count"], "title": c.get("title"), "url": c.get("url"),
                           "vod_seconds": round(mid, 1)}
        judged = os.path.exists(os.path.join(output_dir, "visual_judgments", f"{name[:-4]}_visual.json"))
        if (os.path.exists(dest) or judged) and not force:
            log(f"[{rank}] exists/judged, skipping: {name}")
            continue
        try:
            if sub_path is None:
                sub_path = get_sub_path()
            target = stream_start + timedelta(seconds=mid)
            log(f"[{rank}/{len(picked)}] {m:02d}:{s:02d}  {c['view_count']} views  {c.get('title', '')[:50]!r}")
            clip_path = adapter.fetch_segment_window(sub_path, target, window_seconds=HALF_WINDOW)
            os.replace(clip_path, dest)
            created.append(dest)
        except Exception as e:
            log(f"  fetch failed: {e}")
    json.dump(meta, open(meta_path, "w"), indent=2)
    log(f"added {len(created)} viewer-clip candidates")
    return created
