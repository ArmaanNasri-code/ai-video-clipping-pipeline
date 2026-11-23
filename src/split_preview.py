#!/usr/bin/env python3
"""
split_preview.py -- PREVIEW split-screen clips (stream on top, gameplay below).

Standalone: does not touch the pipeline, the batch, or any existing clip.

Layouts (--layout):
  smart (default) -- FILL the top half and FOLLOW the subject using the tracking the
                     pipeline already saved for the clip (<name>_track.json). If the
                     subject can't be kept in frame (lost / transitions / two people too
                     far apart for long stretches), it falls back to FULL automatically.
  fill            -- fill the top half, fixed position (--x, 0=left 1=right)
  full            -- whole stream frame on top, blurred copy behind it
  all             -- make all three to compare

--top 0.45 = 45% stream / 55% gameplay (default). --top 0.5 = half/half.
Gameplay: --gameplay <file>, or a random file from backgrounds/ (not *.description).
The start point is random, and avoids the stretches used for recent previews
(remembered in backgrounds/_used.json) so consecutive clips don't repeat footage.
Gameplay is muted; only the stream's audio is kept. 1080x1920 output.
Captions/hook are NOT added yet -- this is to judge the layout.

    python3 split_preview.py                              # newest approved clip, smart layout
    python3 split_preview.py --layout all --top 0.45
    python3 split_preview.py --final <path to a *_final.mp4>
Output: split_previews/<clip name>/
"""
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys

OUT_W, OUT_H = 1080, 1920
SMOOTH = 0.08          # per-frame easing of the crop centre (like the vertical renderer)
BAD_LIMIT = 0.25       # more than this share of frames unfit for FILL -> use FULL


def newest_approved():
    links = sorted(glob.glob("batch_runs/*/approved/*.mp4"), key=os.path.getmtime, reverse=True)
    links += sorted(glob.glob("*_final_review/production_ready/*_final.mp4"), key=os.path.getmtime, reverse=True)
    return links[0] if links else None


def find_source(final_path):
    """production_ready/<vod>_<name>_final.mp4 -> visual_approved/<name>.mp4 (+ track file)"""
    real = os.path.realpath(final_path)
    base = os.path.basename(real)
    if base.endswith("_final.mp4"):
        base = base[: -len("_final.mp4")]
    vod, _, name = base.partition("_")
    for run in (os.path.dirname(os.path.dirname(real)), f"{vod}_final_review"):
        cand = os.path.join(run, "visual_approved", f"{name}.mp4")
        if os.path.exists(cand):
            vdir = os.path.join(run, "visual_approved", "vertical")
            tracks = sorted(glob.glob(os.path.join(vdir, f"{name}_track*.json")))
            tracks = [t for t in tracks if "decisions" not in t] + [t for t in tracks if "decisions" in t]
            return cand, name, (tracks[0] if tracks else None)
    return None, name, None


def final_parts(src, name):
    """The parts of the landscape clip the FINAL clip uses: internal cuts, else smart trim, else all."""
    vdir = os.path.join(os.path.dirname(src), "vertical")
    for fp in sorted(glob.glob(os.path.join(vdir, "internal_cuts", f"{name}_*cuts.json"))):
        try:
            j = json.load(open(fp))
            if j.get("applied") and j.get("parts"):
                return [(float(a), float(b)) for a, b in j["parts"]], "internal cuts"
        except Exception:
            pass
    for fp in sorted(glob.glob(os.path.join(vdir, "smart_trim", f"{name}_*trim.json"))):
        try:
            j = json.load(open(fp))
            if j.get("applied") and j.get("final_end"):
                return [(float(j["final_start"]), float(j["final_end"]))], "smart trim"
        except Exception:
            pass
    return None, "whole clip"


def cut_source(src, parts, dst):
    """Make a copy of the landscape clip containing only `parts` (video + audio)."""
    fc, ins = "", ""
    for i, (a, b) in enumerate(parts):
        fc += (f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}];"
               f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[a{i}];")
        ins += f"[v{i}][a{i}]"
    fc += f"{ins}concat=n={len(parts)}:v=1:a=1[v][a]"
    r = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", src, "-filter_complex", fc,
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "16",
                        "-c:a", "aac", dst], stdin=subprocess.DEVNULL)
    return r.returncode == 0


CAPTION_ABOVE_LINE = 0.025  # caption block's TOP edge sits this far (of the height) above the split line;
                            # longer captions grow DOWN into the gameplay, never up over faces


def _ass_playres_y(lines):
    for ln in lines:
        if ln.startswith("PlayResY:"):
            return float(ln.split(":", 1)[1])
    return 1080.0


def move_captions(ass_in, ass_out, top_frac):
    """Copy the clip's caption overlay, moving the captions up to the split line."""
    lines = open(ass_in, encoding="utf-8").read().splitlines()
    pry = _ass_playres_y(lines)
    new_mv = int(round(pry * (top_frac - CAPTION_ABOVE_LINE)))     # measured from the TOP now
    old_mv, fmt, out = None, None, []
    for ln in lines:
        if ln.startswith("Format:") and fmt is None and "MarginV" in ln:
            fmt = [x.strip() for x in ln.split(":", 1)[1].split(",")]
        if ln.startswith("Style:") and fmt:
            vals = ln.split(":", 1)[1].split(",")
            i = fmt.index("MarginV")
            old_mv = int(float(vals[i]))
            vals[i] = str(new_mv)
            if "Alignment" in fmt:                 # bottom-anchored (1/2/3) -> top-anchored (7/8/9)
                ai = fmt.index("Alignment")
                a = int(float(vals[ai]))
                vals[ai] = str({1: 7, 2: 8, 3: 9}.get(a, a))
            ln = "Style:" + ",".join(vals)
        elif ln.startswith("Dialogue:"):
            head, rest = ln.split(":", 1)
            f = rest.split(",", 9)
            if len(f) == 10 and f[7].strip() not in ("", "0"):
                f[7] = str(new_mv)
            f[-1] = re.sub(r"\\an([123])", lambda m: "\\an" + str(int(m.group(1)) + 6), f[-1])
            if old_mv is not None and "\\pos(" in f[-1]:
                def shift(m):
                    return f"\\pos({m.group(1)},{float(m.group(2)) - (new_mv - old_mv):.0f})"
                f[-1] = re.sub(r"\\pos\(([^,]+),([^)]+)\)", shift, f[-1])
            ln = head + ":" + ",".join(f)
        out.append(ln)
    open(ass_out, "w", encoding="utf-8").write("\n".join(out) + "\n")
    return old_mv, new_mv, pry


def burn_overlays(video, finish_dir, top_frac, dst):
    """Burn the clip's own captions (moved to the split line), hook and watermark onto `video`."""
    import shutil
    work = os.path.join(os.path.dirname(dst), "_overlays")
    os.makedirs(work, exist_ok=True)
    chain, notes = [], []
    cap = sorted(glob.glob(os.path.join(finish_dir, "captions", "*.ass")))
    hook = sorted(glob.glob(os.path.join(finish_dir, "hook", "*.ass")))
    wm = sorted(glob.glob(os.path.join(finish_dir, "watermark", "*.ass")))
    if cap:
        old_mv, new_mv, pry = move_captions(cap[0], os.path.join(work, "cap.ass"), top_frac)
        chain.append("ass=cap.ass")
        notes.append(f"captions moved from {100 * (1 - (old_mv or 0) / pry):.0f}% down to start at "
                     f"{100 * new_mv / pry:.1f}% (split line at {100 * top_frac:.0f}%), growing downward")
    if hook:
        shutil.copy(hook[0], os.path.join(work, "hook.ass")); chain.append("ass=hook.ass"); notes.append("hook as in the final")
    if wm:
        shutil.copy(wm[0], os.path.join(work, "wm.ass")); chain.append("ass=wm.ass"); notes.append("watermark as in the final")
    if not chain:
        return False, "no caption/hook/watermark overlays found for this clip"
    r = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", os.path.abspath(video),
                        "-vf", ",".join(chain), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "copy", os.path.abspath(dst)], cwd=work, stdin=subprocess.DEVNULL)
    shutil.rmtree(work, ignore_errors=True)
    return r.returncode == 0, "; ".join(notes)


def ass_time(t):
    h, m, sec = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def speech_share(finish_dir, dur):
    """Share of the final clip's time that has captions on screen."""
    cap = sorted(glob.glob(os.path.join(finish_dir, "captions", "*.ass")))
    if not cap or dur <= 0:
        return None
    spans = []
    for ln in open(cap[0], encoding="utf-8", errors="ignore"):
        if ln.startswith("Dialogue:"):
            f = ln.split(":", 1)[1].split(",", 9)
            try:
                spans.append((ass_time(f[1]), ass_time(f[2])))
            except Exception:
                pass
    spans.sort()
    covered, cur = 0.0, None
    for a, b in spans:
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            if cur:
                covered += cur[1] - cur[0]
            cur = [a, b]
    if cur:
        covered += cur[1] - cur[0]
    return min(1.0, covered / dur)


def motion_level(src, parts, samples_per_sec=4):
    """How much the picture changes (median + 80th percentile of frame difference, 0-255)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = parts or [(0, n / fps)]
    step = max(1, int(round(fps / samples_per_sec)))
    diffs, prev, i = [], None, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0 and any(a <= i / fps < b for a, b in keep):
            ok, fr = cap.retrieve()
            if ok:
                g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (96, 54)).astype(np.int16)
                if prev is not None:
                    diffs.append(float(np.abs(g - prev).mean()))
                prev = g
        elif i % step == 0:
            prev = None
        i += 1
    cap.release()
    if not diffs:
        return None, None
    return float(np.median(diffs)), float(np.percentile(diffs, 80))


def crop_mode_of(vdir, name):
    try:
        j = json.load(open(os.path.join(vdir, "_presence_summary.json")))
    except Exception:
        return "?"
    items = j if isinstance(j, list) else j.get("clips", j)
    rec = None
    if isinstance(items, dict):
        rec = items.get(name) or items.get(f"{name}.mp4")
    elif isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and name in json.dumps(it):
                rec = it
                break
    if isinstance(rec, dict):
        for k in ("crop_mode", "mode", "tracker_mode"):
            if k in rec:
                return str(rec[k])
    return "?"


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                       capture_output=True, text=True)
    j = json.loads(r.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    num, _, den = (s.get("r_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return int(s.get("width", 0)), int(s.get("height", 0)), fps, float(j.get("format", {}).get("duration", 0) or 0)


def pick_gameplay(explicit, dur):
    files = [explicit] if explicit else [f for f in glob.glob("backgrounds/*")
                                         if f.lower().endswith((".mp4", ".mkv", ".webm", ".mov"))]
    if not files:
        return None, 0.0
    used_path = "backgrounds/_used.json"
    try:
        used = json.load(open(used_path))
    except Exception:
        used = []
    random.shuffle(files)
    best = None
    for f in files:
        gdur = probe(f)[3]
        if gdur <= dur + 2:
            if best is None:
                best = (f, 0.0)
            continue
        recent = [(u["start"], u["end"]) for u in used[-30:] if u["file"] == f]
        for _ in range(40):
            s = random.uniform(0, gdur - dur - 1)
            if all(s + dur <= a or s >= b for a, b in recent):
                best = (f, s)
                break
        if best and best[0] == f:
            break
    if best is None:
        best = (files[0], 0.0)
    used.append({"file": best[0], "start": round(best[1], 1), "end": round(best[1] + dur, 1)})
    try:
        os.makedirs("backgrounds", exist_ok=True)
        json.dump(used[-200:], open(used_path, "w"), indent=1)
    except Exception:
        pass
    return best


def load_track(track_path, n):
    with open(track_path) as f:
        raw = {int(k): v for k, v in json.load(f).items()}
    keys = sorted(raw)
    data, ki = [], 0
    for i in range(n):
        while ki < len(keys) - 1 and keys[ki + 1] <= i:
            ki += 1
        data.append(raw[keys[ki]])
    return data


def smart_plan(data, src_w, crop_w):
    """Per-frame crop centres for FILL, or None + reason if FULL is the safer choice."""
    bad, reasons = 0, {"lost": 0, "transition/close-up": 0, "people too far apart": 0}
    for d in data:
        if d.get("lost"):
            bad += 1; reasons["lost"] += 1
        elif d.get("force_full"):
            bad += 1; reasons["transition/close-up"] += 1
        elif d.get("mode") == "joint" and d.get("target_width") and float(d["target_width"]) * 1.1 > crop_w:
            bad += 1; reasons["people too far apart"] += 1
    share = bad / max(1, len(data))
    if share > BAD_LIMIT:
        top = max(reasons, key=reasons.get)
        return None, f"{100 * share:.0f}% of frames can't keep the subject in a fill crop (mostly: {top})"
    centres, sx, last_shot = [], None, None
    half = crop_w / 2
    for d in data:
        tx = float(d.get("pos", [src_w / 2])[0]) if not d.get("lost") else (sx if sx is not None else src_w / 2)
        shot = d.get("shot")
        if sx is None or shot != last_shot:
            sx = tx
        else:
            sx += (tx - sx) * SMOOTH
        last_shot = shot
        ty = float((d.get("pos") or [0, 0])[1] if len(d.get("pos") or []) > 1 else 0) or None
        centres.append((min(max(sx, half), src_w - half), ty, sx))
    return centres, f"subject kept in frame ({100 * (1 - share):.0f}% of frames fit)"


DARK_LEVEL, DARK_SHARE = 24, 0.75   # crop counts as "dark" if 75%+ of it is near-black
MAX_UPSCALE = 3.5                    # never zoom in more than this (keeps it from getting too blurry)
ZOOM_EASE = 0.06


def lit_box(gray_small, scale_x, scale_y, x_off):
    """Bounding box (in source pixels) of the BIGGEST lit block (the facecam) in a mostly-black
    frame -- small lit bits like viewer counts, logos or chat text are ignored."""
    import cv2
    import numpy as np
    m = (gray_small >= DARK_LEVEL).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[k]
    if area < 0.01 * m.size:          # nothing big enough to be a facecam
        return None
    return (x_off + x * scale_x, y * scale_y, x_off + (x + w) * scale_x, (y + h) * scale_y)


def render_fill_top(src, centres, top_h, out_path, fps):
    """FILL crop following the subject. While the crop is mostly black (game screen black /
    loading, only the facecam lit) it zooms onto the LIT area (the facecam) so it fills the
    top half, then eases back out when the screen comes back."""
    import cv2
    cap = cv2.VideoCapture(src)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = min(src_w, int(round(src_h * OUT_W / top_h)))
    aspect = top_h / OUT_W
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, top_h))
    i, dark_frames = 0, 0
    rx, ry, rw = None, src_h / 2, float(crop_w)      # current crop: centre x, centre y, width
    last_box = None
    SW, SH = 96, 54
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cx, _ty, _raw = centres[min(i, len(centres) - 1)]
        x0 = int(max(0, min(src_w - crop_w, round(cx - crop_w / 2))))
        small = cv2.resize(cv2.cvtColor(frame[:, x0:x0 + crop_w], cv2.COLOR_BGR2GRAY), (SW, SH))
        dark = (small < DARK_LEVEL).mean() >= DARK_SHARE
        tx, ty, tw = cx, src_h / 2, float(crop_w)
        if dark:
            box = lit_box(small, crop_w / SW, src_h / SH, x0)
            if box:
                last_box = box
                dark_frames += 1
                bw, bh = box[2] - box[0], box[3] - box[1]
                tw = max(bw * 1.08, bh * 1.08 / aspect, crop_w / MAX_UPSCALE)
                tw = min(tw, float(crop_w))
                tx, ty = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if rx is None:
            rx, ry, rw = tx, ty, tw
        else:
            rx += (tx - rx) * ZOOM_EASE
            ry += (ty - ry) * ZOOM_EASE
            rw += (tw - rw) * ZOOM_EASE
        cw = rw
        ch = min(float(src_h), cw * aspect)
        x = max(0.0, min(src_w - cw, rx - cw / 2))
        y = max(0.0, min(src_h - ch, ry - ch / 2))
        roi = frame[int(y):int(y + ch), int(x):int(x + cw)]
        interp = cv2.INTER_AREA if cw >= crop_w * 0.95 else cv2.INTER_CUBIC
        out = cv2.resize(roi, (OUT_W, top_h), interpolation=interp)
        if last_box is not None and cw < crop_w * 0.9:
            # black bands above/below the facecam -> fill with a blurred copy of the facecam
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            rows = g.max(axis=1) < DARK_LEVEL
            cols = g.max(axis=0) < DARK_LEVEL
            if rows.mean() > 0.04 or cols.mean() > 0.04:
                bx0, by0, bx1, by1 = [int(v) for v in last_box]
                cam = frame[max(0, by0):by1, max(0, bx0):bx1]
                if cam.size:
                    ch_, cw_ = cam.shape[:2]
                    sc = max(OUT_W / cw_, top_h / ch_)
                    big = cv2.resize(cam, (int(cw_ * sc) + 1, int(ch_ * sc) + 1))
                    oy, ox = (big.shape[0] - top_h) // 2, (big.shape[1] - OUT_W) // 2
                    bg = cv2.GaussianBlur(big[oy:oy + top_h, ox:ox + OUT_W], (0, 0), 25)
                    bg = (bg * 0.6).astype(out.dtype)
                    out[rows] = bg[rows]
                    out[:, cols] = bg[:, cols]
        wr.write(out)
        i += 1
    cap.release(); wr.release()
    return i, dark_frames


def compose(top_input_args, top_filter, src, game, goff, dur, top_h, dst):
    bot_h = OUT_H - top_h
    if game:
        gin = ["-ss", f"{goff:.1f}", "-stream_loop", "-1", "-i", game]
    else:
        gin = ["-f", "lavfi", "-i", f"testsrc2=size={OUT_W}x{bot_h}:rate=30:duration={dur:.2f}"]
    fc = (f"{top_filter};"
          f"[2:v]scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,crop={OUT_W}:{bot_h},setsar=1,fps=30[bot];"
          f"[top]fps=30[top30];[top30][bot]vstack=inputs=2,"
          f"drawbox=x=0:y={top_h - 4}:w={OUT_W}:h=8:color=black:t=fill[v]")
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", *top_input_args, "-t", f"{dur:.2f}", "-i", src,
           *gin, "-filter_complex", fc, "-map", "[v]", "-map", "1:a?", "-t", f"{dur:.2f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-shortest", dst]
    return subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", help="a finished clip (default: newest approved)")
    ap.add_argument("--gameplay", help="background video (default: random file from backgrounds/)")
    ap.add_argument("--gameplay-start", type=float, default=None,
                    help="start the gameplay at this second (same footage for fair comparisons)")
    ap.add_argument("--no-overlays", action="store_true",
                    help="skip adding the clip's captions/hook/watermark")
    ap.add_argument("--measure", action="store_true",
                    help="only MEASURE the clip for auto mode (speech, motion, on-camera, crop mode); no video")
    ap.add_argument("--layout", default="smart", choices=["smart", "fill", "full", "all"])
    ap.add_argument("--top", type=float, default=0.45, help="share of height for the stream (0.45 default, 0.5 = half)")
    ap.add_argument("--x", type=float, default=0.5, help="FILL layout: fixed position 0=left 1=right")
    ap.add_argument("--max-seconds", type=float, default=60)
    args = ap.parse_args()

    final = args.final or newest_approved()
    if not final:
        sys.exit("ERROR: no approved clip found -- pass --final <path>.")
    src, name, track = find_source(final)
    if not src:
        sys.exit(f"ERROR: couldn't find the landscape original for {final}")
    out = os.path.join("split_previews", name)
    os.makedirs(out, exist_ok=True)
    parts, how = final_parts(src, name)
    orig_src = src
    src_w, src_h, fps, sdur = probe(src)
    if parts:
        cut = os.path.join(out, "_final_timeline.mp4")
        if cut_source(src, parts, cut):
            src = cut
            print("timeline: same as the final clip (" + how + ": "
                  + " + ".join(f"{a:.1f}-{b:.1f}s" for a, b in parts) + ")")
        else:
            parts, how = None, "whole clip (cutting failed)"
    if not parts:
        print(f"timeline: {how}")
    src_w, src_h, fps, sdur = probe(src)
    dur = min(sdur or args.max_seconds, args.max_seconds)
    top_h = int(round(OUT_H * max(0.3, min(0.7, args.top)) / 2) * 2)
    crop_w = min(src_w, int(round(src_h * OUT_W / top_h)))
    if args.measure:
        vdir = os.path.join(os.path.dirname(orig_src), "vertical")
        finish_dir = os.path.join(vdir, "finished", name, "finish")
        sp = speech_share(finish_dir, dur)
        mmed, m80 = motion_level(orig_src, parts)
        fit = None
        if track:
            ofps = probe(orig_src)[2]
            ft = load_track(track, int(round(probe(orig_src)[3] * ofps)) + 2)
            if parts:
                tr = []
                for a, b in parts:
                    tr += ft[int(a * ofps):int(b * ofps)]
                ft = tr or ft
            bad = sum(1 for d in ft if d.get("lost") or d.get("force_full"))
            fit = 1 - bad / max(1, len(ft))
        row = {"clip": name, "dur": round(dur, 1), "speech": None if sp is None else round(sp, 2),
               "motion_med": None if mmed is None else round(mmed, 1),
               "motion_p80": None if m80 is None else round(m80, 1),
               "on_camera": None if fit is None else round(fit, 2), "crop_mode": crop_mode_of(vdir, name)}
        print("MEASURE " + json.dumps(row))
        with open("split_previews/_measurements.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        for t in ("_final_timeline.mp4",):
            if os.path.exists(os.path.join(out, t)):
                os.remove(os.path.join(out, t))
        return
    game, goff = pick_gameplay(args.gameplay, dur)
    if game and args.gameplay_start is not None:
        goff = args.gameplay_start
    tag = f"_top{int(round(100 * top_h / OUT_H))}" if abs(top_h / OUT_H - 0.45) > 0.01 else ""
    print(f"clip:     {final}\nsource:   {orig_src} ({dur:.0f}s used, {src_w}x{src_h})")
    print(f"tracking: {track or 'none found'}")
    print(f"gameplay: {game + f' from {goff:.0f}s' if game else '(test pattern -- add videos to backgrounds/)'}")

    full_f = (f"[0:v]split[a][b];[a]scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
              f"crop={OUT_W}:{top_h},boxblur=25:2[bg];[b]scale={OUT_W}:-2[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[top]")
    fixed_f = (f"[0:v]scale=-2:{top_h},crop={OUT_W}:{top_h}:(iw-{OUT_W})*{max(0.0, min(1.0, args.x)):.3f}:0,"
               f"setsar=1[top]")
    jobs = []
    if args.layout in ("smart", "all"):
        if not track:
            print("smart:    no tracking file for this clip -> using FULL")
            jobs.append(("split_smart", ["-t", f"{dur:.2f}", "-i", src], full_f))
        else:
            ofps = probe(orig_src)[2]
            full_track = load_track(track, int(round(probe(orig_src)[3] * ofps)) + 2)
            if parts:   # keep only the track frames of the parts the final uses
                tr = []
                for a, b in parts:
                    tr += full_track[int(a * ofps):int(b * ofps)]
                full_track = tr or full_track
            centres, why = smart_plan(full_track, src_w, crop_w)
            if centres is None:
                print(f"smart:    FULL -- {why}")
                jobs.append(("split_smart", ["-t", f"{dur:.2f}", "-i", src], full_f))
            else:
                print(f"smart:    FILL following the subject -- {why}")
                tmp = os.path.join(out, "_smart_top.mp4")
                _, dark = render_fill_top(src, centres, top_h, tmp, fps)
                if dark:
                    print(f"dark:     {dark / fps:.1f}s where the screen around the streamer was mostly black"
                          " -> zoomed onto the lit part (facecam) there")
                jobs.append(("split_smart", ["-i", tmp], "[0:v]setsar=1[top]"))
    if args.layout in ("fill", "all"):
        jobs.append(("split_fill", ["-t", f"{dur:.2f}", "-i", src], fixed_f))
    if args.layout in ("full", "all"):
        jobs.append(("split_full", ["-t", f"{dur:.2f}", "-i", src], full_f))

    for fname, top_in, top_f in jobs:
        dst = os.path.join(out, f"{fname}{tag}.mp4")
        ok = compose(top_in, top_f, src, game, goff, dur, top_h, dst)
        print(("saved " if ok else "FAILED ") + dst)
    finish_dir = os.path.join(os.path.dirname(orig_src), "vertical", "finished", name, "finish")
    if not args.no_overlays and os.path.isdir(finish_dir):
        for fname, _, _ in jobs:
            plain = os.path.join(out, f"{fname}{tag}.mp4")
            if os.path.exists(plain):
                dst = os.path.join(out, f"{fname}{tag}_finished.mp4")
                ok, note = burn_overlays(plain, finish_dir, top_h / OUT_H, dst)
                print(("saved " if ok else "overlays FAILED ") + dst + f"  ({note})")
    elif not args.no_overlays:
        print("overlays: none found for this clip (finished/<clip>/finish/ missing) -- plain preview only")
    for t in ("_smart_top.mp4", "_final_timeline.mp4"):
        if os.path.exists(os.path.join(out, t)):
            os.remove(os.path.join(out, t))
    print(f"\nopen {out}")


if __name__ == "__main__":
    main()
