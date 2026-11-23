#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


PLAYRES_X = 608
PLAYRES_Y = 1080
FONT = "Arial"
FONT_SIZE = 54
MARGIN_V = 305

MIN_DISPLAY = 0.22
GAP = 0.02


EMPHASIS_TERMS = [
    "MOM",
    "MOTHER",
    "DAD",
    "FATHER",
    "SECURITY",
    "POLICE",
    "MONEY",
    "WIN",
    "WON",
    "LOSE",
    "LOST",
    "CRAZY",
    "INSANE",
    "FIGHT",
    "HEART",
    "WORK",
]


CONNECTORS = {
    "FOR",
    "TO",
    "FROM",
    "WITH",
    "ABOUT",
    "YOUR",
    "THE",
    "A",
    "AN",
    "AND",
    "BUT",
    "OR",
}


# ---------------- on-screen profanity masking ----------------
# CAPTION_MASK=all (default) | slurs | off. Only the rendered text changes.
SLUR_WORDS = ["nigga", "nigger", "faggot", "fag", "retard"]
SWEAR_ANYWHERE = ["fuck", "shit", "bitch"]          # also inside words: motherfucker, bullshit
SWEAR_WHOLE = ["pussy", "cunt", "whore", "dick", "hoe"]  # whole word (+s) only


def _star(s):
    return s[0] + "*" * (len(s) - 1)


def mask_words(text):
    mode = os.environ.get("CAPTION_MASK", "all").strip().lower()
    if mode == "off":
        return text

    def one(m):
        w = m.group(0)
        lw = w.lower()
        for root in SLUR_WORDS:
            if lw.startswith(root) and lw[len(root):] in ("", "s", "z", "as", "az", "ed", "ers", "ing"):
                return _star(w[:len(root)]) + w[len(root):]
        if mode != "all":
            return w
        for root in SWEAR_ANYWHERE:
            i = lw.find(root)
            if i != -1:
                return w[:i] + _star(w[i:i + len(root)]) + w[i + len(root):]
        for root in SWEAR_WHOLE:
            if lw in (root, root + "s", root + "es"):
                return _star(w[:len(root)]) + w[len(root):]
        return w

    return re.sub(r"[A-Za-z']+", one, text)


def run(cmd):
    print()
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ass_time(seconds):
    seconds = max(0.0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    return f"{h}:{m:02d}:{s:05.2f}"


def escape_ass(text):
    return (
        text
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def split_phrase(text):
    words = text.strip().split()

    if not words:
        return []

    chunks = []
    i = 0

    while i < len(words):

        remaining = len(words) - i

        if remaining <= 5:
            n = remaining

        elif remaining == 6:
            n = 4

        elif remaining == 7:
            n = 4

        else:
            n = 4

        chunk = words[i:i+n]

        if (
            len(chunk) >= 3
            and re.sub(
                r"[^A-Z']",
                "",
                chunk[-1].upper(),
            )
            in CONNECTORS
            and i + n < len(words)
        ):
            n -= 1
            chunk = words[i:i+n]

        chunks.append(
            " ".join(chunk)
        )

        i += n

    return chunks


def emphasize(text):
    styled = escape_ass(text.upper())

    for term in EMPHASIS_TERMS:

        pattern = (
            r"(?<![A-Z0-9])"
            + re.escape(term)
            + r"(?![A-Z0-9])"
        )

        match = re.search(
            pattern,
            styled,
        )

        if not match:
            continue

        styled = (
            styled[:match.start()]
            + r"{\c&H00FFFF&}"
            + match.group(0)
            + r"{\c&HFFFFFF&}"
            + styled[match.end():]
        )

        break

    return styled


BLANK_MAX_SECONDS = 1.5   # a low-confidence line this short may be left blank
BLANK_MAX_SHARE = 0.10    # ...if all such lines together are at most this share of the captions


def apply_caption_blanking(regions):
    """Drop short low-confidence lines (shown as no caption). Returns (kept, blanked)."""
    if os.environ.get("CAPTION_BLANK_UNCERTAIN", "1").strip() == "0":
        return regions, []

    def _d(x):
        try:
            return max(0.0, float(x["end"]) - float(x["start"]))
        except Exception:
            return 0.0
    total = sum(_d(x) for x in regions)
    cand = [x for x in regions if x.get("confidence") == "low" and 0 < _d(x) <= BLANK_MAX_SECONDS]
    if not cand or total <= 0 or sum(_d(x) for x in cand) > BLANK_MAX_SHARE * total:
        return regions, []
    ids = {id(x) for x in cand}
    return [x for x in regions if id(x) not in ids], cand


def build_events(regions):
    events = []

    for region in regions:

        start = float(region["start"])
        end = float(region["end"])

        text = (
            region.get(
                "resolved_text",
                "",
            )
            .strip()
        )

        if not text or end <= start:
            continue

        chunks = split_phrase(mask_words(text))

        if not chunks:
            continue

        weights = [
            max(
                1,
                len(chunk.split()),
            )
            for chunk in chunks
        ]

        segment_duration = end - start

        ideal_durations = []

        for weight in weights:

            ideal = (
                0.48 * weight
                + 0.15
            )

            ideal = max(
                0.35,
                ideal,
            )

            ideal = min(
                2.20,
                ideal,
            )

            ideal_durations.append(
                ideal
            )

        ideal_total = sum(
            ideal_durations
        )

        sparse_segment = (
            segment_duration
            > ideal_total * 1.45
        )

        cursor = start

        if sparse_segment:

            for chunk, beat_duration in zip(
                chunks,
                ideal_durations,
            ):

                chunk_end = min(
                    end,
                    cursor + beat_duration,
                )

                if chunk_end <= cursor:
                    continue

                events.append({
                    "start": cursor,
                    "end": chunk_end,
                    "text": chunk,
                    "status": region.get(
                        "status",
                        "approved",
                    ),
                    "confidence": region.get(
                        "confidence",
                        "unknown",
                    ),
                })

                cursor = chunk_end

        else:

            total_weight = sum(weights)
            duration_left = segment_duration

            for i, (chunk, weight) in enumerate(
                zip(chunks, weights)
            ):

                if i == len(chunks) - 1:
                    chunk_end = end

                else:
                    share = (
                        weight
                        / total_weight
                    )

                    chunk_end = (
                        cursor
                        + duration_left
                        * share
                    )

                events.append({
                    "start": cursor,
                    "end": chunk_end,
                    "text": chunk,
                    "status": region.get(
                        "status",
                        "approved",
                    ),
                    "confidence": region.get(
                        "confidence",
                        "unknown",
                    ),
                })

                used = chunk_end - cursor

                cursor = chunk_end
                duration_left -= used
                total_weight -= weight

    return events


def repair_events(events):
    events.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    repaired = []

    for event in events:

        start = event["start"]
        end = event["end"]

        if repaired:

            prev = repaired[-1]

            if start < prev["end"] + GAP:

                midpoint = (
                    prev["end"]
                    + start
                ) / 2.0

                prev_min_end = (
                    prev["start"]
                    + MIN_DISPLAY
                )

                current_max_start = (
                    end
                    - MIN_DISPLAY
                )

                if (
                    prev_min_end
                    <= current_max_start
                ):

                    boundary = max(
                        prev_min_end,
                        min(
                            midpoint,
                            current_max_start,
                        ),
                    )

                    prev["end"] = boundary
                    start = boundary + GAP

                else:

                    start = (
                        prev["end"]
                        + GAP
                    )

        if end - start < MIN_DISPLAY:
            end = start + MIN_DISPLAY

        event["start"] = start
        event["end"] = end

        repaired.append(event)

    return repaired


def validate(events):
    invalid = 0
    overlaps = 0

    for e in events:

        if e["end"] <= e["start"]:
            invalid += 1

    for a, b in zip(
        events,
        events[1:],
    ):

        if a["end"] > b["start"] + 0.01:
            overlaps += 1

    print()
    print("===== RENDER VALIDATION =====")
    print("Caption events:", len(events))
    print("Invalid:", invalid)
    print("Overlaps:", overlaps)

    if invalid or overlaps:
        raise SystemExit(
            "ERROR: caption timing validation failed"
        )


CAPTION_BLOCK = 0.13      # height of a two-line caption block (fraction of the frame)
FACE_HIT_SHARE = 0.30     # move captions if a face is in the caption area this often
CAPTION_MIN_TOP = 0.17    # never push captions up into the hook area


def _find_faces(video, n=16):
    """Per sampled frame: list of (top, bottom) of faces as fractions of the height."""
    import cv2
    detect = None
    try:
        import face_recognition

        def detect(rgb_small):
            return [(t, b) for (t, r, b, l) in face_recognition.face_locations(rgb_small)]
    except Exception:
        xml = os.path.join(getattr(cv2, "data", None).haarcascades if hasattr(cv2, "data") else "",
                           "haarcascade_frontalface_default.xml")
        cas = cv2.CascadeClassifier(xml) if os.path.exists(xml) else None
        if cas is None or cas.empty():
            return None

        def detect(rgb_small):
            g = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
            w = g.shape[1]
            return [(y, y + h) for (x, y, ww, h) in cas.detectMultiScale(g, 1.1, 6, minSize=(int(w * .1), int(w * .1)))]
    cap = cv2.VideoCapture(str(video))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + .5) * tot / n))
        ok, fr = cap.read()
        if not ok:
            continue
        small = cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        hgt = rgb.shape[0]
        out.append([(t / hgt, b / hgt) for t, b in detect(rgb) if (b - t) / hgt > 0.04])
    cap.release()
    return out


def plan_caption_position(video):
    """Returns (new MARGIN_V or None, explanation)."""
    if os.environ.get("CAPTION_FACE_AVOID", "1").strip() == "0":
        return None, "face avoidance off (CAPTION_FACE_AVOID=0)"
    try:
        frames = _find_faces(video)
    except Exception as e:
        return None, f"face check failed ({e.__class__.__name__}); default position"
    if not frames:
        return None, "no face detector available / no frames; default position"
    bottom = 1 - MARGIN_V / PLAYRES_Y
    top = bottom - CAPTION_BLOCK
    hits = [min(a for a, b in f if a < bottom and b > top) for f in frames
            if any(a < bottom and b > top for a, b in f)]
    share = len(hits) / len(frames)
    if share < FACE_HIT_SHARE:
        return None, f"face in the caption area in {share:.0%} of frames; default position"
    hits.sort()
    face_top = hits[max(0, int(len(hits) * 0.2) - 1)]
    new_bottom = face_top - 0.02
    if new_bottom - CAPTION_BLOCK < CAPTION_MIN_TOP:
        return None, f"face in the caption area ({share:.0%}) but too high to fit captions above; default"
    return int(round((1 - new_bottom) * PLAYRES_Y)), (
        f"face in the caption area in {share:.0%} of frames -> captions moved above the face "
        f"(bottom edge {100 * bottom:.0f}% -> {100 * new_bottom:.0f}% down)")


def write_ass(events, path):

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAYRES_X}
PlayResY: {PLAYRES_Y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: TikTok,{FONT},{FONT_SIZE},&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,1,2,35,35,{MARGIN_V},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(header)

        for event in events:

            f.write(
                "Dialogue: 0,"
                f"{ass_time(event['start'])},"
                f"{ass_time(event['end'])},"
                "TikTok,,0,0,0,,"
                f"{emphasize(event['text'])}\n"
            )


def write_review_report(
    regions,
    path,
):

    review = [
        x
        for x in regions
        if x.get("status")
        == "review_required"
    ]

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "CAPTION REVIEW REPORT\n"
        )

        f.write(
            "=====================\n\n"
        )

        f.write(
            f"Regions requiring review: "
            f"{len(review)}\n\n"
        )

        for i, x in enumerate(
            review,
            1,
        ):

            f.write(
                f"[{i}] "
                f"{x['start']:.2f}s -> "
                f"{x['end']:.2f}s\n"
            )

            f.write(
                f"Confidence: "
                f"{x.get('confidence')}\n"
            )

            f.write(
                "Resolved text:\n"
            )

            f.write(
                x.get(
                    "resolved_text",
                    "",
                )
                + "\n"
            )

            if x.get("reason"):

                f.write(
                    "Resolver note:\n"
                )

                f.write(
                    x["reason"]
                    + "\n"
                )

            f.write("\n")


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
    )

    ap.add_argument(
        "--resolved",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    video = Path(args.video)
    resolved_path = Path(
        args.resolved
    )
    outdir = Path(
        args.output_dir
    )

    if not video.exists():
        raise SystemExit(
            f"ERROR: video not found: "
            f"{video}"
        )

    if not resolved_path.exists():
        raise SystemExit(
            f"ERROR: resolved transcript "
            f"not found: {resolved_path}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    regions = json.load(
        open(
            resolved_path,
            encoding="utf-8",
        )
    )

    regions, _blanked = apply_caption_blanking(regions)
    for _b in _blanked:
        print(f"Caption left blank (uncertain): {float(_b['start']):.2f}-{float(_b['end']):.2f}s "
              f"{_b.get('resolved_text', '')!r}")

    events = build_events(
        regions
    )

    events = repair_events(
        events
    )

    validate(
        events
    )

    stem = video.stem

    ass_path = (
        outdir
        / f"{stem}_resolved.ass"
    )

    video_out = (
        outdir
        / f"{stem}_resolved_captioned.mp4"
    )

    report_path = (
        outdir
        / "caption_review_report.txt"
    )

    global MARGIN_V
    _mv, _why = plan_caption_position(video)
    print("Caption position:", _why)
    try:
        with open(outdir / "caption_position.json", "w") as _f:
            json.dump({"default_margin_v": MARGIN_V, "margin_v": _mv or MARGIN_V, "why": _why}, _f, indent=2)
    except Exception:
        pass
    if _mv:
        MARGIN_V = _mv

    write_ass(
        events,
        ass_path,
    )

    write_review_report(
        regions,
        report_path,
    )

    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"setpts=PTS-STARTPTS,ass={ass_path}",
        "-c:v",
        "libx264", "-pix_fmt", "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(video_out),
    ])

    review_count = sum(
        1
        for x in regions
        if x.get("status")
        == "review_required"
    )

    print()
    print(
        "===== RESOLVED CAPTION RENDER COMPLETE ====="
    )
    print(
        "Video:",
        video_out,
    )
    print(
        "ASS:",
        ass_path,
    )
    print(
        "Review report:",
        report_path,
    )
    print(
        "Review-required regions:",
        review_count,
    )


if __name__ == "__main__":
    main()
