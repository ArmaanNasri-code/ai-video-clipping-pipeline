#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# Frozen visual style from the benchmark.
PLAYRES_X = 608
PLAYRES_Y = 1080
FONT = "Arial"
FONT_SIZE = 54
MARGIN_V = 305

MIN_DISPLAY = 0.22
HANDOFF_GAP = 0.02


# Generic high-signal words only.
# We can make this smarter with an LLM later.
EMPHASIS_TERMS = [
    "MOM",
    "MOTHER",
    "DAD",
    "FATHER",
    "MONEY",
    "WIN",
    "WON",
    "LOSE",
    "LOST",
    "WORK",
    "HEART",
    "CRAZY",
    "INSANE",
    "SECURITY",
    "POLICE",
    "FIGHT",
    "GO",
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


def run(cmd):
    print()
    print(
        "$",
        " ".join(str(x) for x in cmd),
    )

    subprocess.run(
        cmd,
        check=True,
    )


def ass_time(seconds):
    seconds = max(
        0.0,
        float(seconds),
    )

    h = int(
        seconds // 3600
    )

    m = int(
        (seconds % 3600) // 60
    )

    s = (
        seconds % 60
    )

    return (
        f"{h}:"
        f"{m:02d}:"
        f"{s:05.2f}"
    )


def escape_ass(text):
    return (
        text
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def normalize_plain(text):
    text = re.sub(
        r"\{[^}]*\}",
        "",
        text,
    )

    return (
        " ".join(
            text.upper().split()
        )
    )


def collapse_repeats(words):
    out = []

    previous = None
    count = 0

    for word in words:
        key = re.sub(
            r"[^A-Z0-9']",
            "",
            word.upper(),
        )

        if key == previous:
            count += 1
        else:
            previous = key
            count = 1

        # Preserve repetition energy,
        # but stop absurd ASR loops.
        if count <= 3:
            out.append(word)

    return out


def split_phrase(text):
    words = (
        text.strip().split()
    )

    words = collapse_repeats(
        words
    )

    if not words:
        return []

    chunks = []
    i = 0

    while i < len(words):

        remaining = (
            len(words) - i
        )

        if remaining <= 5:
            n = remaining

        elif remaining == 6:
            n = 4

        elif remaining == 7:
            n = 4

        elif remaining == 8:
            n = 4

        else:
            n = 4

        chunk_words = (
            words[i:i+n]
        )

        if (
            len(chunk_words) >= 3
            and re.sub(
                r"[^A-Z']",
                "",
                chunk_words[-1].upper(),
            )
            in CONNECTORS
            and i + n < len(words)
        ):
            n -= 1

            chunk_words = (
                words[i:i+n]
            )

        chunks.append(
            " ".join(
                chunk_words
            )
        )

        i += n

    return chunks


def is_chunk_hallucination(seg):
    text = (
        seg.get(
            "text",
            "",
        )
        .strip()
        .lower()
    )

    start = float(
        seg.get(
            "start",
            0,
        )
    )

    # Known Whisper chunk-boundary artifact:
    # "Thank you." starting exactly at 30s boundaries.
    boundary = (
        abs(
            start / 30
            - round(
                start / 30
            )
        )
        < 0.02
    )

    if (
        boundary
        and text
        in {
            "thank you",
            "thank you.",
        }
    ):
        return True

    return False


def build_events(transcript):
    events = []

    removed_hallucinations = 0

    for seg in transcript.get(
        "segments",
        [],
    ):

        if is_chunk_hallucination(
            seg
        ):
            removed_hallucinations += 1
            continue

        start = float(
            seg["start"]
        )

        end = float(
            seg["end"]
        )

        text = (
            seg.get(
                "text",
                "",
            )
            .strip()
        )

        if (
            not text
            or end <= start
        ):
            continue

        chunks = split_phrase(
            text
        )

        if not chunks:
            continue

        weights = [
            max(
                1,
                len(
                    chunk.split()
                ),
            )
            for chunk in chunks
        ]

        segment_duration = (
            end - start
        )

        # Estimate how long each caption beat should
        # naturally remain visible.
        #
        # Sparse ASR segments sometimes span many seconds
        # even though they contain only a few spoken words.
        # In those cases, do NOT stretch the caption across
        # the entire ASR window.
        ideal_durations = []

        for weight in weights:

            ideal = (
                0.48 * weight
                + 0.15
            )

            ideal = max(
                MIN_DISPLAY,
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

        # Dense speech keeps the old proportional behavior.
        # Sparse speech gets capped caption durations,
        # leaving real silence blank.
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
                    "text": chunk.upper(),
                })

                cursor = chunk_end

        else:

            duration = (
                segment_duration
            )

            total_weight = sum(
                weights
            )

            for index, (
                chunk,
                weight,
            ) in enumerate(
                zip(
                    chunks,
                    weights,
                )
            ):

                if (
                    index
                    == len(chunks) - 1
                ):
                    chunk_end = end

                else:
                    chunk_end = (
                        cursor
                        + duration
                        * (
                            weight
                            / total_weight
                        )
                    )

                    duration -= (
                        chunk_end
                        - cursor
                    )

                    total_weight -= (
                        weight
                    )

                events.append({
                    "start": cursor,
                    "end": chunk_end,
                    "text": chunk.upper(),
                })

                cursor = chunk_end

    print(
        "Removed chunk hallucinations:",
        removed_hallucinations,
    )

    return events


def dedupe_events(events):
    deduped = []

    removed = 0

    for event in events:

        if deduped:

            prev = deduped[-1]

            same_text = (
                normalize_plain(
                    prev["text"]
                )
                ==
                normalize_plain(
                    event["text"]
                )
            )

            same_time = (
                abs(
                    prev["start"]
                    - event["start"]
                )
                < 0.05
                and
                abs(
                    prev["end"]
                    - event["end"]
                )
                < 0.05
            )

            if (
                same_text
                and same_time
            ):
                removed += 1
                continue

        deduped.append(
            event
        )

    print(
        "Removed exact duplicates:",
        removed,
    )

    return deduped


def repair_overlaps(events):

    fixes = 0

    events.sort(
        key=lambda e:
        (
            e["start"],
            e["end"],
        )
    )

    for i in range(
        len(events) - 1
    ):

        a = events[i]
        b = events[i + 1]

        overlap = (
            a["end"]
            - b["start"]
        )

        if overlap <= 0.01:
            continue

        boundary = (
            a["end"]
            + b["start"]
        ) / 2.0

        min_boundary = (
            a["start"]
            + MIN_DISPLAY
        )

        max_boundary = (
            b["end"]
            - MIN_DISPLAY
            - HANDOFF_GAP
        )

        if (
            max_boundary
            >= min_boundary
        ):

            boundary = max(
                min_boundary,
                min(
                    boundary,
                    max_boundary,
                ),
            )

            a["end"] = boundary

            b["start"] = (
                boundary
                + HANDOFF_GAP
            )

        else:

            # Very tight case:
            # preserve ordering rather
            # than allowing stacking.
            boundary = (
                a["start"]
                + b["end"]
                - HANDOFF_GAP
            ) / 2.0

            a["end"] = max(
                a["start"] + 0.10,
                boundary,
            )

            b["start"] = min(
                b["end"] - 0.10,
                boundary + HANDOFF_GAP,
            )

        fixes += 1

    print(
        "Overlap repairs:",
        fixes,
    )

    return events


def emphasize(text):

    styled = escape_ass(
        text
    )

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

        highlighted = (
            r"{\c&H00FFFF&}"
            + match.group(0)
            + r"{\c&HFFFFFF&}"
        )

        styled = (
            styled[
                :match.start()
            ]
            + highlighted
            + styled[
                match.end():
            ]
        )

        break

    return styled


def write_ass(
    events,
    output_path,
):

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
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            header
        )

        for event in events:

            f.write(
                "Dialogue: 0,"
                f"{ass_time(event['start'])},"
                f"{ass_time(event['end'])},"
                "TikTok,,0,0,0,,"
                f"{emphasize(event['text'])}\n"
            )


def validate(events):

    overlaps = 0
    invalid = 0

    for event in events:

        if (
            event["end"]
            <= event["start"]
        ):
            invalid += 1

    for a, b in zip(
        events,
        events[1:],
    ):

        if (
            a["end"]
            - b["start"]
            > 0.01
        ):
            overlaps += 1

    print()
    print(
        "===== CAPTION VALIDATION ====="
    )
    print(
        "Events:",
        len(events),
    )
    print(
        "Invalid durations:",
        invalid,
    )
    print(
        "Overlap count:",
        overlaps,
    )

    if (
        invalid
        or overlaps
    ):
        raise SystemExit(
            "ERROR: caption validation failed"
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
    )

    ap.add_argument(
        "--force-transcribe",
        action="store_true",
    )

    args = ap.parse_args()

    video = Path(
        args.video
    )

    if not video.exists():
        raise SystemExit(
            f"ERROR: video not found: {video}"
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else video.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = video.stem

    transcript_path = (
        output_dir
        / f"{stem}_captions_semantic.json"
    )

    ass_path = (
        output_dir
        / f"{stem}_captions.ass"
    )

    captioned_path = (
        output_dir
        / f"{stem}_captioned.mp4"
    )

    if (
        args.force_transcribe
        or not transcript_path.exists()
    ):

        run([
            sys.executable,
            "transcribe_groq.py",
            str(video),
            str(transcript_path),
        ])

    else:

        print(
            "Using cached transcript:",
            transcript_path,
        )

    transcript = json.load(
        open(
            transcript_path,
            encoding="utf-8",
        )
    )

    events = build_events(
        transcript
    )

    events = dedupe_events(
        events
    )

    events = repair_overlaps(
        events
    )

    validate(
        events
    )

    write_ass(
        events,
        ass_path,
    )

    print(
        "Saved ASS:",
        ass_path,
    )

    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"ass={ass_path}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(captioned_path),
    ])

    print()
    print(
        "===== CAPTION STAGE COMPLETE ====="
    )
    print(
        "Transcript:",
        transcript_path,
    )
    print(
        "ASS:",
        ass_path,
    )
    print(
        "Video:",
        captioned_path,
    )


if __name__ == "__main__":
    main()
