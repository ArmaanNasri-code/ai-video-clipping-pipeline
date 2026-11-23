#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


PLAYRES_X = 608
PLAYRES_Y = 1080

FONT = "Arial"
FONT_SIZE = 42

MARGIN_L = 35
MARGIN_R = 35
MARGIN_V = 90

HOOK_DURATION = 7.0


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


def split_hook(text):
    words = text.strip().split()

    if len(words) <= 4:
        return text

    # Prefer balanced 2-line layout.
    midpoint = len(words) // 2

    left = words[:midpoint]
    right = words[midpoint:]

    # For odd counts, allow first line to be longer
    # if that reads more naturally.
    if len(words) == 5:
        left = words[:3]
        right = words[3:]

    return (
        " ".join(left)
        + r"\N"
        + " ".join(right)
    )


def write_ass(hook, path):

    display = split_hook(
        hook.upper()
    )

    display = escape_ass(
        display
    )

    # Restore ASS newline after escaping.
    display = display.replace(
        r"\\N",
        r"\N",
    )

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAYRES_X}
PlayResY: {PLAYRES_Y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Hook,{FONT},{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,1,8,{MARGIN_L},{MARGIN_R},{MARGIN_V},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,{ass_time(0)},{ass_time(HOOK_DURATION)},Hook,,0,0,0,,{display}
"""

    path.write_text(
        header,
        encoding="utf-8",
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
    )

    ap.add_argument(
        "--hook-json",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    video = Path(
        args.video
    )

    hook_path = Path(
        args.hook_json
    )

    outdir = Path(
        args.output_dir
    )

    if not video.exists():
        raise SystemExit(
            f"ERROR: video not found: {video}"
        )

    if not hook_path.exists():
        raise SystemExit(
            f"ERROR: hook JSON not found: {hook_path}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hook_data = json.load(
        open(
            hook_path,
            encoding="utf-8",
        )
    )

    hook = (
        hook_data.get(
            "hook",
            "",
        )
        .strip()
    )

    if not hook:
        raise SystemExit(
            "ERROR: hook JSON contains no hook"
        )

    stem = video.stem

    ass_path = (
        outdir
        / f"{stem}_hook.ass"
    )

    output_video = (
        outdir
        / f"{stem}_hooked.mp4"
    )

    write_ass(
        hook,
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
        "libx264", "-pix_fmt", "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_video),
    ])

    print()
    print(
        "===== HOOK RENDER COMPLETE ====="
    )

    print(
        "Hook:",
        hook,
    )

    print(
        "ASS:",
        ass_path,
    )

    print(
        "Video:",
        output_video,
    )


if __name__ == "__main__":
    main()
