#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


PLAYRES_X = 608
PLAYRES_Y = 1080

TEXT = "@chuzzzclips"
FONT = "Arial"
FONT_SIZE = 24

MARGIN_R = 20
MARGIN_V = 28


def run(cmd):
    print()
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def get_duration(video):
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(r.stdout.strip())


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


def write_ass(duration, path):

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAYRES_X}
PlayResY: {PLAYRES_Y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Watermark,{FONT},{FONT_SIZE},&H4DFFFFFF,&H4DFFFFFF,&H66000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,9,20,{MARGIN_R},{MARGIN_V},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,{ass_time(0)},{ass_time(duration)},Watermark,,0,0,0,,{escape_ass(TEXT)}
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
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    video = Path(
        args.video
    )

    if not video.exists():
        raise SystemExit(
            f"ERROR: video not found: {video}"
        )

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    duration = get_duration(
        video
    )

    stem = video.stem

    ass_path = (
        outdir
        / f"{stem}_watermark.ass"
    )

    output_video = (
        outdir
        / f"{stem}_watermarked.mp4"
    )

    write_ass(
        duration,
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
        "===== WATERMARK RENDER COMPLETE ====="
    )

    print(
        "Watermark:",
        TEXT,
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
