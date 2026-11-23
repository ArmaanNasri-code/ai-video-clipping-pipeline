#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


PAD = 0.35


def run(cmd):
    print()
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


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

    ap.add_argument(
        "--transcriber",
        choices=[
            "turbo",
            "full",
        ],
        default="full",
    )

    args = ap.parse_args()

    transcriber_script = (
        "transcribe_groq.py"
        if args.transcriber == "turbo"
        else "transcribe_groq_review.py"
    )

    video = Path(args.video)
    outdir = Path(args.output_dir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = json.load(
        open(args.resolved)
    )

    flagged = [
        x
        for x in data
        if x.get("status") == "review_required"
    ]

    print()
    print("===== THIRD PASS V2 =====")
    print(
        "Review-required regions:",
        len(flagged),
    )

    results = []

    for i, region in enumerate(
        flagged,
        1,
    ):
        target_start = float(
            region["start"]
        )

        target_end = float(
            region["end"]
        )

        crop_start = max(
            0.0,
            target_start - PAD,
        )

        crop_end = (
            target_end + PAD
        )

        duration = (
            crop_end - crop_start
        )

        stem = (
            f"review_{i:02d}_"
            f"{target_start:.2f}_"
            f"{target_end:.2f}"
        )

        audio_path = (
            outdir
            / f"{stem}.m4a"
        )

        transcript_path = (
            outdir
            / f"{stem}.json"
        )

        run([
            "ffmpeg",
            "-y",
            "-ss",
            f"{crop_start:.3f}",
            "-i",
            video,
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            audio_path,
        ])

        run([
            sys.executable,
            transcriber_script,
            audio_path,
            transcript_path,
        ])

        transcript = json.load(
            open(transcript_path)
        )

        text = " ".join(
            seg.get(
                "text",
                "",
            ).strip()
            for seg in transcript.get(
                "segments",
                [],
            )
        ).strip()

        results.append({
            "window_start":
                round(target_start, 3),

            "window_end":
                round(target_end, 3),

            "crop_start":
                round(crop_start, 3),

            "crop_end":
                round(crop_end, 3),

            "third_pass_text":
                text,
        })

        print()
        print("=" * 60)

        print(
            f"REGION {i}: "
            f"{target_start:.2f}s -> "
            f"{target_end:.2f}s"
        )

        print(
            f"Audio crop: "
            f"{crop_start:.2f}s -> "
            f"{crop_end:.2f}s"
        )

        print(text)

    final_path = (
        outdir
        / "third_pass_results.json"
    )

    json.dump(
        results,
        open(
            final_path,
            "w",
        ),
        indent=2,
    )

    print()
    print(
        "===== THIRD PASS V2 COMPLETE ====="
    )

    print(
        "Regions:",
        len(results),
    )

    print(
        "Saved:",
        final_path,
    )


if __name__ == "__main__":
    main()
