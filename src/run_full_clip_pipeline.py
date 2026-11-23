#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable


def run(cmd):
    print()
    print("=" * 72)
    print("$", " ".join(str(x) for x in cmd))
    print("=" * 72)

    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def require(path, label):
    path = Path(path)

    if not path.exists():
        raise SystemExit(
            f"ERROR: {label} not found: {path}"
        )

    return path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run complete autonomous caption + "
            "finishing pipeline for one vertical clip."
        )
    )

    ap.add_argument(
        "--video",
        required=True,
        help="Finished vertical clip before captions",
    )

    ap.add_argument(
        "--context",
        default="",
        help="Short factual context for hook generation",
    )

    ap.add_argument(
        "--clip-id",
        required=True,
    )

    ap.add_argument(
        "--vod",
        required=True,
    )

    ap.add_argument(
        "--timestamp",
        required=True,
    )

    ap.add_argument(
        "--score",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--production-dir",
        default=None,
        help="where the READY_TO_POST clip goes (passed to finish_clip.py)",
    )

    args = ap.parse_args()

    video = require(
        args.video,
        "input video",
    )

    require(
        "run_full_caption_pipeline.py",
        "caption pipeline",
    )

    require(
        "finish_clip.py",
        "finish pipeline",
    )

    root = Path(
        args.output_dir
    )

    caption_dir = (
        root / "caption_pipeline"
    )

    finish_dir = (
        root / "finish"
    )

    review_dir = (
        root / "needs_caption_review"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    recovered = (
        caption_dir
        / "recovery"
        / "captions_support_only.json"
    )

    print()
    print("=" * 72)
    print("FULL CLIP PIPELINE START")
    print("=" * 72)
    print("Video:", video)
    print("Clip ID:", args.clip_id)
    print("Output:", root)

    # --------------------------------------------------
    # 1. CAPTION / RESOLUTION / RECOVERY PIPELINE
    # --------------------------------------------------

    run([
        PYTHON,
        "run_full_caption_pipeline.py",
        "--video",
        video,
        "--output-dir",
        caption_dir,
    ])

    require(
        recovered,
        "recovery-safe resolved captions",
    )

    # --------------------------------------------------
    # 2. FROZEN FINISHING PIPELINE
    # --------------------------------------------------

    finish_cmd = [
        PYTHON,
        "finish_clip.py",
        "--video",
        video,
        "--resolved",
        recovered,
        "--clip-id",
        args.clip_id,
        "--vod",
        args.vod,
        "--timestamp",
        args.timestamp,
        "--score",
        args.score,
        "--work-dir",
        finish_dir,
        "--review-dir",
        review_dir,
    ]

    if args.production_dir:
        finish_cmd.extend([
            "--production-dir",
            args.production_dir,
        ])

    if args.context.strip():
        finish_cmd.extend([
            "--context",
            args.context.strip(),
        ])

    run(
        finish_cmd
    )

    print()
    print("=" * 72)
    print("FULL CLIP PIPELINE COMPLETE")
    print("=" * 72)
    print(
        "Recovery-safe captions:",
        recovered,
    )
    print(
        "Finish workspace:",
        finish_dir,
    )
    print(
        "Review workspace:",
        review_dir,
    )


if __name__ == "__main__":
    main()
