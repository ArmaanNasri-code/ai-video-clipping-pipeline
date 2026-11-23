#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--turbo-exact",
        required=True,
        help="retry_results_absolute.json",
    )

    ap.add_argument(
        "--retry-dir",
        required=True,
        help="Directory containing retry_XX.mp4 files",
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional retry indices to run. "
            "If omitted, run all indices."
        ),
    )

    ap.add_argument(
        "--transcriber",
        default="transcribe_groq_review.py",
        help="Full-v3 transcriber script",
    )

    args = ap.parse_args()

    turbo = json.load(
        open(args.turbo_exact)
    )

    retry_dir = Path(
        args.retry_dir
    )

    wanted = (
        set(args.indices)
        if args.indices
        else None
    )

    results = []

    for item in turbo:
        i = int(
            item["retry_index"]
        )

        if (
            wanted is not None
            and i not in wanted
        ):
            continue

        video = (
            retry_dir
            / f"retry_{i:02d}.mp4"
        )

        if not video.exists():
            print(
                "SKIP: missing video",
                video,
            )
            continue

        raw_out = (
            retry_dir
            / f"retry_{i:02d}_fullv3.json"
        )

        print()
        print(
            f"===== FULL-V3 EXACT RETRY {i} ====="
        )

        subprocess.run(
            [
                "python3",
                args.transcriber,
                str(video),
                str(raw_out),
            ],
            check=True,
        )

        data = json.load(
            open(raw_out)
        )

        segments = (
            data.get(
                "segments",
                [],
            )
        )

        text = " ".join(
            s.get(
                "text",
                "",
            ).strip()
            for s in segments
            if s.get(
                "text",
                "",
            ).strip()
        ).strip()

        results.append({
            "retry_index":
                i,
            "text":
                text,
            "segments":
                segments,
        })

        print(
            "TEXT:",
            text or "(none)",
        )

    results.sort(
        key=lambda x:
            x["retry_index"]
    )

    json.dump(
        results,
        open(
            args.output,
            "w",
        ),
        indent=2,
    )

    print()
    print(
        "===== FULL-V3 EXACT COMPLETE ====="
    )

    print(
        "Runs:",
        len(results),
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
