#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
    )

    ap.add_argument(
        "--coverage",
        required=True,
        help="Coverage JSON containing retry_windows",
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--transcriber",
        default="transcribe_groq.py",
    )

    args = ap.parse_args()

    coverage = json.load(
        open(args.coverage)
    )

    windows = coverage.get(
        "retry_windows",
        [],
    )

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for i, window in enumerate(
        windows,
        start=1,
    ):
        start = float(
            window["start"]
        )

        end = float(
            window["end"]
        )

        uncovered_start = float(
            window["uncovered_start"]
        )

        uncovered_end = float(
            window["uncovered_end"]
        )

        duration = max(
            0.01,
            end - start,
        )

        crop = (
            outdir
            / f"retry_{i:02d}.mp4"
        )

        raw_json = (
            outdir
            / f"retry_{i:02d}.json"
        )

        print()
        print(
            f"===== EXACT TURBO RETRY {i} ====="
        )

        print(
            f"window: {start:.2f} -> "
            f"{end:.2f}"
        )

        print(
            f"missing: {uncovered_start:.2f} -> "
            f"{uncovered_end:.2f}"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                args.video,
                "-c",
                "copy",
                str(crop),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                "python3",
                args.transcriber,
                str(crop),
                str(raw_json),
            ],
            check=True,
        )

        data = json.load(
            open(raw_json)
        )

        absolute_segments = []

        for s in data.get(
            "segments",
            [],
        ):
            absolute_segments.append({
                "start":
                    float(s["start"])
                    + start,
                "end":
                    float(s["end"])
                    + start,
                "text":
                    s.get(
                        "text",
                        "",
                    ).strip(),
            })

        text = " ".join(
            s["text"]
            for s in absolute_segments
            if s["text"]
        ).strip()

        results.append({
            "retry_index":
                i,
            "window_start":
                start,
            "window_end":
                end,
            "uncovered_start":
                uncovered_start,
            "uncovered_end":
                uncovered_end,
            "text":
                text,
            "segments":
                absolute_segments,
        })

        print(
            "TEXT:",
            text or "(none)",
        )

    output = (
        outdir
        / "retry_results_absolute.json"
    )

    json.dump(
        results,
        open(
            output,
            "w",
        ),
        indent=2,
    )

    print()
    print(
        "===== EXACT TURBO COMPLETE ====="
    )

    print(
        "Runs:",
        len(results),
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
