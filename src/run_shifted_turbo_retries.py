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
        "--plan",
        required=True,
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

    plan = json.load(
        open(args.plan)
    )

    windows = plan.get(
        "windows",
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

    for i, w in enumerate(
        windows,
        start=1,
    ):
        name = w["name"]

        start = float(
            w["start"]
        )

        end = float(
            w["end"]
        )

        duration = max(
            0.01,
            end - start,
        )

        crop = (
            outdir
            / f"{i:02d}_{name}.mp4"
        )

        raw_json = (
            outdir
            / f"{i:02d}_{name}.json"
        )

        print()
        print(
            "=" * 50
        )

        print(
            f"{i}. {name}"
        )

        print(
            "TARGET:",
            w["target"],
        )

        print(
            f"CROP: {start:.2f} -> {end:.2f}"
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

        result = {
            "name":
                name,
            "target":
                w["target"],
            "start":
                start,
            "end":
                end,
            "text":
                text,
            "segments":
                absolute_segments,
        }

        results.append(
            result
        )

        print(
            "TEXT:",
            text or "(none)",
        )

        print(
            "ABSOLUTE SEGMENTS:"
        )

        for s in absolute_segments:
            print(
                f"  {s['start']:.2f} -> "
                f"{s['end']:.2f} | "
                f"{s['text']}"
            )

    summary = (
        outdir
        / "shifted_turbo_results.json"
    )

    json.dump(
        results,
        open(
            summary,
            "w",
        ),
        indent=2,
    )

    print()
    print(
        "===== SHIFTED TURBO COMPLETE ====="
    )

    print(
        "Runs:",
        len(results),
    )

    print(
        "Saved:",
        summary,
    )


if __name__ == "__main__":
    main()
