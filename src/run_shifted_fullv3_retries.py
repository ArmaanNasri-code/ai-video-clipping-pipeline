#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--shifted-turbo",
        required=True,
        help="shifted_turbo_results.json",
    )

    ap.add_argument(
        "--crop-dir",
        required=True,
        help="Directory containing shifted crop mp4 files",
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--transcriber",
        default="transcribe_groq_review.py",
    )

    args = ap.parse_args()

    turbo = json.load(
        open(args.shifted_turbo)
    )

    crop_dir = Path(
        args.crop_dir
    )

    results = []

    for i, item in enumerate(
        turbo,
        start=1,
    ):
        name = item["name"]

        start = float(
            item["start"]
        )

        crop = (
            crop_dir
            / f"{i:02d}_{name}.mp4"
        )

        if not crop.exists():
            print(
                "SKIP: missing crop",
                crop,
            )
            continue

        raw_out = (
            crop_dir
            / f"{name}_fullv3.json"
        )

        print()
        print(
            "=" * 50
        )

        print(
            "FULL-V3:",
            name,
        )

        subprocess.run(
            [
                "python3",
                args.transcriber,
                str(crop),
                str(raw_out),
            ],
            check=True,
        )

        data = json.load(
            open(raw_out)
        )

        absolute = []

        for s in data.get(
            "segments",
            [],
        ):
            absolute.append({
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
            for s in absolute
            if s["text"]
        ).strip()

        results.append({
            "name":
                name,
            "target":
                item["target"],
            "start":
                item["start"],
            "end":
                item["end"],
            "turbo_text":
                item.get(
                    "text",
                    "",
                ),
            "full_v3_text":
                text,
            "full_v3_segments":
                absolute,
        })

        print(
            "TURBO:",
            item.get(
                "text",
                "",
            )
            or "(none)",
        )

        print(
            "FULL :",
            text
            or "(none)",
        )

        print(
            "FULL ABSOLUTE:"
        )

        for s in absolute:
            print(
                f"  {s['start']:.2f} -> "
                f"{s['end']:.2f} | "
                f"{s['text']}"
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
        "===== SHIFTED FULL-V3 COMPLETE ====="
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
