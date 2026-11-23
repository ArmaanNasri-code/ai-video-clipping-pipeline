#!/usr/bin/env python3

import argparse
import json
from copy import deepcopy


def overlap(a0, a1, b0, b1):
    return max(
        0.0,
        min(a1, b1)
        - max(a0, b0),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base",
        required=True,
    )

    ap.add_argument(
        "--evidence",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    base = json.load(
        open(args.base)
    )

    evidence = json.load(
        open(args.evidence)
    )

    out = deepcopy(base)

    annotated = []

    for region in out:
        r0 = float(
            region["start"]
        )

        r1 = float(
            region["end"]
        )

        stable = []

        for recovery_region in evidence.get(
            "regions",
            [],
        ):
            for item in recovery_region.get(
                "stable_support",
                [],
            ):
                # Production rule:
                # correlated recovery must remain
                # support-only.
                if item.get(
                    "auto_usable",
                    False,
                ):
                    raise RuntimeError(
                        "Unexpected auto_usable=True "
                        "in stable recovery evidence."
                    )

                i0 = float(
                    item["start"]
                )

                i1 = float(
                    item["end"]
                )

                if overlap(
                    r0,
                    r1,
                    i0,
                    i1,
                ) <= 0:
                    continue

                stable.append(
                    item
                )

        if not stable:
            continue

        stable.sort(
            key=lambda x: (
                float(x["start"]),
                float(x["end"]),
            )
        )

        # CRITICAL:
        # Do NOT alter resolved_text.
        # Do NOT alter confidence.
        # Do NOT alter status.
        region[
            "stable_recovery"
        ] = stable

        region[
            "stable_recovery_policy"
        ] = "support_only"

        annotated.append({
            "start": r0,
            "end": r1,
            "text":
                region.get(
                    "resolved_text",
                    "",
                ),
            "status":
                region.get(
                    "status"
                ),
            "confidence":
                region.get(
                    "confidence"
                ),
            "stable_items":
                len(stable),
        })

    json.dump(
        out,
        open(
            args.output,
            "w",
        ),
        indent=2,
    )

    print(
        "===== STABLE RECOVERY SUPPORT ====="
    )

    print(
        "Regions:",
        len(out),
    )

    print(
        "Annotated:",
        len(annotated),
    )

    for item in annotated:
        print()
        print(
            f"{item['start']:.2f} -> "
            f"{item['end']:.2f}"
        )

        print(
            "TEXT UNCHANGED:",
            item["text"],
        )

        print(
            "STATUS:",
            item["status"],
            item["confidence"],
        )

        print(
            "STABLE ITEMS:",
            item["stable_items"],
        )

    print()
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
