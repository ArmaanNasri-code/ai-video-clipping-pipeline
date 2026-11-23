#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
from pathlib import Path


def norm(s):
    return " ".join(
        re.findall(
            r"[a-z0-9']+",
            (s or "").lower(),
        )
    )


def family(name):
    return re.sub(
        r"_[ab]$",
        "",
        name or "",
    )


def parse_target(s):
    a, b = s.split("-", 1)

    return (
        float(a),
        float(b),
    )


def overlap(a0, a1, b0, b1):
    return max(
        0.0,
        min(a1, b1)
        - max(a0, b0),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--turbo-exact",
        required=True,
    )

    ap.add_argument(
        "--full-exact",
        required=True,
    )

    ap.add_argument(
        "--shifted-turbo",
        required=True,
    )

    ap.add_argument(
        "--shifted-full",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    # --------------------------------------------------
    # First run the existing conservative builder.
    # --------------------------------------------------

    subprocess.run(
        [
            "python3",
            "build_stable_recovery_evidence.py",
            "--turbo-exact",
            args.turbo_exact,
            "--full-exact",
            args.full_exact,
            "--shifted-turbo",
            args.shifted_turbo,
            "--shifted-full",
            args.shifted_full,
            "--output",
            args.output,
        ],
        check=True,
    )

    evidence = json.load(
        open(args.output)
    )

    turbo = json.load(
        open(args.shifted_turbo)
    )

    full = json.load(
        open(args.shifted_full)
    )

    full_by_name = {
        x["name"]: x
        for x in full
    }

    groups = {}

    for t in turbo:
        name = t.get(
            "name",
            "",
        )

        f = full_by_name.get(
            name
        )

        if not f:
            continue

        turbo_text = (
            t.get(
                "text",
                "",
            )
            or t.get(
                "turbo_text",
                "",
            )
        ).strip()

        full_text = (
            f.get(
                "full_v3_text",
                "",
            )
        ).strip()

        nt = norm(
            turbo_text
        )

        nf = norm(
            full_text
        )

        # Both Whisper variants must agree
        # inside this specific shifted crop.
        if (
            not nt
            or nt != nf
        ):
            continue

        target = (
            t.get(
                "target"
            )
            or f.get(
                "target"
            )
        )

        if not target:
            continue

        key = (
            family(name),
            target,
        )

        groups.setdefault(
            key,
            [],
        ).append({
            "name":
                name,
            "text":
                turbo_text,
            "normalized":
                nt,
            "target":
                target,
        })

    additions = []

    for (
        fam,
        target,
    ), runs in groups.items():

        # Need repeated shifted crops.
        if len(runs) < 2:
            continue

        normalized = {
            x["normalized"]
            for x in runs
        }

        # All shifted crops must agree on
        # the entire normalized wording.
        if len(normalized) != 1:
            continue

        t0, t1 = parse_target(
            target
        )

        # Do not duplicate a stable item
        # already covering the same target
        # with the same normalized text.
        duplicate = False

        for region in evidence.get(
            "regions",
            [],
        ):
            for item in region.get(
                "stable_support",
                [],
            ):
                if (
                    norm(
                        item.get(
                            "text",
                            "",
                        )
                    )
                    == runs[0]["normalized"]
                    and overlap(
                        float(
                            item["start"]
                        ),
                        float(
                            item["end"]
                        ),
                        t0,
                        t1,
                    )
                    >= (
                        0.8
                        * (t1 - t0)
                    )
                ):
                    duplicate = True

        if duplicate:
            continue

        # Attach only to the recovery region
        # containing this exact target.
        for region in evidence.get(
            "regions",
            [],
        ):
            r0 = float(
                region[
                    "uncovered_start"
                ]
            )

            r1 = float(
                region[
                    "uncovered_end"
                ]
            )

            if overlap(
                r0,
                r1,
                t0,
                t1,
            ) <= 0:
                continue

            item = {
                "start":
                    t0,
                "end":
                    t1,
                "text":
                    runs[0][
                        "text"
                    ],
                "evidence_type":
                    (
                        "repeated_shifted_"
                        "whisper_stability_"
                        "whole_target"
                    ),
                "confidence":
                    "strong_correlated",
                "auto_usable":
                    False,
                "shift_count":
                    len(runs),
                "shift_names":
                    [
                        x["name"]
                        for x in runs
                    ],
            }

            region.setdefault(
                "stable_support",
                [],
            ).append(
                item
            )

            region[
                "stable_support"
            ].sort(
                key=lambda x: (
                    float(
                        x["start"]
                    ),
                    float(
                        x["end"]
                    ),
                )
            )

            additions.append(
                item
            )

    evidence.setdefault(
        "policy",
        {},
    )

    evidence[
        "policy"
    ][
        "whole_target_shift_stability"
    ] = {
        "requires_at_least_two_shifts":
            True,
        "requires_turbo_full_agreement":
            True,
        "requires_cross_shift_agreement":
            True,
        "confidence":
            "strong_correlated",
        "auto_usable":
            False,
    }

    json.dump(
        evidence,
        open(
            args.output,
            "w",
        ),
        indent=2,
    )

    print()
    print(
        "===== WHOLE-TARGET STABILITY ADDITIONS ====="
    )

    print(
        "Added:",
        len(additions),
    )

    for x in additions:
        print(
            f"{x['start']:.2f} -> "
            f"{x['end']:.2f} | "
            f"{x['text']} | "
            f"{x['evidence_type']}"
        )

    print()
    print(
        "===== FINAL STABLE RECOVERY EVIDENCE ====="
    )

    total = 0

    for region in evidence.get(
        "regions",
        [],
    ):
        stable = region.get(
            "stable_support",
            [],
        )

        total += len(
            stable
        )

        print()
        print(
            f"{region['uncovered_start']:.2f} "
            f"-> "
            f"{region['uncovered_end']:.2f}"
        )

        print(
            "STABLE:",
            len(stable),
        )

        for item in stable:
            print(
                f"  {item['start']:.2f} -> "
                f"{item['end']:.2f} | "
                f"{item['text']} | "
                f"{item['evidence_type']}"
            )

    print()
    print(
        "TOTAL STABLE PHRASES:",
        total,
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
