#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher


MIN_EXACT_SIM = 0.95
MIN_INSIDE_FRACTION = 0.60
MIN_ACTUAL_OVERLAP = 0.30


def norm(s):
    return " ".join(
        re.findall(
            r"[a-z0-9']+",
            (s or "").lower(),
        )
    )


def overlap(a0, a1, b0, b1):
    return max(
        0.0,
        min(a1, b1)
        - max(a0, b0),
    )


def similarity(a, b):
    return SequenceMatcher(
        None,
        norm(a),
        norm(b),
    ).ratio()


def family_name(name):
    return re.sub(
        r"_[ab]$",
        "",
        name,
    )


def target_bounds(target):
    a, b = target.split("-", 1)
    return float(a), float(b)


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

    turbo_exact = json.load(
        open(args.turbo_exact)
    )

    full_exact = json.load(
        open(args.full_exact)
    )

    shifted_turbo = json.load(
        open(args.shifted_turbo)
    )

    shifted_full = json.load(
        open(args.shifted_full)
    )

    turbo_by_index = {
        int(x["retry_index"]): x
        for x in turbo_exact
    }

    full_by_index = {
        int(x["retry_index"]): x
        for x in full_exact
    }

    regions = []

    for retry in turbo_exact:
        regions.append({
            "uncovered_start":
                float(retry["uncovered_start"]),
            "uncovered_end":
                float(retry["uncovered_end"]),
            "stable_support": [],
        })

    # --------------------------------------------------
    # Tier 1:
    # Exact local turbo/full-v3 agreement.
    # --------------------------------------------------

    for retry_index, turbo in turbo_by_index.items():
        full = full_by_index.get(
            retry_index
        )

        if not full:
            continue

        u0 = float(
            turbo["uncovered_start"]
        )

        u1 = float(
            turbo["uncovered_end"]
        )

        window_start = float(
            turbo["window_start"]
        )

        region = next(
            x for x in regions
            if abs(
                x["uncovered_start"] - u0
            ) < 0.001
            and abs(
                x["uncovered_end"] - u1
            ) < 0.001
        )

        for t in turbo.get(
            "segments",
            [],
        ):
            t0 = float(t["start"])
            t1 = float(t["end"])

            t_dur = max(
                0.001,
                t1 - t0,
            )

            for f in full.get(
                "segments",
                [],
            ):
                # Full-v3 exact timestamps are local
                # to the retry crop.
                f0 = (
                    float(f["start"])
                    + window_start
                )

                f1 = (
                    float(f["end"])
                    + window_start
                )

                f_dur = max(
                    0.001,
                    f1 - f0,
                )

                pair_overlap = overlap(
                    t0,
                    t1,
                    f0,
                    f1,
                )

                if pair_overlap <= 0:
                    continue

                actual_start = max(
                    t0,
                    f0,
                    u0,
                )

                actual_end = min(
                    t1,
                    f1,
                    u1,
                )

                actual_overlap = max(
                    0.0,
                    actual_end - actual_start,
                )

                t_inside = overlap(
                    t0,
                    t1,
                    u0,
                    u1,
                )

                f_inside = overlap(
                    f0,
                    f1,
                    u0,
                    u1,
                )

                t_fraction = (
                    t_inside / t_dur
                )

                f_fraction = (
                    f_inside / f_dur
                )

                sim = similarity(
                    t.get("text", ""),
                    f.get("text", ""),
                )

                valid = (
                    actual_overlap
                    >= MIN_ACTUAL_OVERLAP
                    and
                    t_fraction
                    >= MIN_INSIDE_FRACTION
                    and
                    f_fraction
                    >= MIN_INSIDE_FRACTION
                    and
                    sim
                    >= MIN_EXACT_SIM
                )

                if not valid:
                    continue

                region[
                    "stable_support"
                ].append({
                    "start":
                        max(t0, f0),
                    "end":
                        min(t1, f1),
                    "text":
                        t["text"].strip(),
                    "evidence_type":
                        "exact_local_dual_whisper",
                    "confidence":
                        "strong_correlated",
                    "auto_usable":
                        False,
                    "similarity":
                        round(sim, 3),
                    "retry_index":
                        retry_index,
                })

    # --------------------------------------------------
    # Tier 2:
    # Repeated shifted-crop stability.
    #
    # Requirements:
    # - same family has >=2 crop shifts
    # - turbo == full-v3 within each shift
    # - whole crop text is stable across shifts
    # - corresponding full-v3 segments agree
    # --------------------------------------------------

    shifted_turbo_by_name = {
        x["name"]: x
        for x in shifted_turbo
    }

    shifted_full_by_name = {
        x["name"]: x
        for x in shifted_full
    }

    families = defaultdict(list)

    for name in (
        set(shifted_turbo_by_name)
        & set(shifted_full_by_name)
    ):
        families[
            family_name(name)
        ].append(name)

    for family, names in families.items():
        names = sorted(names)

        if len(names) < 2:
            continue

        valid_runs = []

        for name in names:
            turbo = shifted_turbo_by_name[
                name
            ]

            full = shifted_full_by_name[
                name
            ]

            turbo_text = norm(
                turbo.get(
                    "text",
                    "",
                )
            )

            full_text = norm(
                full.get(
                    "full_v3_text",
                    "",
                )
            )

            if (
                not turbo_text
                or turbo_text != full_text
            ):
                continue

            valid_runs.append(
                (
                    name,
                    turbo,
                    full,
                )
            )

        if len(valid_runs) < 2:
            continue

        crop_texts = {
            norm(
                x[1].get(
                    "text",
                    "",
                )
            )
            for x in valid_runs
        }

        if len(crop_texts) != 1:
            continue

        # Compare full-v3 segment sequence across
        # repeated shifted crops.
        segment_sequences = []

        for _, _, full in valid_runs:
            seq = [
                (
                    norm(s.get("text", "")),
                    s,
                )
                for s in full.get(
                    "full_v3_segments",
                    [],
                )
                if norm(
                    s.get(
                        "text",
                        "",
                    )
                )
            ]

            segment_sequences.append(
                seq
            )

        min_len = min(
            len(x)
            for x in segment_sequences
        )

        for i in range(min_len):
            normalized_phrases = [
                seq[i][0]
                for seq in segment_sequences
            ]

            if len(
                set(normalized_phrases)
            ) != 1:
                continue

            phrase = (
                segment_sequences[0][i][1]
                ["text"]
                .strip()
            )

            starts = [
                float(seq[i][1]["start"])
                for seq in segment_sequences
            ]

            ends = [
                float(seq[i][1]["end"])
                for seq in segment_sequences
            ]

            # Conservative time window shared
            # by the repeated shifted runs.
            stable_start = max(starts)
            stable_end = min(ends)

            if stable_end <= stable_start:
                continue

            target0, target1 = (
                target_bounds(
                    valid_runs[0][1][
                        "target"
                    ]
                )
            )

            if overlap(
                stable_start,
                stable_end,
                target0,
                target1,
            ) <= 0:
                continue

            # Attach to the uncovered region that
            # contains/overlaps this stable phrase.
            matching_regions = [
                r
                for r in regions
                if overlap(
                    stable_start,
                    stable_end,
                    r["uncovered_start"],
                    r["uncovered_end"],
                ) > 0
            ]

            if not matching_regions:
                continue

            for region in matching_regions:
                region[
                    "stable_support"
                ].append({
                    "start":
                        stable_start,
                    "end":
                        stable_end,
                    "text":
                        phrase,
                    "evidence_type":
                        "repeated_shifted_whisper_stability",
                    "confidence":
                        "strong_correlated",
                    "auto_usable":
                        False,
                    "shift_family":
                        family,
                    "runs":
                        [
                            x[0]
                            for x in valid_runs
                        ],
                })

    # --------------------------------------------------
    # Deduplicate stable phrases inside each region.
    # Prefer shifted stability when duplicate wording
    # appears in both evidence tiers.
    # --------------------------------------------------

    for region in regions:
        items = region[
            "stable_support"
        ]

        by_text = {}

        for item in items:
            key = norm(
                item["text"]
            )

            if not key:
                continue

            existing = by_text.get(
                key
            )

            if existing is None:
                by_text[key] = item
                continue

            if (
                item["evidence_type"]
                ==
                "repeated_shifted_whisper_stability"
            ):
                by_text[key] = item

        region[
            "stable_support"
        ] = sorted(
            by_text.values(),
            key=lambda x: (
                float(x["start"]),
                float(x["end"]),
            ),
        )

    output = {
        "policy": {
            "auto_usable":
                False,
            "note":
                (
                    "Stable support is still "
                    "correlated Whisper-family "
                    "evidence. Unstable spans "
                    "remain unresolved."
                ),
        },
        "regions":
            regions,
    }

    json.dump(
        output,
        open(
            args.output,
            "w",
        ),
        indent=2,
    )

    print(
        "===== STABLE RECOVERY EVIDENCE ====="
    )

    total = 0

    for region in regions:
        stable = region[
            "stable_support"
        ]

        total += len(stable)

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
