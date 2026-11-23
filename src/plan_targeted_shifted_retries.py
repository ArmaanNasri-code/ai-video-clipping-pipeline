#!/usr/bin/env python3

import argparse
import json
import re
from difflib import SequenceMatcher


MIN_SIM = 0.82
MAX_SIM = 0.995
MIN_OVERLAP = 0.25
JOIN_GAP = 0.35

LEFT_CONTEXT = 0.55
RIGHT_CONTEXT = 0.40
SHIFT = 0.25


def norm(s):
    return " ".join(
        re.findall(
            r"[a-z0-9']+",
            (s or "").lower(),
        )
    )


def similarity(a, b):
    return SequenceMatcher(
        None,
        norm(a),
        norm(b),
    ).ratio()


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
        "--output",
        required=True,
    )

    args = ap.parse_args()

    turbo = json.load(
        open(args.turbo_exact)
    )

    full = json.load(
        open(args.full_exact)
    )

    full_by_index = {
        int(x["retry_index"]): x
        for x in full
    }

    candidates = []

    for t in turbo:
        idx = int(
            t["retry_index"]
        )

        f = full_by_index.get(
            idx
        )

        if not f:
            continue

        window_start = float(
            t["window_start"]
        )

        u0 = float(
            t["uncovered_start"]
        )

        u1 = float(
            t["uncovered_end"]
        )

        for ts in t.get(
            "segments",
            [],
        ):
            t0 = float(
                ts["start"]
            )

            t1 = float(
                ts["end"]
            )

            for fs in f.get(
                "segments",
                [],
            ):
                f0 = (
                    float(fs["start"])
                    + window_start
                )

                f1 = (
                    float(fs["end"])
                    + window_start
                )

                ov = overlap(
                    t0,
                    t1,
                    f0,
                    f1,
                )

                if ov < MIN_OVERLAP:
                    continue

                sim = similarity(
                    ts.get(
                        "text",
                        "",
                    ),
                    fs.get(
                        "text",
                        "",
                    ),
                )

                # We specifically want:
                # highly similar but NOT exact wording.
                #
                # Exact agreement already has enough
                # evidence and does not need shifted crops.
                if not (
                    MIN_SIM
                    <= sim
                    < MAX_SIM
                ):
                    continue

                start = max(
                    t0,
                    f0,
                    u0,
                )

                end = min(
                    t1,
                    f1,
                    u1,
                )

                if end <= start:
                    continue

                candidates.append({
                    "retry_index":
                        idx,
                    "start":
                        start,
                    "end":
                        end,
                    "turbo_text":
                        ts.get(
                            "text",
                            "",
                        ).strip(),
                    "full_v3_text":
                        fs.get(
                            "text",
                            "",
                        ).strip(),
                    "similarity":
                        round(
                            sim,
                            3,
                        ),
                })

    candidates.sort(
        key=lambda x: (
            x["retry_index"],
            x["start"],
            x["end"],
        )
    )

    # --------------------------------------------
    # Merge neighboring ambiguous phrases into
    # one target subspan.
    # --------------------------------------------

    groups = []

    for item in candidates:
        if not groups:
            groups.append({
                "retry_index":
                    item["retry_index"],
                "start":
                    item["start"],
                "end":
                    item["end"],
                "evidence":
                    [item],
            })
            continue

        g = groups[-1]

        same_retry = (
            g["retry_index"]
            == item["retry_index"]
        )

        close = (
            item["start"]
            - g["end"]
            <= JOIN_GAP
        )

        if (
            same_retry
            and close
        ):
            g["end"] = max(
                g["end"],
                item["end"],
            )

            g["evidence"].append(
                item
            )

        else:
            groups.append({
                "retry_index":
                    item["retry_index"],
                "start":
                    item["start"],
                "end":
                    item["end"],
                "evidence":
                    [item],
            })

    windows = []

    for i, g in enumerate(
        groups,
        start=1,
    ):
        target0 = g["start"]
        target1 = g["end"]

        a0 = max(
            0.0,
            target0
            - LEFT_CONTEXT,
        )

        a1 = (
            target1
            + RIGHT_CONTEXT
        )

        b0 = (
            a0
            + SHIFT
        )

        b1 = (
            a1
            + SHIFT
        )

        target = (
            f"{target0:.2f}-"
            f"{target1:.2f}"
        )

        windows.extend([
            {
                "name":
                    f"target{i}_a",
                "target_index":
                    i,
                "retry_index":
                    g["retry_index"],
                "target":
                    target,
                "start":
                    round(a0, 3),
                "end":
                    round(a1, 3),
                "evidence":
                    g["evidence"],
            },
            {
                "name":
                    f"target{i}_b",
                "target_index":
                    i,
                "retry_index":
                    g["retry_index"],
                "target":
                    target,
                "start":
                    round(b0, 3),
                "end":
                    round(b1, 3),
                "evidence":
                    g["evidence"],
            },
        ])

    output = {
        "windows":
            windows,
        "count":
            len(windows),
        "target_count":
            len(groups),
        "policy": {
            "min_similarity":
                MIN_SIM,
            "max_similarity":
                MAX_SIM,
            "purpose":
                (
                    "Shift only local near-agreement "
                    "ambiguities; do not retry whole "
                    "uncovered spans."
                ),
        },
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
        "===== TARGETED SHIFTED RETRY PLAN ====="
    )

    print(
        "Near-agreement phrases:",
        len(candidates),
    )

    print(
        "Target subspans:",
        len(groups),
    )

    print(
        "Planned shifted windows:",
        len(windows),
    )

    for g_i, g in enumerate(
        groups,
        start=1,
    ):
        print()
        print(
            f"TARGET {g_i}: "
            f"{g['start']:.2f} -> "
            f"{g['end']:.2f}"
        )

        for e in g["evidence"]:
            print(
                f"  sim={e['similarity']:.3f}"
            )

            print(
                "  turbo:",
                e["turbo_text"],
            )

            print(
                "  full :",
                e["full_v3_text"],
            )

    print()
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
