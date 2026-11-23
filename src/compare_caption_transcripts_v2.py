#!/usr/bin/env python3

import argparse
import json
import re
from difflib import SequenceMatcher


def norm(text):
    return re.sub(
        r"[^a-z0-9' ]+",
        " ",
        text.lower(),
    ).strip()


def overlap(a0, a1, b0, b1):
    return max(
        0.0,
        min(a1, b1) - max(a0, b0),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--semantic",
        required=True,
    )

    ap.add_argument(
        "--timed",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    semantic = json.load(
        open(args.semantic)
    ).get("segments", [])

    timed_data = json.load(
        open(args.timed)
    )

    timed_words = timed_data.get(
        "words",
        [],
    )

    if not timed_words:
        raise SystemExit(
            "ERROR: timed transcript has no top-level words array"
        )

    WORD_PAD = 0.05

    results = []

    for s in semantic:

        s0 = float(s["start"])
        s1 = float(s["end"])

        window_start = (
            s0 - WORD_PAD
        )

        window_end = (
            s1 + WORD_PAD
        )

        matches = []

        for w in timed_words:

            w0 = float(w["start"])
            w1 = float(w["end"])

            midpoint = (
                w0 + w1
            ) / 2.0

            if (
                window_start
                <= midpoint
                <= window_end
            ):
                matches.append(w)

        matches.sort(
            key=lambda x: x["start"]
        )

        timed_text = " ".join(
            x.get("word", "").strip()
            for x in matches
        ).strip()

        semantic_text = (
            s.get("text", "").strip()
        )

        a = norm(
            semantic_text
        )

        b = norm(
            timed_text
        )

        similarity = (
            SequenceMatcher(
                None,
                a,
                b,
            ).ratio()
            if a and b
            else 0.0
        )

        results.append({
            "start": s0,
            "end": s1,
            "semantic_text": semantic_text,
            "timed_text": timed_text,
            "similarity": round(
                similarity,
                3,
            ),
            "needs_review": (
                similarity < 0.72
            ),
        })

    json.dump(
        results,
        open(
            args.output,
            "w",
        ),
        indent=2,
    )

    flagged = [
        x for x in results
        if x["needs_review"]
    ]

    print()
    print(
        "===== TRANSCRIPT CONSENSUS ====="
    )
    print(
        "Semantic regions:",
        len(results),
    )
    print(
        "Disagreements:",
        len(flagged),
    )

    for x in flagged:
        print()
        print(
            f"{x['start']:.2f}s "
            f"-> {x['end']:.2f}s"
        )
        print(
            "Similarity:",
            x["similarity"],
        )
        print(
            "A:",
            x["semantic_text"],
        )
        print(
            "B:",
            x["timed_text"],
        )

    print()
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
