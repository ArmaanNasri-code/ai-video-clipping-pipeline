#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path


BLANK_MAX_SECONDS = 1.5   # a low-confidence line this short may be left blank
BLANK_MAX_SHARE = 0.10    # ...if all such lines together are at most this share of the captions


def apply_caption_blanking(regions):
    """Drop short low-confidence lines (shown as no caption). Returns (kept, blanked)."""
    if os.environ.get("CAPTION_BLANK_UNCERTAIN", "1").strip() == "0":
        return regions, []

    def _d(x):
        try:
            return max(0.0, float(x["end"]) - float(x["start"]))
        except Exception:
            return 0.0
    total = sum(_d(x) for x in regions)
    cand = [x for x in regions if x.get("confidence") == "low" and 0 < _d(x) <= BLANK_MAX_SECONDS]
    if not cand or total <= 0 or sum(_d(x) for x in cand) > BLANK_MAX_SHARE * total:
        return regions, []
    ids = {id(x) for x in cand}
    return [x for x in regions if id(x) not in ids], cand


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--resolved",
        required=True,
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    ap.add_argument(
        "--max-medium-frac",
        type=float,
        default=0.25,
        help="auto-approve a clip with no low-confidence captions if its "
             "medium-confidence review regions cover at most this share of "
             "the captioned time (default 0.25). 0 = old strict behaviour.",
    )

    args = ap.parse_args()

    resolved_path = Path(
        args.resolved
    )

    if not resolved_path.exists():
        raise SystemExit(
            f"ERROR: not found: {resolved_path}"
        )

    data = json.load(
        open(
            resolved_path,
            encoding="utf-8",
        )
    )

    data, blanked = apply_caption_blanking(data)

    review = [
        x
        for x in data
        if x.get("status")
        == "review_required"
    ]

    low = [
        x
        for x in data
        if x.get("confidence")
        == "low"
    ]

    def dur(x):
        try:
            return max(0.0, float(x["end"]) - float(x["start"]))
        except Exception:
            return 0.0

    spoken_s = sum(dur(x) for x in data)
    review_s = sum(dur(x) for x in review)
    review_frac = review_s / spoken_s if spoken_s > 0 else 0.0
    # anything flagged for review that is not "medium" (low, missing, odd) blocks
    blocking = [
        x
        for x in review
        if x.get("confidence") != "medium"
    ]

    approved = (
        len(low) == 0
        and len(blocking) == 0
        and review_frac <= args.max_medium_frac
    )
    allowed_medium = approved and len(review) > 0

    status = (
        "CAPTIONS_APPROVED"
        if approved
        else "NEEDS_CAPTION_REVIEW"
    )

    result = {
        "status": status,
        "ready_to_post": approved,
        "total_regions": len(data),
        "review_required": len(review),
        "low_confidence": len(low),
        "captioned_seconds": round(spoken_s, 2),
        "review_seconds": round(review_s, 2),
        "review_fraction": round(review_frac, 4),
        "max_medium_frac": args.max_medium_frac,
        "approved_with_medium_regions": allowed_medium,
        "blanked_regions": [
            {"start": x["start"], "end": x["end"], "text": x.get("resolved_text", "")}
            for x in blanked
        ],
        "review_regions": [
            {
                "start": x["start"],
                "end": x["end"],
                "confidence":
                    x.get("confidence"),
                "text":
                    x.get(
                        "resolved_text",
                        "",
                    ),
            }
            for x in review
        ],
    }

    if args.output:

        output = Path(
            args.output
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        json.dump(
            result,
            open(
                output,
                "w",
                encoding="utf-8",
            ),
            indent=2,
        )

    print()
    print(
        "===== CAPTION QUALITY GATE ====="
    )

    print(
        "Status:",
        status,
    )

    print(
        "Total regions:",
        len(data),
    )

    print(
        "Review required:",
        len(review),
    )

    print(
        "Low confidence:",
        len(low),
    )

    print(
        f"Medium review time: {review_s:.1f}s of {spoken_s:.1f}s captioned "
        f"({100 * review_frac:.0f}%, limit {100 * args.max_medium_frac:.0f}%)"
    )

    print(
        "Ready to post:",
        approved,
    )

    for x in blanked:
        print(f"Left blank (short uncertain line): {float(x['start']):.2f}-{float(x['end']):.2f}s "
              f"{x.get('resolved_text', '')!r}")

    if args.output:
        print(
            "Saved:",
            args.output,
        )

    if review:

        print()
        print(
            "===== MEDIUM-CONFIDENCE WINDOWS (allowed) ====="
            if allowed_medium
            else "===== REVIEW WINDOWS ====="
        )

        for x in review:

            print(
                f"{x['start']:.2f}s -> "
                f"{x['end']:.2f}s | "
                f"{x.get('confidence')} | "
                f"{x.get('resolved_text', '')}"
            )


if __name__ == "__main__":
    main()
