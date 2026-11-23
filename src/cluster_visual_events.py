#!/usr/bin/env python3

import argparse
import json


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--cluster-gap",
        type=float,
        default=25.0,
    )

    ap.add_argument(
        "--pre-roll",
        type=float,
        default=12.0,
    )

    ap.add_argument(
        "--post-roll",
        type=float,
        default=15.0,
    )

    ap.add_argument(
        "--top-n",
        type=int,
        default=20,
    )

    args = ap.parse_args()

    data = json.load(
        open(args.input)
    )

    duration = float(
        data["duration"]
    )

    events = sorted(
        data["top_events"],
        key=lambda x: x[
            "time_sec"
        ],
    )

    clusters = []

    current = []

    for event in events:

        if not current:
            current = [
                event
            ]
            continue

        gap = (
            event["time_sec"]
            - current[-1]["time_sec"]
        )

        if gap <= args.cluster_gap:
            current.append(
                event
            )
        else:
            clusters.append(
                current
            )

            current = [
                event
            ]

    if current:
        clusters.append(
            current
        )


    output = []

    for cluster in clusters:

        first = min(
            x["time_sec"]
            for x in cluster
        )

        last = max(
            x["time_sec"]
            for x in cluster
        )

        strongest = max(
            x["event_score"]
            for x in cluster
        )

        total_strength = sum(
            x["event_score"]
            for x in cluster
        )

        start = max(
            0.0,
            first - args.pre_roll,
        )

        end = min(
            duration,
            last + args.post_roll,
        )

        output.append(
            {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": (
                    end - start
                ),
                "first_peak": first,
                "last_peak": last,
                "peak_count": len(
                    cluster
                ),
                "strongest_peak": strongest,
                "total_strength": total_strength,
                "peaks": cluster,
            }
        )


    output.sort(
        key=lambda x: -x[
            "total_strength"
        ]
    )

    output = output[
        :args.top_n
    ]

    result = {
        "video": data["video"],
        "duration": duration,
        "clusters": output,
    }

    with open(
        args.output,
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )


    print(
        "\n===== VISUAL EVENT CLUSTERS ====="
    )

    for i, c in enumerate(
        output,
        start=1,
    ):

        print(
            f"{i:02d}. "
            f"{c['start_sec']:.1f}s"
            f" -> "
            f"{c['end_sec']:.1f}s | "
            f"{c['peak_count']} peaks | "
            f"strength="
            f"{c['total_strength']:.3f}"
        )


if __name__ == "__main__":
    main()
