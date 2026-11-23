#!/usr/bin/env python3

import argparse
import json
import cv2
import numpy as np

from pathlib import Path
from ultralytics import YOLO


def frame_hist(frame):
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        hist,
        hist,
    )

    return hist


def hist_distance(a, b):
    return cv2.compareHist(
        a,
        b,
        cv2.HISTCMP_BHATTACHARYYA,
    )


def motion_score(a, b):
    diff = cv2.absdiff(
        a,
        b,
    )

    return float(
        np.mean(diff)
        / 255.0
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--sample-every",
        type=float,
        default=3.0,
    )

    ap.add_argument(
        "--top-n",
        type=int,
        default=80,
    )

    ap.add_argument(
        "--min-gap",
        type=float,
        default=8.0,
    )

    args = ap.parse_args()

    weights = (
        Path(__file__).resolve().parent
        / "yolov8n.pt"
    )

    if not weights.exists():
        raise SystemExit(
            f"Missing YOLO weights: {weights}"
        )

    print(
        f"Loading YOLO person detector: "
        f"{weights.name}"
    )

    yolo = YOLO(
        str(weights)
    )

    cap = cv2.VideoCapture(
        args.video
    )

    if not cap.isOpened():
        raise SystemExit(
            f"Could not open {args.video}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = float(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        raise SystemExit(
            "Invalid FPS"
        )

    duration = (
        frame_count / fps
    )

    sample_frames = max(
        1,
        int(
            round(
                fps
                * args.sample_every
            )
        ),
    )

    print(
        f"Video duration: "
        f"{duration / 60:.1f} min"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Sequential sampling every "
        f"{args.sample_every:.1f}s "
        f"(~{sample_frames} frames)"
    )

    rows = []

    frame_index = 0
    sample_index = 0

    prev_hist = None
    prev_gray = None

    while True:

        ok = cap.grab()

        if not ok:
            break

        if (
            frame_index
            % sample_frames
            != 0
        ):
            frame_index += 1
            continue

        ok, frame = cap.retrieve()

        if not ok:
            frame_index += 1
            continue

        time_sec = (
            frame_index / fps
        )

        h, w = frame.shape[:2]

        scale = min(
            640.0 / max(w, 1),
            1.0,
        )

        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (
                    int(w * scale),
                    int(h * scale),
                ),
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        hist = frame_hist(
            frame
        )

        if (
            prev_hist is None
            or prev_gray is None
        ):
            scene = 0.0
            motion = 0.0

        else:
            scene = hist_distance(
                prev_hist,
                hist,
            )

            motion = motion_score(
                prev_gray,
                gray,
            )

        # Raw instantaneous visual activity.
        raw_score = (
            2.5 * scene
            + 2.0 * motion
        )

        # --------------------------------------------------
        # V3 SEMANTIC HUMAN ACTIVITY
        # --------------------------------------------------

        detection = yolo.predict(
            source=frame,
            classes=[0],
            conf=0.35,
            imgsz=640,
            verbose=False,
        )[0]

        person_count = 0
        largest_person_area = 0.0
        largest_person_center_y = 0.5

        frame_h, frame_w = frame.shape[:2]

        frame_area = float(
            max(
                frame_h * frame_w,
                1,
            )
        )

        if (
            detection.boxes is not None
            and len(detection.boxes) > 0
        ):
            person_count = int(
                len(detection.boxes)
            )

            for box in detection.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [
                    float(v)
                    for v in box
                ]

                bw = max(
                    0.0,
                    x2 - x1,
                )

                bh = max(
                    0.0,
                    y2 - y1,
                )

                area_ratio = (
                    bw * bh
                ) / frame_area

                if (
                    area_ratio
                    > largest_person_area
                ):
                    largest_person_area = (
                        area_ratio
                    )

                    largest_person_center_y = (
                        ((y1 + y2) / 2.0)
                        / max(frame_h, 1)
                    )

        if rows:
            prev_person_count = int(
                rows[-1].get(
                    "person_count",
                    person_count,
                )
            )

            prev_largest_area = float(
                rows[-1].get(
                    "largest_person_area",
                    largest_person_area,
                )
            )

            prev_center_y = float(
                rows[-1].get(
                    "largest_person_center_y",
                    largest_person_center_y,
                )
            )

        else:
            prev_person_count = person_count
            prev_largest_area = (
                largest_person_area
            )
            prev_center_y = (
                largest_person_center_y
            )

        person_count_change = abs(
            person_count
            - prev_person_count
        )

        crowd_increase = max(
            0,
            person_count
            - prev_person_count,
        )

        large_person_change = abs(
            largest_person_area
            - prev_largest_area
        )

        vertical_shift = abs(
            largest_person_center_y
            - prev_center_y
        )

        # Human-change signal.
        # Count changes alone are intentionally weak
        # because camera movement can create YOLO jitter.
        semantic_score = (
            0.22
            * min(
                person_count_change,
                4,
            )
            + 0.18
            * min(
                crowd_increase,
                4,
            )
            + 2.5
            * min(
                large_person_change,
                0.40,
            )
            + 1.25
            * min(
                vertical_shift,
                0.40,
            )
        )

        # Reward moments where human-state change
        # happens alongside actual visual activity.
        semantic_bonus = (
            semantic_score
            * min(
                1.0,
                0.35
                + raw_score / 2.5,
            )
        )

        # Basic frame-quality checks.
        brightness = float(
            np.mean(gray) / 255.0
        )

        blur_variance = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        # Look at roughly the previous 18 seconds
        # when sampling every ~3 seconds.
        recent_rows = rows[-6:]

        if recent_rows:
            recent_activity = float(
                np.mean(
                    [
                        x.get(
                            "raw_score",
                            x["event_score"],
                        )
                        for x in recent_rows
                    ]
                )
            )

            active_fraction = float(
                np.mean(
                    [
                        1.0
                        if x.get(
                            "raw_score",
                            x["event_score"],
                        ) >= 0.75
                        else 0.0
                        for x in recent_rows
                    ]
                )
            )
        else:
            recent_activity = 0.0
            active_fraction = 0.0

        # Reward sustained visual activity.
        sustained_bonus = (
            0.70 * recent_activity
            + 0.80 * active_fraction
        )

        # A single isolated camera cut should not dominate.
        isolation_penalty = 0.0

        if (
            raw_score >= 2.0
            and recent_activity < 0.55
        ):
            isolation_penalty = 0.75

        # Penalize unusable frames.
        quality_multiplier = 1.0

        if brightness < 0.05:
            quality_multiplier *= 0.20

        elif brightness < 0.10:
            quality_multiplier *= 0.55

        if blur_variance < 12:
            quality_multiplier *= 0.45

        elif blur_variance < 25:
            quality_multiplier *= 0.75

        event_score = max(
            0.0,
            (
                raw_score
                + sustained_bonus
                + semantic_bonus
                - isolation_penalty
            )
            * quality_multiplier,
        )

        rows.append(
            {
                "time_sec": round(
                    time_sec,
                    3,
                ),
                "scene_change": round(
                    float(scene),
                    6,
                ),
                "motion": round(
                    float(motion),
                    6,
                ),
                "raw_score": round(
                    float(raw_score),
                    6,
                ),
                "recent_activity": round(
                    float(recent_activity),
                    6,
                ),
                "active_fraction": round(
                    float(active_fraction),
                    6,
                ),
                "brightness": round(
                    float(brightness),
                    6,
                ),
                "blur_variance": round(
                    float(blur_variance),
                    3,
                ),
                "person_count": int(
                    person_count
                ),
                "person_count_change": int(
                    person_count_change
                ),
                "crowd_increase": int(
                    crowd_increase
                ),
                "largest_person_area": round(
                    float(
                        largest_person_area
                    ),
                    6,
                ),
                "largest_person_center_y": round(
                    float(
                        largest_person_center_y
                    ),
                    6,
                ),
                "large_person_change": round(
                    float(
                        large_person_change
                    ),
                    6,
                ),
                "vertical_shift": round(
                    float(vertical_shift),
                    6,
                ),
                "semantic_score": round(
                    float(semantic_score),
                    6,
                ),
                "semantic_bonus": round(
                    float(semantic_bonus),
                    6,
                ),
                "event_score": round(
                    float(event_score),
                    6,
                ),
            }
        )

        prev_hist = hist
        prev_gray = gray

        sample_index += 1

        if (
            sample_index % 500
            == 0
        ):
            print(
                f"  scanned "
                f"{time_sec / 60:.1f} min"
            )

        frame_index += 1

    cap.release()

    print(
        f"\nCollected "
        f"{len(rows)} samples."
    )

    ranked = sorted(
        rows,
        key=lambda x: -x[
            "event_score"
        ],
    )

    selected = []

    for row in ranked:

        too_close = any(
            abs(
                row["time_sec"]
                - old["time_sec"]
            )
            < args.min_gap
            for old in selected
        )

        if too_close:
            continue

        selected.append(
            row
        )

        if (
            len(selected)
            >= args.top_n
        ):
            break

    result = {
        "video": args.video,
        "duration": duration,
        "sample_every": args.sample_every,
        "top_events": selected,
        "all_samples": rows,
    }

    Path(
        args.output
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "\n===== TOP VISUAL EVENTS ====="
    )

    for i, row in enumerate(
        selected[:30],
        start=1,
    ):

        total = int(
            row["time_sec"]
        )

        mins, secs = divmod(
            total,
            60,
        )

        print(
            f"{i:02d}. "
            f"{mins:03d}:{secs:02d} | "
            f"event="
            f"{row['event_score']:.4f} | "
            f"scene="
            f"{row['scene_change']:.4f} | "
            f"motion="
            f"{row['motion']:.4f}"
        )


if __name__ == "__main__":
    main()
