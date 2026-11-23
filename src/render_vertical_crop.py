#!/usr/bin/env python3

import argparse
import json
import os
import subprocess

import cv2
import numpy as np


TARGET_ASPECT = 9 / 16


# ------------------------------------------------------------
# Camera tuning
# ------------------------------------------------------------

# Normal portrait crop responsiveness.
CENTER_SMOOTH_ALPHA = 0.14

# Zoom should move slightly slower than position.
ZOOM_SMOOTH_ALPHA = 0.08

# Never make a joint shot wider than this fraction
# of the original landscape width.
MAX_JOINT_WIDTH_FRAC = 0.43

# Minimum extra width required before we bother entering
# wide/joint presentation.
MIN_WIDE_FACTOR = 1.08

# Extra safety padding around builder's target_width.
JOINT_EXTRA_PADDING = 1.08


def lerp(current, target, alpha):
    return (
        current
        + (target - current) * alpha
    )


def clamp(v, lo, hi):
    return max(
        lo,
        min(v, hi),
    )


def make_blurred_background(
    frame,
    output_width,
    output_height,
):
    """
    Create a full 9:16 blurred background without distortion.
    """

    src_h, src_w = frame.shape[:2]

    scale = max(
        output_width / src_w,
        output_height / src_h,
    )

    resized_w = int(
        round(src_w * scale)
    )

    resized_h = int(
        round(src_h * scale)
    )

    resized = cv2.resize(
        frame,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR,
    )

    x1 = max(
        0,
        (resized_w - output_width) // 2,
    )

    y1 = max(
        0,
        (resized_h - output_height) // 2,
    )

    bg = resized[
        y1:y1 + output_height,
        x1:x1 + output_width,
    ]

    bg = cv2.GaussianBlur(
        bg,
        (0, 0),
        sigmaX=28,
        sigmaY=28,
    )

    # Slight darkening makes foreground easier to read.
    bg = (
        bg.astype(np.float32)
        * 0.72
    ).astype(np.uint8)

    return bg


def render_normal_crop(
    frame,
    center_x,
    crop_width,
):
    src_h, src_w = frame.shape[:2]

    x1 = int(
        round(
            center_x
            - crop_width / 2
        )
    )

    x1 = clamp(
        x1,
        0,
        src_w - crop_width,
    )

    x1 = int(x1)

    return frame[
        0:src_h,
        x1:x1 + crop_width,
    ]


def render_wide_joint(
    frame,
    center_x,
    virtual_width,
    output_width,
    output_height,
):
    """
    Take a wider landscape slice while preserving its aspect ratio.

    Because a wider slice cannot fill a 9:16 canvas vertically without
    distortion, scale it proportionally and place it over a blurred
    portrait background.
    """

    src_h, src_w = frame.shape[:2]

    virtual_width = int(
        round(
            clamp(
                virtual_width,
                output_width,
                src_w,
            )
        )
    )

    x1 = int(
        round(
            center_x
            - virtual_width / 2
        )
    )

    x1 = int(
        clamp(
            x1,
            0,
            src_w - virtual_width,
        )
    )

    wide_crop = frame[
        0:src_h,
        x1:x1 + virtual_width,
    ]

    # Scale based on output width.
    scale = (
        output_width
        / virtual_width
    )

    foreground_h = int(
        round(
            src_h * scale
        )
    )

    foreground_h = min(
        foreground_h,
        output_height,
    )

    foreground = cv2.resize(
        wide_crop,
        (
            output_width,
            foreground_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = make_blurred_background(
        frame,
        output_width,
        output_height,
    )

    y1 = (
        output_height
        - foreground_h
    ) // 2

    canvas[
        y1:y1 + foreground_h,
        0:output_width,
    ] = foreground

    return canvas


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Render a smoothed 9:16 speaker-aware crop "
            "with optional wide joint framing."
        )
    )

    ap.add_argument(
        "--clip",
        required=True,
    )

    ap.add_argument(
        "--track",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    with open(args.track) as f:
        raw_track = json.load(f)

    track = {
        int(k): v
        for k, v in raw_track.items()
    }

    if not track:
        print(
            f"ERROR: {args.track} has no tracked positions."
        )
        return

    cap = cv2.VideoCapture(
        args.clip
    )

    src_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    src_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    # Normal portrait crop.
    base_crop_width = int(
        round(
            src_height
            * TARGET_ASPECT
        )
    )

    base_crop_width = min(
        base_crop_width,
        src_width,
    )

    output_width = base_crop_width
    output_height = src_height

    max_joint_width = (
        src_width
        * MAX_JOINT_WIDTH_FRAC
    )

    print(
        f"Source: "
        f"{src_width}x{src_height}"
    )

    print(
        f"Portrait output: "
        f"{output_width}x{output_height}"
    )

    print(
        f"Maximum joint view width: "
        f"{max_joint_width:.0f}px"
    )

    temp_video = (
        args.output.rsplit(
            ".",
            1,
        )[0]
        + "_noaudio.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        temp_video,
        fourcc,
        fps,
        (
            output_width,
            output_height,
        ),
    )

    sorted_frames = sorted(
        track.keys()
    )

    first = track[
        sorted_frames[0]
    ]

    smooth_center_x = float(
        first["pos"][0]
    )

    smooth_width = float(
        base_crop_width
    )

    track_idx = 0
    frame_idx = 0

    joint_frames = 0
    wide_frames = 0

    print(
        "Rendering dynamic vertical crop..."
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        while (
            track_idx
            < len(sorted_frames) - 1
            and sorted_frames[
                track_idx + 1
            ] <= frame_idx
        ):
            track_idx += 1

        data = track[
            sorted_frames[
                track_idx
            ]
        ]

        target_x = float(
            data["pos"][0]
        )

        mode = data.get(
            "mode",
            "single",
        )

        target_width = data.get(
            "target_width"
        )

        # ----------------------------------------------------
        # Position smoothing
        # ----------------------------------------------------

        smooth_center_x = lerp(
            smooth_center_x,
            target_x,
            CENTER_SMOOTH_ALPHA,
        )

        # ----------------------------------------------------
        # Decide desired camera width.
        # ----------------------------------------------------

        desired_width = float(
            base_crop_width
        )

        if (
            mode == "joint"
            and target_width
            is not None
        ):

            joint_frames += 1

            desired_width = max(
                base_crop_width,
                float(
                    target_width
                )
                * JOINT_EXTRA_PADDING,
            )

            desired_width = min(
                desired_width,
                max_joint_width,
            )

        smooth_width = lerp(
            smooth_width,
            desired_width,
            ZOOM_SMOOTH_ALPHA,
        )

        # ----------------------------------------------------
        # Only enter wide presentation once it's actually
        # meaningfully wider than normal.
        # ----------------------------------------------------

        use_wide = (
            smooth_width
            >= (
                base_crop_width
                * MIN_WIDE_FACTOR
            )
        )

        if use_wide:

            wide_frames += 1

            rendered = render_wide_joint(
                frame,
                smooth_center_x,
                smooth_width,
                output_width,
                output_height,
            )

        else:

            rendered = render_normal_crop(
                frame,
                smooth_center_x,
                base_crop_width,
            )

        writer.write(
            rendered
        )

        frame_idx += 1

    cap.release()
    writer.release()

    print(
        f"Frames rendered: "
        f"{frame_idx}"
    )

    if frame_idx:

        print(
            f"Joint metadata: "
            f"{100 * joint_frames / frame_idx:.1f}%"
        )

        print(
            f"Wide presentation: "
            f"{100 * wide_frames / frame_idx:.1f}%"
        )

    print(
        "Muxing original audio back in..."
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_video,
        "-i",
        args.clip,

        "-c:v",
        "copy",

        "-c:a",
        "aac",

        "-map",
        "0:v:0",

        "-map",
        "1:a:0",

        "-shortest",

        args.output,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print(
            "ffmpeg mux failed:"
        )

        print(
            result.stderr[-1500:]
        )

        return

    os.remove(
        temp_video
    )

    print(
        f"Done. {args.output}"
    )


if __name__ == "__main__":
    main()
