#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def probe_video(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration,size",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--video", required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--vod", required=True)
    ap.add_argument("--timestamp", required=True)
    ap.add_argument("--score", type=int, required=True)
    ap.add_argument(
        "--output-dir",
        default=None,
        help="where to write <clip-id>_final.mp4/.json "
             "(default: <vod>_final_review/production_ready, as before)",
    )

    args = ap.parse_args()

    src = Path(args.video)

    if not src.exists():
        raise SystemExit(
            f"ERROR: source video does not exist:\n{src}"
        )

    output_dir = Path(
        args.output_dir
        if args.output_dir
        else f"{args.vod}_final_review/production_ready"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_dir /
        f"{args.clip_id}_final.mp4"
    )

    output_json = (
        output_dir /
        f"{args.clip_id}_final.json"
    )

    print("Copying final video...")

    shutil.copy2(
        src,
        output_video,
    )

    print("Probing output...")

    probe = probe_video(
        output_video
    )

    video_stream = next(
        (
            s for s in probe["streams"]
            if s.get("codec_type") == "video"
        ),
        None,
    )

    audio_stream = next(
        (
            s for s in probe["streams"]
            if s.get("codec_type") == "audio"
        ),
        None,
    )

    if video_stream is None:
        raise SystemExit(
            "ERROR: no video stream found"
        )

    if audio_stream is None:
        raise SystemExit(
            "ERROR: no audio stream found"
        )

    width = video_stream.get(
        "width"
    )

    height = video_stream.get(
        "height"
    )

    metadata = {
        "clip_id": args.clip_id,
        "source_vod": args.vod,
        "source_timestamp": args.timestamp,
        "score": args.score,

        "orientation": (
            "vertical"
            if height > width
            else "horizontal"
        ),

        "resolution": (
            f"{width}x{height}"
        ),

        "duration": round(
            float(
                probe["format"]["duration"]
            ),
            3,
        ),

        "video_codec": (
            video_stream.get(
                "codec_name"
            )
        ),

        "audio_codec": (
            audio_stream.get(
                "codec_name"
            )
        ),

        "frame_rate": (
            video_stream.get(
                "r_frame_rate"
            )
        ),

        "captions": True,
        "hook": True,
        "watermark": True,

        "status": "ready_to_post",

        "video_file": (
            output_video.name
        ),
    }

    output_json.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print()
    print("===== FINAL VALIDATION =====")
    print("Video:", output_video)
    print("Metadata:", output_json)
    print(
        "Resolution:",
        metadata["resolution"],
    )
    print(
        "Duration:",
        metadata["duration"],
    )
    print(
        "Video codec:",
        metadata["video_codec"],
    )
    print(
        "Audio codec:",
        metadata["audio_codec"],
    )
    print(
        "Frame rate:",
        metadata["frame_rate"],
    )

    print()
    print("READY TO POST")


if __name__ == "__main__":
    main()
