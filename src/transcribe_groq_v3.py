#!/usr/bin/env python3

import sys
import os
import json
import subprocess
import tempfile
from groq import Groq


MODEL = "whisper-large-v3-turbo"

# Final clips are short, so use smaller chunks than the long-VOD pipeline.
CHUNK_SECONDS = 30

PROMPT = None


def field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def get_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(
        result.stdout.strip()
    )


def extract_audio_chunk(
    input_path,
    start,
    duration,
    out_path,
):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", input_path,

            "-vn",

            # Speech-optimized input.
            "-ac", "1",
            "-ar", "16000",

            # Lossless instead of 32 kbps MP3.
            "-c:a", "flac",

            out_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 transcribe_groq_v2.py "
            "<input.mp4> [output.json]"
        )
        sys.exit(1)

    input_path = sys.argv[1]

    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else input_path.rsplit(".", 1)[0]
        + "_groq_v2.json"
    )

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "ERROR: GROQ_API_KEY is not set."
        )

    client = Groq()

    duration = get_duration(
        input_path
    )

    print(
        f"Input duration: {duration:.2f}s"
    )

    print(
        f"Model: {MODEL}"
    )

    all_segments = []
    all_words = []

    chunk_start = 0.0
    chunk_num = 0

    with tempfile.TemporaryDirectory() as tmpdir:

        while chunk_start < duration:

            chunk_num += 1

            chunk_dur = min(
                CHUNK_SECONDS,
                duration - chunk_start,
            )

            chunk_path = os.path.join(
                tmpdir,
                f"chunk_{chunk_num}.flac",
            )

            print(
                f"[chunk {chunk_num}] "
                f"{chunk_start:.2f}s -> "
                f"{chunk_start + chunk_dur:.2f}s"
            )

            extract_audio_chunk(
                input_path,
                chunk_start,
                chunk_dur,
                chunk_path,
            )

            size_mb = (
                os.path.getsize(chunk_path)
                / 1024
                / 1024
            )

            print(
                f"[chunk {chunk_num}] "
                f"{size_mb:.2f} MB"
            )

            with open(
                chunk_path,
                "rb",
            ) as f:

                transcription = (
                    client.audio.transcriptions.create(
                        file=(
                            os.path.basename(chunk_path),
                            f.read(),
                        ),
                        model=MODEL,
                        language="en",
                        response_format="verbose_json",
                        timestamp_granularities=[
                            "word",
                            "segment",
                        ],
                        temperature=0.0,
                    )
                )

            segs = field(
                transcription,
                "segments",
                [],
            ) or []

            words = field(
                transcription,
                "words",
                [],
            ) or []

            for seg in segs:

                all_segments.append(
                    {
                        "start": (
                            float(
                                field(seg, "start", 0)
                            )
                            + chunk_start
                        ),
                        "end": (
                            float(
                                field(seg, "end", 0)
                            )
                            + chunk_start
                        ),
                        "text": field(
                            seg,
                            "text",
                            "",
                        ),
                        "avg_logprob": field(
                            seg,
                            "avg_logprob",
                            None,
                        ),
                        "no_speech_prob": field(
                            seg,
                            "no_speech_prob",
                            None,
                        ),
                        "compression_ratio": field(
                            seg,
                            "compression_ratio",
                            None,
                        ),
                    }
                )

            for word in words:

                text = field(
                    word,
                    "word",
                    field(
                        word,
                        "text",
                        "",
                    ),
                )

                all_words.append(
                    {
                        "start": (
                            float(
                                field(word, "start", 0)
                            )
                            + chunk_start
                        ),
                        "end": (
                            float(
                                field(word, "end", 0)
                            )
                            + chunk_start
                        ),
                        "word": text,
                    }
                )

            print(
                f"[chunk {chunk_num}] "
                f"{len(segs)} segments, "
                f"{len(words)} words"
            )

            chunk_start += chunk_dur

    output = {
        "model": MODEL,
        "duration": duration,
        "segments": all_segments,
        "words": all_words,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Done: {len(all_segments)} segments"
    )

    print(
        f"Done: {len(all_words)} timed words"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
