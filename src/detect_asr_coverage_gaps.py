#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import tempfile
import wave

import torch

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
)


def load_audio_from_video(video_path, sr=16000):
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as f:
        wav_path = f.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-acodec",
                "pcm_s16le",
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(
                wf.getnframes()
            )

        return (
            torch.frombuffer(
                bytearray(frames),
                dtype=torch.int16,
            ).float()
            / 32768.0
        )

    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def merged_intersections(
    start,
    end,
    intervals,
):
    pieces = []

    for a, b in intervals:
        a = max(
            float(a),
            start,
        )
        b = min(
            float(b),
            end,
        )

        if b > a:
            pieces.append(
                [a, b]
            )

    if not pieces:
        return []

    pieces.sort(
        key=lambda x: x[0]
    )

    merged = [
        pieces[0]
    ]

    for a, b in pieces[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(
                merged[-1][1],
                b,
            )
        else:
            merged.append(
                [a, b]
            )

    return merged


def union_coverage(
    start,
    end,
    intervals,
):
    merged = merged_intersections(
        start,
        end,
        intervals,
    )

    return sum(
        b - a
        for a, b in merged
    )


def subtract_intervals(
    start,
    end,
    covered_intervals,
):
    """
    Return pieces of [start, end] not covered
    by covered_intervals.
    """

    covered = merged_intersections(
        start,
        end,
        covered_intervals,
    )

    if not covered:
        return [
            [start, end]
        ]

    gaps = []
    cursor = start

    for a, b in covered:
        if a > cursor:
            gaps.append(
                [cursor, a]
            )

        cursor = max(
            cursor,
            b,
        )

    if cursor < end:
        gaps.append(
            [cursor, end]
        )

    return gaps


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
    )

    ap.add_argument(
        "--asr",
        required=True,
    )

    ap.add_argument(
        "--min-speech",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--min-asr-coverage",
        type=float,
        default=0.55,
    )

    ap.add_argument(
        "--min-uncovered",
        type=float,
        default=0.75,
        help=(
            "Minimum uncovered VAD speech span "
            "to retry."
        ),
    )

    ap.add_argument(
        "--retry-padding",
        type=float,
        default=0.50,
        help=(
            "Context padding added around each "
            "uncovered retry span."
        ),
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    with open(args.asr) as f:
        data = json.load(f)

    segments = data.get(
        "segments",
        [],
    )

    asr_intervals = [
        (
            float(s["start"]),
            float(s["end"]),
        )
        for s in segments
        if s.get(
            "text",
            "",
        ).strip()
    ]

    print(
        "Loading audio + running VAD..."
    )

    wav = load_audio_from_video(
        args.video
    )

    model = load_silero_vad()

    speech = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        return_seconds=True,
    )

    failures = []
    uncovered_spans = []
    retry_windows = []

    for s in speech:
        start = float(
            s["start"]
        )
        end = float(
            s["end"]
        )

        speech_dur = (
            end - start
        )

        if (
            speech_dur
            < args.min_speech
        ):
            continue

        covered = union_coverage(
            start,
            end,
            asr_intervals,
        )

        coverage = (
            covered / speech_dur
            if speech_dur > 0
            else 0.0
        )

        if (
            coverage
            >= args.min_asr_coverage
        ):
            continue

        failure = {
            "start":
                start,
            "end":
                end,
            "speech_duration":
                speech_dur,
            "asr_covered_seconds":
                covered,
            "coverage":
                coverage,
        }

        failures.append(
            failure
        )

        gaps = subtract_intervals(
            start,
            end,
            asr_intervals,
        )

        for gap_start, gap_end in gaps:
            gap_duration = (
                gap_end - gap_start
            )

            if (
                gap_duration
                < args.min_uncovered
            ):
                continue

            span = {
                "start":
                    gap_start,
                "end":
                    gap_end,
                "duration":
                    gap_duration,
                "parent_vad_start":
                    start,
                "parent_vad_end":
                    end,
                "parent_coverage":
                    coverage,
            }

            uncovered_spans.append(
                span
            )

            retry_windows.append({
                "start":
                    max(
                        0.0,
                        gap_start
                        - args.retry_padding,
                    ),
                "end":
                    gap_end
                    + args.retry_padding,
                "uncovered_start":
                    gap_start,
                "uncovered_end":
                    gap_end,
                "uncovered_duration":
                    gap_duration,
                "padding":
                    args.retry_padding,
            })

    print()
    print(
        "===== BROAD VAD COVERAGE FAILURES ====="
    )

    if not failures:
        print(
            "No suspicious coverage gaps found."
        )

    for f in failures:
        print()
        print(
            f"{f['start']:.2f} -> "
            f"{f['end']:.2f}"
        )
        print(
            "speech_duration:",
            round(
                f[
                    "speech_duration"
                ],
                2,
            ),
        )
        print(
            "asr_covered_seconds:",
            round(
                f[
                    "asr_covered_seconds"
                ],
                2,
            ),
        )
        print(
            "asr_coverage:",
            round(
                f["coverage"]
                * 100,
                1,
            ),
            "%",
        )

    print()
    print(
        "===== EXACT UNCOVERED SPEECH SPANS ====="
    )

    if not uncovered_spans:
        print(
            "No retry-worthy uncovered spans."
        )

    for i, span in enumerate(
        uncovered_spans,
        1,
    ):
        print(
            f"{i}. "
            f"{span['start']:.2f} -> "
            f"{span['end']:.2f} "
            f"({span['duration']:.2f}s)"
        )

    print()
    print(
        "===== PADDED RETRY WINDOWS ====="
    )

    for i, window in enumerate(
        retry_windows,
        1,
    ):
        print(
            f"{i}. "
            f"{window['start']:.2f} -> "
            f"{window['end']:.2f} "
            f"[missing "
            f"{window['uncovered_start']:.2f} -> "
            f"{window['uncovered_end']:.2f}]"
        )

    print()
    print(
        "TOTAL FAILURES:",
        len(failures),
    )

    print(
        "TOTAL UNCOVERED SPANS:",
        len(uncovered_spans),
    )

    print(
        "TOTAL RETRY WINDOWS:",
        len(retry_windows),
    )

    if args.output:
        output = {
            "failures":
                failures,
            "uncovered_spans":
                uncovered_spans,
            "retry_windows":
                retry_windows,
            "count":
                len(failures),
            "uncovered_count":
                len(
                    uncovered_spans
                ),
            "retry_window_count":
                len(
                    retry_windows
                ),
            "settings": {
                "min_speech":
                    args.min_speech,
                "min_asr_coverage":
                    args.min_asr_coverage,
                "min_uncovered":
                    args.min_uncovered,
                "retry_padding":
                    args.retry_padding,
            },
        }

        with open(
            args.output,
            "w",
        ) as f:
            json.dump(
                output,
                f,
                indent=2,
            )

        print()
        print(
            "Saved:",
            args.output,
        )


if __name__ == "__main__":
    main()
