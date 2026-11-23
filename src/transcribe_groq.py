#!/usr/bin/env python3
"""
transcribe_groq.py

Fast cloud transcription using Groq's Whisper API (whisper-large-v3-turbo).
Extracts audio-only from an mp4, chunks it to stay under Groq's 25MB limit,
transcribes each chunk, and stitches results into the SAME JSON format that
local `whisper --output_format json` produces -- so it's a drop-in
replacement. Output works directly with score_transcript_clips.py.

Usage:
    python3 transcribe_groq.py <input.mp4> [output.json]

Requires: GROQ_API_KEY environment variable set.
Requires: pip3 install groq --break-system-packages
Requires: ffmpeg (already installed)
"""

import sys
import os
import subprocess
import json
import tempfile
from groq import Groq

MODEL = "whisper-large-v3-turbo"
CHUNK_SECONDS = 1200  # 20 min chunks, comfortably under 25MB at low bitrate


def get_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def extract_audio_chunk(input_path, start, duration, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
         "-i", input_path, "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "32k", out_path],
        capture_output=True
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_groq.py <input.mp4> [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.rsplit(".", 1)[0] + "_groq.json"

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print('Run: echo \'export GROQ_API_KEY="your_key_here"\' >> ~/.zshrc && source ~/.zshrc')
        sys.exit(1)

    client = Groq()

    duration = get_duration(input_path)
    print(f"Input duration: {duration:.1f}s ({duration/3600:.2f} hours)")

    all_segments = []
    all_words = []
    chunk_start = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_num = 0
        while chunk_start < duration:
            chunk_num += 1
            chunk_dur = min(CHUNK_SECONDS, duration - chunk_start)
            chunk_path = os.path.join(tmpdir, f"chunk_{chunk_num}.mp3")

            print(f"[chunk {chunk_num}] extracting audio {chunk_start:.0f}s -> {chunk_start+chunk_dur:.0f}s ...")
            extract_audio_chunk(input_path, chunk_start, chunk_dur, chunk_path)

            size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            print(f"[chunk {chunk_num}] size: {size_mb:.1f}MB, transcribing via Groq ...")

            with open(chunk_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(chunk_path, f.read()),
                    model=MODEL,
                    response_format="verbose_json",
                    timestamp_granularities=[
                        "word",
                        "segment",
                    ],
                )

            # offset timestamps by chunk_start so they're absolute in the full file
            segs = transcription.segments if hasattr(transcription, "segments") else transcription.get("segments", [])
            for seg in segs:
                item = {
                    "start": seg["start"] + chunk_start,
                    "end": seg["end"] + chunk_start,
                    "text": seg["text"],
                }

                for key in [
                    "avg_logprob",
                    "compression_ratio",
                    "no_speech_prob",
                    "temperature",
                ]:
                    try:
                        value = seg[key]
                    except (KeyError, TypeError):
                        value = getattr(
                            seg,
                            key,
                            None,
                        )

                    if value is not None:
                        item[key] = value

                all_segments.append(item)

            raw_words = getattr(
                transcription,
                "words",
                [],
            ) or []

            for word in raw_words:
                try:
                    word_text = word["word"]
                    word_start = word["start"]
                    word_end = word["end"]
                except (KeyError, TypeError):
                    word_text = getattr(
                        word,
                        "word",
                        "",
                    )
                    word_start = getattr(
                        word,
                        "start",
                        None,
                    )
                    word_end = getattr(
                        word,
                        "end",
                        None,
                    )

                if (
                    word_start is None
                    or word_end is None
                ):
                    continue

                all_words.append({
                    "word": word_text,
                    "start":
                        word_start + chunk_start,
                    "end":
                        word_end + chunk_start,
                })

            print(f"[chunk {chunk_num}] got {len(segs)} segments")
            chunk_start += chunk_dur

    # --- clean-up: muted audio + speech-to-text filler on silence ---
    FILLER = {"thank you", "thank you so much", "thanks for watching", "thank you for watching",
              "you", "bye", "please subscribe", "subtitles by the amaraorg community"}

    def _norm(t):
        return "".join(ch for ch in t.lower() if ch.isalnum() or ch == " ").strip()

    muted, mean_db = False, None
    try:
        if duration and duration < 600:
            vd = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", input_path,
                                 "-af", "volumedetect", "-f", "null", "-"],
                                capture_output=True, text=True).stderr
            for ln in vd.splitlines():
                if "mean_volume:" in ln:
                    mean_db = float(ln.split("mean_volume:")[1].split("dB")[0])
            muted = mean_db is not None and mean_db < -50.0
    except Exception as e:
        print(f"(loudness check skipped: {e})")
    dropped = []
    if muted:
        dropped = list(all_segments)
        all_segments, all_words = [], []
        print(f"MUTED: average loudness {mean_db:.1f} dB -- transcript left empty")
    else:
        kept = []
        for s in all_segments:
            if _norm(s.get("text", "")) in FILLER and float(s["end"]) - float(s["start"]) >= 8.0:
                dropped.append(s)
            else:
                kept.append(s)
        if dropped:
            spans = [(float(s["start"]), float(s["end"])) for s in dropped]
            all_words = [w for w in all_words
                         if not any(a <= w["start"] <= b for a, b in spans)]
            print(f"removed {len(dropped)} filler segment(s) the model invented on silence")
        all_segments = kept

    output = {
        "segments": all_segments,
        "words": all_words,
        "muted": muted,
        "mean_volume_db": mean_db,
        "filler_removed": len(dropped),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(all_segments)} total segments written to {output_path}")


if __name__ == "__main__":
    main()
