#!/usr/bin/env python3

import argparse
import base64
import json
import os
import subprocess
import tempfile

from anthropic import Anthropic


MODEL = "claude-sonnet-4-6"
NUM_FRAMES = 12


SYSTEM_PROMPT = """
You are a highly selective visual pre-filter for TikTok livestream clips.

You are seeing ONLY chronological sampled frames from a candidate window.
You do NOT have audio or transcript.

Your job is NOT to reward activity, movement, crowds, intensity, or visual
change by itself.

Your job is:

"Do these frames contain evidence of a COMPLETE or potentially COMPLETE
visual story that deserves expensive multimodal review?"

The most important requirement is a PAYOFF.

A strong candidate usually contains evidence of:

1. BEFORE / SETUP
   A recognizable starting situation.

2. EVENT / CHANGE
   Something meaningful happens.

3. AFTER / PAYOFF
   A visible result, reveal, reaction, reversal, failure, success,
   transformation, confrontation outcome, or other concrete result.

Examples of GOOD evidence:

- prank setup -> reveal -> reactions
- unusual object introduced -> revealed result
- challenge attempt -> success/failure -> reaction
- person confronts someone -> visible escalation/result
- physical action -> fall/collapse/result -> reaction
- before state -> visible transformation
- threat/countdown -> stream suddenly cuts out
- surprising event -> clear facial/group reaction afterward

IMPORTANT FALSE POSITIVES:

The following are NOT enough by themselves:

- a huge crowd appearing
- celebrity arrival
- walking into a building
- changing rooms/locations
- ordinary gym reps
- rep counter being near a goal
- somebody looking exhausted
- somebody lying on the floor
- somebody being helped up
- coaching or motivational speech
- generic hugging
- people gathering
- camera shake
- blurry chaos
- generic laughter
- strong facial expressions with unclear cause
- a person moving closer to another person
- people entering or leaving frame

Do NOT infer that a milestone was reached merely because a counter is
close to its target.

Do NOT infer a confrontation merely because two people stand close.

Do NOT infer a celebration merely because people hug or smile.

Judge ONLY what is actually supported by the chronological frames.

CLASSIFICATION:

YES:
The supplied frames themselves show a concrete visual payoff/result,
or an exceptionally clear before -> event -> after sequence.
This should be rare.

MAYBE:
A genuinely unusual event is visibly occurring and the payoff could
reasonably fall between sampled frames, but the frames do not prove it.
Use MAYBE sparingly.

NO:
There is movement, energy, progression, crowd activity, exercise,
conversation, arrival, exhaustion, or scene change, but no concrete
visual payoff/result is established.

When uncertain between MAYBE and NO, choose NO unless there is a specific
unusual event visible.

Return ONLY JSON:

{
  "decision": "YES" | "MAYBE" | "NO",
  "event_type": "<short description or 'none'>",
  "visual_progression": "<what visibly changes or 'none'>",
  "possible_payoff": "<specific visible payoff/result, or 'none'>",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}
"""


def get_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(
        result.stdout.strip()
    )


def extract_frames(
    video_path,
    duration,
    output_dir,
):
    margin = min(
        1.0,
        duration * 0.03,
    )

    # Seeking extremely close to EOF can legitimately return no frame,
    # especially for short/re-encoded candidate clips. Keep the original
    # small start margin, but guarantee at least 1 second of safety at EOF.
    end_margin = max(
        1.0,
        margin,
    )

    usable_start = margin

    usable_end = max(
        usable_start,
        duration - end_margin,
    )

    if NUM_FRAMES == 1:
        times = [
            duration / 2
        ]
    else:
        times = [
            usable_start
            + (
                (
                    usable_end
                    - usable_start
                )
                * i
                / (
                    NUM_FRAMES - 1
                )
            )
            for i in range(
                NUM_FRAMES
            )
        ]

    frames = []

    for i, t in enumerate(times):
        path = os.path.join(
            output_dir,
            f"frame_{i:02d}_{t:.2f}.png",
        )

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", "scale='min(640,iw)':-2",
            "-c:v", "png",
            "-y",
            path,
        ]

        subprocess.run(
            cmd,
            check=True,
        )

        frames.append(
            (t, path)
        )

    return frames


def encode_image(path):
    with open(
        path,
        "rb",
    ) as f:
        return base64.b64encode(
            f.read()
        ).decode(
            "utf-8"
        )


def clean_json(raw):
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")

        if raw.startswith("json"):
            raw = raw[
                4:
            ].strip()

    try:
        return json.loads(
            raw
        )
    except json.JSONDecodeError:
        a, b = raw.find("{"), raw.rfind("}")
        if a >= 0 and b > a:
            return json.loads(raw[a:b + 1])
        raise


def filter_candidate(video_path):
    if not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set."
        )

    duration = get_duration(
        video_path
    )

    client = Anthropic()

    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(
            video_path,
            duration,
            tmp,
        )

        content = []

        content.append(
            {
                "type": "text",
                "text": (
                    "These are chronological frames "
                    "from one livestream candidate.\n\n"
                    f"Candidate duration: "
                    f"{duration:.2f} seconds.\n\n"
                    "Evaluate only whether the VISUAL "
                    "sequence deserves full review."
                ),
            }
        )

        for t, path in frames:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Frame at "
                        f"{t:.1f} seconds:"
                    ),
                }
            )

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": (
                            "image/png"
                        ),
                        "data": encode_image(
                            path
                        ),
                    },
                }
            )

        # UNREADABLE_REPLY handling: retry once, then mark NO instead of crashing
        messages = [{"role": "user", "content": content}]
        result = None
        for attempt in range(2):
            response = client.messages.create(
                model=MODEL,
                max_tokens=600,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            reply = response.content[0].text if response.content else ""
            try:
                result = clean_json(reply)
                if isinstance(result, dict) and "decision" in result:
                    break
                result = None
            except (json.JSONDecodeError, ValueError):
                result = None
            print(f"WARNING: unreadable reply (attempt {attempt + 1}): {reply[:200]!r}")
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": reply or "(empty)"},
                {"role": "user", "content": "Reply with ONLY the JSON object in the required schema, nothing else."},
            ]
        if result is None:
            result = {
                "decision": "NO",
                "event_type": "unknown",
                "visual_progression": "",
                "possible_payoff": "none",
                "confidence": 0,
                "reason": "model reply could not be read as JSON twice -- skipped this candidate",
            }

    result["video"] = video_path
    result["duration"] = round(
        duration,
        2,
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    result = filter_candidate(
        args.video
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if args.output:
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
            f"\nSaved: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()
