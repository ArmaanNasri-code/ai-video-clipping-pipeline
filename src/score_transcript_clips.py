#!/usr/bin/env python3

import sys
import json
import os
from anthropic import Anthropic


MODEL = "claude-sonnet-4-6"


SYSTEM_PROMPT = """
You are an extremely selective TikTok clip editor scouting moments from a livestream transcript.

You are given a short raw Whisper transcript window. Whisper may contain:
- misheard words
- duplicated words
- garbled phrases
- noisy crosstalk

Your job is NOT to reward energy.

Your job is to answer:

"Would a normal TikTok viewer have a reason to keep watching this moment from setup to payoff?"

Most livestream conversation is NOT clip-worthy.

IMPORTANT:
Do NOT score something highly merely because it contains:
- shouting
- chaos
- profanity
- repeated phrases
- trash talk
- an argument
- laughter
- fast back-and-forth
- hype
- vague escalation
- a possible reaction
- a provocative sentence with no real setup/payoff

Those things can support a clip, but they are not themselves enough.

A strong clip usually has some combination of:

1. HOOK / CURIOSITY
   There is an immediate premise, question, unusual situation, tension, challenge,
   claim, accusation, reveal setup, or other reason to keep watching.

2. PROGRESSION
   Something meaningfully develops. The clip does not stay at the same level
   the entire time.

3. PAYOFF / CHANGE
   Something actually happens:
   - punchline
   - reveal
   - unexpected answer
   - prank payoff
   - challenge result
   - confrontation turning point
   - shocking admission
   - reversal
   - concrete reaction-worthy event

4. REACTION
   Someone responds in a meaningful way: surprise, laughter, disbelief,
   anger, embarrassment, celebration, etc.

5. SHAREABILITY
   The moment is easy to describe, caption, repost, or send to someone.

6. STANDALONE CLARITY
   A viewer who was not watching the livestream can understand why this matters.

CRITICAL RULE:
Before scoring 8+, you must be able to identify:
- the setup
- what changes/happens
- the payoff

If any of those are missing, score 6 or below.

A transcript-only window can miss visual humor or visual reactions.
If the transcript suggests a possibly strong VISUAL event but does not prove it,
you may score 7, but mark postable=false so it can be visually reviewed later.

SCORE SCALE:

0-2:
Nothing useful. Background talk, logistics, dead air, filler, fragmented speech.

3-4:
Some personality or energy, but no meaningful event or payoff.

5:
Interesting conversation or mild comedic/dramatic beat, but not enough reason
for a viewer to watch a full TikTok.

6:
A legitimate moment exists, but it is weak, minor, context-dependent,
too drawn out, or lacks a strong payoff.

7:
Promising candidate for visual review.
There is a real setup/event/reaction possibility, but transcript alone does not
justify posting it automatically.

8:
Clearly clip-worthy.
Strong setup -> development -> payoff/reaction.
A viewer has a clear reason to keep watching.

9:
Very strong post candidate.
Memorable, funny, shocking, dramatic, or highly shareable with minimal dead time.

10:
Exceptional.
Obvious standout/viral-quality moment with a strong hook, progression,
payoff, and memorable reaction.

Return ONLY valid JSON in this exact schema:

{
  "score": <integer 0-10>,
  "hook": <integer 0-2>,
  "progression": <integer 0-2>,
  "payoff": <integer 0-2>,
  "reaction": <integer 0-1>,
  "shareability": <integer 0-2>,
  "standalone": <integer 0-1>,
  "dead_air_risk": <integer 0-2>,
  "postable": <true or false>,
  "setup": "<short description, or 'none'>",
  "payoff_description": "<short description, or 'none'>",
  "why_keep_watching": "<one short sentence, or 'unclear'>",
  "reason": "<one concise sentence explaining the final score>"
}

SCORING GUIDANCE:

The component scores are diagnostic, not a mechanical sum.

A score of 8+ should normally require:
- hook >= 1
- progression >= 1
- payoff >= 1
- postable = true
- dead_air_risk <= 1

If the window is mostly generic shouting/arguing/roasting with no meaningful
turning point, cap it at 6.

If the strongest justification is only:
"high energy," "good reactions," "chaotic," "escalating," or "funny banter,"
cap it at 6 unless you can name the actual event/payoff.

Do not invent visual events that are not supported by the transcript.
"""


def format_window(segments):
    lines = []

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        m_s, s_s = divmod(start, 60)
        m_e, s_e = divmod(end, 60)

        lines.append(
            f"[{int(m_s):02d}:{s_s:04.1f} -> "
            f"{int(m_e):02d}:{s_e:04.1f}] {text}"
        )

    return "\n".join(lines)


def normalize_result(result):
    defaults = {
        "score": 0,
        "hook": 0,
        "progression": 0,
        "payoff": 0,
        "reaction": 0,
        "shareability": 0,
        "standalone": 0,
        "dead_air_risk": 2,
        "postable": False,
        "setup": "none",
        "payoff_description": "none",
        "why_keep_watching": "unclear",
        "reason": "No valid scoring result.",
    }

    for key, value in defaults.items():
        if key not in result:
            result[key] = value

    # Clamp integer fields defensively.
    bounds = {
        "score": (0, 10),
        "hook": (0, 2),
        "progression": (0, 2),
        "payoff": (0, 2),
        "reaction": (0, 1),
        "shareability": (0, 2),
        "standalone": (0, 1),
        "dead_air_risk": (0, 2),
    }

    for key, (lo, hi) in bounds.items():
        try:
            result[key] = int(result[key])
        except Exception:
            result[key] = defaults[key]

        result[key] = max(
            lo,
            min(
                hi,
                result[key],
            ),
        )

    result["postable"] = bool(
        result["postable"]
    )

    # Hard guardrail:
    # 8+ is not allowed unless the structure is genuinely present.
    if result["score"] >= 8:
        if (
            result["hook"] < 1
            or result["progression"] < 1
            or result["payoff"] < 1
            or not result["postable"]
            or result["dead_air_risk"] > 1
        ):
            result["score"] = 7
            result["postable"] = False

    return result


def score_window(
    client,
    window_text,
):

    try:

        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {
                        "type": "ephemeral"
                    },
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Evaluate this livestream transcript "
                        "for TikTok watchability.\n\n"
                        + window_text
                    ),
                }
            ],
        )

        raw = response.content[
            0
        ].text.strip()

        if raw.startswith("```"):
            raw = raw.strip("`")

            if raw.startswith("json"):
                raw = raw[
                    4:
                ]

        parsed = json.loads(
            raw
        )

        return normalize_result(
            parsed
        )

    except Exception as e:

        return {
            "score": -1,
            "hook": 0,
            "progression": 0,
            "payoff": 0,
            "reaction": 0,
            "shareability": 0,
            "standalone": 0,
            "dead_air_risk": 2,
            "postable": False,
            "setup": "none",
            "payoff_description": "none",
            "why_keep_watching": "unclear",
            "reason": f"ERROR: {e}",
        }


def main():

    if len(sys.argv) < 2:

        print(
            "Usage: python3 score_transcript_clips.py "
            "<transcript.json> "
            "[window_seconds] "
            "[step_seconds]"
        )

        sys.exit(1)


    transcript_path = sys.argv[
        1
    ]

    window_seconds = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 35
    )

    step_seconds = (
        float(sys.argv[3])
        if len(sys.argv) > 3
        else 20
    )


    if not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):

        print(
            "ERROR: ANTHROPIC_API_KEY "
            "environment variable not set."
        )

        sys.exit(1)


    client = Anthropic()


    data = json.load(
        open(
            transcript_path
        )
    )

    all_segments = data[
        "segments"
    ]

    if not all_segments:

        print(
            "No segments found in transcript."
        )

        sys.exit(1)


    total_duration = all_segments[
        -1
    ]["end"]

    print(
        f"Transcript duration: "
        f"{total_duration:.1f}s"
    )

    print(
        f"Window: "
        f"{window_seconds}s, "
        f"step: {step_seconds}s"
    )


    out_path = (
        transcript_path.replace(
            ".json",
            "_scores.json",
        )
    )


    results = []
    already_done = set()


    if os.path.exists(
        out_path
    ):

        try:

            with open(
                out_path
            ) as f:

                results = json.load(
                    f
                )

            already_done = {
                r["window_start"]
                for r in results
            }

            print(
                f"Resuming: found "
                f"{len(results)} previously "
                f"scored windows."
            )

        except (
            json.JSONDecodeError,
            KeyError,
        ):

            print(
                "Existing scores file "
                "unreadable, starting fresh."
            )

            results = []


    print()


    window_start = 0.0


    while (
        window_start
        < total_duration
    ):

        window_end = (
            window_start
            + window_seconds
        )

        if (
            window_start
            in already_done
        ):

            window_start += (
                step_seconds
            )

            continue


        window_segs = [
            s
            for s in all_segments
            if (
                window_start
                <= s["start"]
                < window_end
            )
        ]


        if window_segs:

            window_text = (
                format_window(
                    window_segs
                )
            )

            result = score_window(
                client,
                window_text,
            )

            result[
                "window_start"
            ] = window_start

            result[
                "window_end"
            ] = window_end

            results.append(
                result
            )


            m_s, s_s = divmod(
                window_start,
                60,
            )

            print(
                f"[{int(m_s):02d}:"
                f"{s_s:04.1f}] "
                f"score="
                f"{result['score']:>2} "
                f"postable="
                f"{str(result['postable']):5s} "
                f"| "
                f"{result['reason']}"
            )


            with open(
                out_path,
                "w",
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=2,
                )


        window_start += (
            step_seconds
        )


    print(
        f"\nSaved full results to "
        f"{out_path}"
    )


    top = sorted(
        [
            r
            for r in results
            if r["score"] >= 0
        ],
        key=lambda r: (
            -r["score"],
            not r.get(
                "postable",
                False,
            ),
        ),
    )[:15]


    print(
        "\n--- TOP CANDIDATES ---"
    )


    for r in top:

        m_s, s_s = divmod(
            r["window_start"],
            60,
        )

        print(
            f"[{int(m_s):02d}:"
            f"{s_s:04.1f}] "
            f"score="
            f"{r['score']:>2} "
            f"postable="
            f"{r.get('postable', False)} "
            f"| {r['reason']}"
        )


if __name__ == "__main__":
    main()
