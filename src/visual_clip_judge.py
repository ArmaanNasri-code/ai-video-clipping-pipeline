#!/usr/bin/env python3

import argparse
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from anthropic import Anthropic


MODEL = "claude-sonnet-4-6"
NUM_FRAMES = 10


SYSTEM_PROMPT = """
You are a highly selective professional TikTok livestream clip editor.

You are evaluating a RAW candidate window cut from a livestream, using:
1. sampled video frames in chronological order
2. the Whisper transcript
3. timestamps for each sampled frame

THE CANDIDATE IS RAW FOOTAGE, NOT THE FINISHED CLIP.
It is a ~60-70 second window cut around a moment. Before posting, an editor trims it to
the best continuous 15-45 seconds and removes dead time. So:
- Find the BEST continuous 15-45 second stretch inside the window and score THAT.
- Dead time, loading screens or filler OUTSIDE that stretch must NOT lower the score.
- Set recommended_start / recommended_end to that stretch.
- If the payoff is cut off at the very end or the setup is missing at the very start,
  that DOES count against it (the editor cannot add footage that is not there).

Your main question is:

"If a normal TikTok viewer saw only the best 15-45 seconds of this, would they watch to
the end and want to share it?"

Judge the STORY using BOTH what you see and what is said.
A moment can land VISUALLY or VERBALLY. Strong payoffs include:
- prank setups and reveals, an object or screen being revealed
- surprising physical actions, someone entering or leaving
- a big genuine facial or verbal reaction (shock, disbelief, laughing fit, rage)
- a sharp punchline or roast that lands
- a heated argument or confrontation where someone clearly wins or gets exposed
- a shocking, absurd or quotable statement
- embarrassment, getting caught, a lie being exposed
- a challenge with a clear result
- a genuinely emotional or wholesome moment with a clear trigger

A static camera on people talking is FINE if what they say is the payoff.
Do not mark a moment down just for being "talking head" when the words carry it.

A strong clip usually has:

HOOK: an immediate situation, question, unusual visual, claim, challenge, tension,
or curiosity gap within the first few seconds of the best stretch.

PROGRESSION: something meaningfully changes or escalates.

PAYOFF: a reveal, punchline, result, reaction, reversal, or other concrete event
worth waiting for.

REACTION: someone visibly or verbally responds in a meaningful way.

SHAREABILITY: the clip can easily be summarized/captioned in one sentence.

STANDALONE CLARITY: a viewer who never watched the livestream can understand it.
A one-line caption can supply simple context ("Kai finds out his friend lied about X");
it cannot supply weeks of stream lore.

STILL SCORE LOW (these remain rejections):
- inside jokes or references that need stream history to be funny
- generic banter, hype, shouting, profanity or laughter with no punchline or event
- routine gameplay, menus, chat reading, sponsor reads, thank-yous
- a story or argument that never resolves inside the window
- moments that are only interesting if you already know the people

SCORE SCALE (for the best 15-45s stretch):

0-2: Nothing useful. Dead air, filler, contextless conversation.
3-4: Some activity, but no real story or payoff.
5:   Potentially interesting, but too weak for a finished TikTok.
6:   A legitimate moment, but average, context-dependent, or missing a satisfying payoff.
7:   Worth human review, but not clearly postable yet.
8:   Clearly clip-worthy. Strong enough to post after editing.
9:   Very strong candidate with clear hook, progression, payoff, and reaction.
10:  Exceptional standout moment.

VIEWER EVIDENCE (only sometimes supplied):
If told that Twitch viewers clipped this moment, use it as a hint about WHERE the moment
is and that it landed with the live audience. It is NOT proof it works for a stranger on
TikTok: popular live moments are often inside jokes. Never raise a score for views alone;
the footage must earn it.

Return ONLY JSON:

{
  "score": <integer 0-10>,
  "postable": <true or false>,
  "payoff_type": "<visual, verbal, reaction, or none>",
  "visual_hook": "<what makes the viewer curious, or 'none'>",
  "setup": "<what establishes the situation, or 'none'>",
  "progression": "<what changes/develops, or 'none'>",
  "payoff": "<specific payoff/reveal/reaction, or 'none'>",
  "why_keep_watching": "<specific reason a viewer waits for the ending, or 'unclear'>",
  "dead_time": "<low, medium, or high, INSIDE the best stretch>",
  "recommended_start": <number of seconds>,
  "recommended_end": <number of seconds>,
  "caption_idea": "<short factual description of the clip>",
  "reason": "<concise explanation of the score>"
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

    return float(result.stdout.strip())


def load_transcript(path):
    if not path:
        return "(No transcript supplied.)"

    data = json.load(open(path))

    if data.get("muted"):
        return ("(NO AUDIO: this clip's sound is muted/silent -- Twitch muted this part of "
                "the VOD. On TikTok it would play with no sound.)")
    if not data.get("segments"):
        return "(No speech detected in this clip.)"

    lines = []

    for seg in data.get("segments", []):
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg["text"].strip()

        lines.append(
            f"[{start:.1f}s -> {end:.1f}s] {text}"
        )

    return "\n".join(lines)


def extract_frames(video_path, duration, output_dir):
    # Avoid sampling the exact first/last frame because those can
    # sometimes be black/fade frames.
    margin = min(1.0, duration * 0.03)

    usable_start = margin
    usable_end = max(
        usable_start,
        duration - margin,
    )

    if NUM_FRAMES == 1:
        times = [duration / 2]
    else:
        times = [
            usable_start
            + (
                (usable_end - usable_start)
                * i
                / (NUM_FRAMES - 1)
            )
            for i in range(NUM_FRAMES)
        ]

    frames = []

    for i, t in enumerate(times):
        path = os.path.join(
            output_dir,
            f"frame_{i:02d}_{t:.2f}.jpg",
        )

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", "scale='min(768,iw)':-2",
            "-q:v", "4",
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
    with open(path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


def clean_json(raw):
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")

        if raw.startswith("json"):
            raw = raw[4:].strip()

    return json.loads(raw)


def _viewer_hint(video_path):
    """Twitch-viewer evidence for viewer_* candidates (hint only)."""
    try:
        name = os.path.splitext(os.path.basename(video_path))[0]
        if not name.startswith("viewer_"):
            return ""
        meta = json.load(open(os.path.join(os.path.dirname(video_path) or ".", "_viewer_candidates.json")))
        m = meta.get(name)
        if not m:
            return ""
        return ("\n\nVIEWER EVIDENCE: Twitch viewers clipped this moment live; the most-viewed "
                f"viewer clip has {m['views']} views (title: {m.get('title')!r}). "
                "Hint only -- the footage must earn the score.")
    except Exception:
        return ""


def judge(video_path, transcript_path):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set."
        )

    duration = get_duration(
        video_path
    )

    transcript = load_transcript(
        transcript_path
    )

    client = Anthropic()

    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(
            video_path,
            duration,
            tmp,
        )

        content = [
            {
                "type": "text",
                "text": (
                    f"VIDEO DURATION: {duration:.2f} seconds\n\n"
                    "The following images are sampled chronologically "
                    "from the candidate clip."
                ),
            }
        ]

        for t, path in frames:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"FRAME AT {t:.2f} SECONDS"
                    ),
                }
            )

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encode_image(path),
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": (
                    "\nWHISPER TRANSCRIPT:\n\n"
                    + transcript
                    + _viewer_hint(video_path)
                    + "\n\nFind the best 15-45 second stretch and score it."
                ),
            }
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=900,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        def _parse(resp):
            import re as _re
            txt = "".join(getattr(b, "text", "") for b in resp.content)
            try:
                return clean_json(txt)
            except Exception:
                m = _re.search(r"\{.*\}", txt, _re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
            return None

        result = _parse(response)
        if result is None:
            print("  judge reply unreadable -- asking once more")
            response = client.messages.create(
                model=MODEL,
                max_tokens=900,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            result = _parse(response)
        if result is None:
            result = {"score": 0, "postable": False,
                      "reason": "judge reply unreadable (twice) -- rejected automatically"}

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
        "--transcript",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    result = judge(
        args.video,
        args.transcript,
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
            f"\nSaved: {args.output}"
        )


if __name__ == "__main__":
    main()
