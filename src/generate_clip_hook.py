#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path

from anthropic import Anthropic


MODEL = "claude-sonnet-4-5"


SYSTEM = """
You write short on-screen hooks for TikTok/YouTube Shorts clips.

The hook appears at the TOP of the video for the first several seconds.

Rules:

1. Describe the actual interesting event of the clip.
2. Do NOT invent facts, names, relationships, or events.
3. Prefer concrete events over vague clickbait.
4. The viewer should understand why they should keep watching.
5. Do not spoil the full payoff if suspense is useful.
6. Keep it extremely short.
7. Ideal length: 3-7 words.
8. Maximum: 8 words.
9. No emoji.
10. No hashtags.
11. No quotation marks.
12. No ending punctuation.
13. Avoid generic hooks like:
   - WATCH TILL THE END
   - THIS IS CRAZY
   - YOU WON'T BELIEVE THIS
14. Use natural wording a human clipping account would actually post.
15. Describe only what is SEEN or HEARD happening. Never state or imply a real
    person's condition (drunk, high), health or disability, character (psycho,
    weirdo), sexuality, crimes or allegations (abuser, woman-beater, racist,
    steroids), even if someone in the clip says it.
16. No slurs or insults about people. Use neutral words: uninvited, rival,
    fighter, guest, streamer.
17. Return valid JSON only.

Schema:

{
  "hook": "SHORT HOOK",
  "confidence": "high|medium|low",
  "reason": "brief explanation"
}
"""


RISKY_HOOK = re.compile(
    r"\b(?:drunk|wasted|tipsy|(?:is|was|so|getting|got|gets|super|mad|hella)\s+high|stoned|"
    r"on drugs|crackhead|junkie|addict|"
    r"disab\w*|retard\w*|autis\w*|cripple\w*|mental(?:ly)?|cross[- ]?eyed|"
    r"psycho\w*|weirdo|ugly|fat|"
    r"wom[ae]n[- ]?beat\w*|wife[- ]?beat\w*|abuser|abusive|rapist|pedo\w*|groomer|"
    r"racist|sexist|misogyn\w*|criminal|felon|thief|steroid\w*|juiced|"
    r"gay|nigg\w*|fag\w*|whore|slut|bitch\w*|"
    # safety2: mocking + physical traits
    r"chink\w*|spic|tranny|trannies|towel[- ]?head|"
    r"fatass|fatty|fatso|obese|chubby|midget|dwarf|bald(?:ing|ie|y)?|hairline|forehead|"
    r"buck[- ]?teeth|unibrow|lisp\w*|stutter\w*|"
    r"(?:mock\w*|roast\w*|clown\w*|cook(?:s|ed|ing)?|flam(?:e|es|ed|ing)|diss(?:es|ed|ing)?|"
    r"insult\w*|violat\w*|laugh(?:s|ed|ing)?\s+at|mak(?:e|es|ing)\s+fun\s+of|made\s+fun\s+of|"
    r"jok(?:e|es|ed|ing)\s+about)\s+(?:\w+\s+){0,3}?"
    r"(?:eyes?|face|teeth|tooth|nose|ears?|lips?|chin|skin|acne|hair|weight|body|height|size|"
    r"looks|voice|accent|glasses|braces|outfit))\b",
    re.I,
)


def clean_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return json.loads(text)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--resolved",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--context",
        default="",
    )

    args = ap.parse_args()

    if not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        raise SystemExit(
            "ERROR: ANTHROPIC_API_KEY not set"
        )

    resolved_path = Path(
        args.resolved
    )

    data = json.load(
        open(
            resolved_path,
            encoding="utf-8",
        )
    )

    transcript_parts = []

    for region in data:

        text = (
            region.get(
                "resolved_text",
                "",
            )
            .strip()
        )

        if not text:
            continue

        transcript_parts.append(
            f"[{region['start']:.2f}-"
            f"{region['end']:.2f}] "
            f"{text}"
        )

    transcript = "\n".join(
        transcript_parts
    )

    context = (
        args.context.strip()
        or "[none]"
    )

    prompt = f"""
Clip context:
{context}

Resolved transcript:
{transcript}

Generate ONE short on-screen hook.

Remember:
- 3-7 words preferred
- maximum 8
- concrete
- natural
- no invented details
- no generic clickbait
"""

    client = Anthropic()

    messages = [{"role": "user", "content": prompt}]
    result = None
    for attempt in range(3):
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            temperature=0,
            system=SYSTEM,
            messages=messages,
        )
        result = clean_json(
            response.content[0].text
        )
        bad = RISKY_HOOK.findall(result.get("hook", ""))
        if not bad:
            break
        print(f"HOOK REJECTED (attempt {attempt + 1}): {result.get('hook')!r} "
              f"-- contains {bad}")
        messages = messages + [
            {"role": "assistant", "content": response.content[0].text},
            {"role": "user", "content":
                f"That hook uses {', '.join(sorted(set(b.lower() for b in bad)))}, which makes a "
                "claim about a real person or is an insult. Rewrite it describing only what "
                "visibly happens, with neutral words. Same JSON schema."},
        ]
    else:
        raise SystemExit(
            "ERROR: could not get a hook without claims about people: "
            + str(result.get("hook") if result else "")
        )

    hook = (
        result["hook"]
        .strip()
        .upper()
    )

    words = hook.split()

    if len(words) > 8:
        raise SystemExit(
            "ERROR: hook exceeded 8 words: "
            + hook
        )

    if len(words) < 2:
        raise SystemExit(
            "ERROR: hook too short: "
            + hook
        )

    output = {
        "hook": hook,
        "word_count": len(words),
        "confidence":
            result.get(
                "confidence",
                "unknown",
            ),
        "reason":
            result.get(
                "reason",
                "",
            ),
    }

    Path(
        args.output
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json.dump(
        output,
        open(
            args.output,
            "w",
            encoding="utf-8",
        ),
        indent=2,
    )

    print()
    print(
        "===== HOOK GENERATION ====="
    )

    print(
        "Hook:",
        hook,
    )

    print(
        "Words:",
        len(words),
    )

    print(
        "Confidence:",
        output["confidence"],
    )

    print(
        "Reason:",
        output["reason"],
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
