#!/usr/bin/env python3

import argparse
import json
import os
import re
from difflib import SequenceMatcher

from anthropic import Anthropic


MODEL = "claude-sonnet-4-5"


# ---------------- streamer name list (spelling reference) ----------------
def _load_names():
    names, fix = [], {}
    path = os.environ.get("CLIP_NAMES_FILE", "").strip()
    if path and os.path.exists(path):
        try:
            d = json.load(open(path))
            names += [str(n) for n in d.get("names", [])]
            fix = {str(k): str(v) for k, v in d.get("fix", {}).items()}
        except Exception as e:
            print("WARNING: could not read name list", path, repr(e))
    extra = os.environ.get("CLIP_EXTRA_NAMES", "")
    names += [n.strip() for n in extra.split(",") if n.strip()]
    names += list(fix.values())
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out, fix


NAMES, NAME_FIX = _load_names()
NAME_TOKENS = {
    t for n in NAMES for t in re.findall(r"[a-z0-9']+", n.lower()) if len(t) >= 3
}
if NAMES:
    print("Name list:", ", ".join(NAMES))


def _canon(t):
    return t.replace("3", "e").replace("0", "o").replace("1", "i")


def name_supported(word, candidate_texts):
    """A listed name counts as supported if some candidate heard a similar
    word (same first letter, close spelling), e.g. 'adin' <- 'aiden'."""
    if word not in NAME_TOKENS:
        return False
    w = _canon(word)
    for cand in candidate_texts:
        for t in re.findall(r"[a-z0-9']+", (cand or "").lower()):
            c = _canon(t)
            if len(c) >= 3 and c[0] == w[0] and SequenceMatcher(None, c, w).ratio() >= 0.6:
                return True
    return False


def names_prompt():
    if not NAMES:
        return ""
    return (
        "\nKNOWN NAMES (spelling reference only): " + ", ".join(NAMES) + "\n"
        "- If a candidate clearly heard a word that SOUNDS like one of these names "
        "(e.g. 'Aiden' for 'Adin'), write the listed spelling.\n"
        "- Never insert a name that no candidate heard, and never replace an ordinary "
        "word with a name just because it is listed.\n"
    )


def apply_name_fix(text):
    for wrong, right in NAME_FIX.items():
        text = re.sub(r"\b" + re.escape(wrong) + r"\b", right, text, flags=re.IGNORECASE)
    return text

SYSTEM = """
You resolve noisy automatic speech recognition transcripts.

You are given multiple ASR versions of the SAME audio region.

Rules:
1. Produce the most likely spoken wording using ONLY evidence from the supplied ASR candidates.
2. You may fix obvious phonetic ASR mistakes when another candidate strongly supports the correction.
3. Never add facts, dialogue, names, or phrases unsupported by the candidates.
4. Preserve profanity exactly when supported.
5. Preserve repetitions when they sound intentional, but remove obvious ASR-loop hallucinations.
6. If the wording cannot be resolved confidently, mark confidence "low".
7. Return valid JSON only.

Schema:
{
  "resolved_text": "...",
  "confidence": "high|medium|low",
  "reason": "brief explanation"
}
"""


def clean_json(text):
    text = (text or "").strip()

    if not text:
        raise ValueError(
            "Empty Claude response"
        )

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

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and
            end != -1
            and
            end > start
        ):
            return json.loads(
                text[start:end + 1]
            )

        raise


def salvage_fields(text):
    """Pull resolved_text/confidence out of a reply whose JSON was cut off."""
    if not text:
        return None
    m = re.search(r'"resolved_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    c = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', text)
    if not (m and c):
        return None
    try:
        resolved = json.loads('"' + m.group(1) + '"')
    except Exception:
        resolved = m.group(1)
    conf = c.group(1)
    if conf == "high":
        conf = "medium"
    return {
        "resolved_text": resolved,
        "confidence": conf,
        "reason": "salvaged from a cut-off Claude reply",
    }


def overlap(a0, a1, b0, b1):
    return max(
        0.0,
        min(a1, b1)
        - max(a0, b0),
    )


def normalized_text(s):
    return " ".join(
        re.findall(
            r"[a-z0-9']+",
            s.lower(),
        )
    )


def direct_support_count(
    resolved_text,
    candidate_texts,
):
    target = normalized_text(
        resolved_text
    )

    if not target:
        return 0

    count = 0

    for candidate in candidate_texts:
        candidate_norm = normalized_text(
            candidate or ""
        )

        if target in candidate_norm:
            count += 1

    return count


def unsupported_words(
    resolved_text,
    candidate_texts,
):
    def words(s):
        return re.findall(
            r"[a-z0-9']+",
            s.lower(),
        )

    supported = set()

    for candidate in candidate_texts:
        supported.update(
            words(candidate or "")
        )

    ignore = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "to",
        "of",
        "in",
        "on",
        "at",
    }

    bad = []

    for word in words(
        resolved_text
    ):
        if (
            word not in supported
            and
            word not in ignore
            and
            len(word) >= 4
        ):
            bad.append(word)

    return sorted(
        set(bad)
    )


def suspicious_text(text):
    t = text.lower().strip()

    words = t.split()

    if len(words) >= 8:
        unique_ratio = (
            len(set(words))
            / max(1, len(words))
        )

        if unique_ratio < 0.45:
            return True

    return False


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--semantic",
        required=True,
    )

    ap.add_argument(
        "--timed",
        required=True,
    )

    ap.add_argument(
        "--consensus",
        required=True,
    )

    ap.add_argument(
        "--third-pass",
        required=True,
    )

    ap.add_argument(
        "--third-pass-2",
        required=False,
        default=None,
    )

    ap.add_argument(
        "--recovery-evidence",
        required=False,
        default=None,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    if not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        raise SystemExit(
            "ERROR: ANTHROPIC_API_KEY not set"
        )

    semantic = json.load(
        open(args.semantic)
    )["segments"]

    timed = json.load(
        open(args.timed)
    )["segments"]

    consensus = json.load(
        open(args.consensus)
    )

    third = json.load(
        open(args.third_pass)
    )

    third_2 = []

    if args.third_pass_2:
        third_2 = json.load(
            open(args.third_pass_2)
        )

    recovery_regions = []

    if args.recovery_evidence:
        recovery_data = json.load(
            open(args.recovery_evidence)
        )

        recovery_regions = (
            recovery_data.get(
                "regions",
                [],
            )
        )

    client = Anthropic()

    resolved = []

    for region in consensus:

        start = float(
            region["start"]
        )

        end = float(
            region["end"]
        )

        semantic_text = (
            region["semantic_text"]
        )

        timed_parts = []

        for seg in timed:

            if overlap(
                start,
                end,
                float(seg["start"]),
                float(seg["end"]),
            ) > 0:

                text = (
                    seg.get(
                        "text",
                        "",
                    ).strip()
                )

                # Remove known boundary hallucination.
                if text.lower() in {
                    "thank you.",
                    "thank you",
                }:
                    continue

                timed_parts.append(
                    text
                )

        timed_text = " ".join(
            timed_parts
        ).strip()

        third_parts = []

        THIRD_MATCH_EPS = 0.02

        for window in third:

            w0 = float(
                window["window_start"]
            )

            w1 = float(
                window["window_end"]
            )

            if (
                abs(start - w0)
                <= THIRD_MATCH_EPS
                and
                abs(end - w1)
                <= THIRD_MATCH_EPS
            ):
                third_parts.append(
                    window[
                        "third_pass_text"
                    ]
                )

        third_text = " ".join(
            third_parts
        ).strip()

        third_2_parts = []

        for window in third_2:

            w0 = float(
                window["window_start"]
            )

            w1 = float(
                window["window_end"]
            )

            if (
                abs(start - w0)
                <= THIRD_MATCH_EPS
                and
                abs(end - w1)
                <= THIRD_MATCH_EPS
            ):
                third_2_parts.append(
                    window[
                        "third_pass_text"
                    ]
                )

        third_2_text = " ".join(
            third_2_parts
        ).strip()

        recovery_aligned_items = []
        recovery_single_items = []
        recovery_support_phrases = []

        for recovery in recovery_regions:
            for item in recovery.get(
                "aligned_support",
                [],
            ):
                i0 = float(
                    item.get(
                        "start",
                        -1,
                    )
                )
                i1 = float(
                    item.get(
                        "end",
                        -1,
                    )
                )

                if overlap(
                    start,
                    end,
                    i0,
                    i1,
                ) <= 0:
                    continue

                turbo_text = (
                    item.get(
                        "turbo_text",
                        "",
                    ).strip()
                )

                full_text = (
                    item.get(
                        "full_v3_text",
                        "",
                    ).strip()
                )

                similarity = float(
                    item.get(
                        "similarity",
                        0.0,
                    )
                )

                recovery_aligned_items.append({
                    "start": i0,
                    "end": i1,
                    "turbo_text": turbo_text,
                    "full_v3_text": full_text,
                    "similarity": similarity,
                })

                recovery_support_phrases.append(
                    (
                        f"{i0:.2f}-{i1:.2f}: "
                        f"turbo={turbo_text!r}; "
                        f"full_v3={full_text!r}; "
                        f"sim={similarity:.3f}"
                    )
                )

            for item in recovery.get(
                "single_source_support",
                [],
            ):
                i0 = float(
                    item.get(
                        "start",
                        -1,
                    )
                )
                i1 = float(
                    item.get(
                        "end",
                        -1,
                    )
                )

                if overlap(
                    start,
                    end,
                    i0,
                    i1,
                ) <= 0:
                    continue

                source = (
                    item.get(
                        "source",
                        "unknown",
                    )
                )

                item_text = (
                    item.get(
                        "text",
                        "",
                    ).strip()
                )

                recovery_single_items.append({
                    "start": i0,
                    "end": i1,
                    "source": source,
                    "text": item_text,
                })

        # --------------------------------------------------
        # Stable recovery tier.
        #
        # These phrases survived repeated shifted-crop
        # Whisper checks. They still must NOT affect
        # confidence scoring or automatic approval.
        # --------------------------------------------------

        recovery_stable_items = []

        for recovery in recovery_regions:
            for item in recovery.get(
                "stable_support",
                [],
            ):
                i0 = float(
                    item.get(
                        "start",
                        -1,
                    )
                )

                i1 = float(
                    item.get(
                        "end",
                        -1,
                    )
                )

                if overlap(
                    start,
                    end,
                    i0,
                    i1,
                ) <= 0:
                    continue

                item_text = (
                    item.get(
                        "text",
                        "",
                    ).strip()
                )

                if not item_text:
                    continue

                recovery_stable_items.append({
                    "start": i0,
                    "end": i1,
                    "text": item_text,
                    "evidence_type":
                        item.get(
                            "evidence_type",
                            "stable_recovery",
                        ),
                    "confidence":
                        item.get(
                            "confidence",
                            "strong_correlated",
                        ),
                })

        recovery_stable_items.sort(
            key=lambda x: (
                x["start"],
                x["end"],
            )
        )

        recovery_stable_text = "\\n".join(
            (
                f"{x['start']:.2f}-{x['end']:.2f} "
                f"[{x['evidence_type']}]: "
                f"{x['text']}"
            )
            for x in recovery_stable_items
        )

        # IMPORTANT:
        # Time-aligned recovery is support-only.
        # It must never participate in automatic
        # approval or unsupported-word promotion.
        recovery_text = ""

        recovery_aligned_text = "\\n".join(
            (
                f"{x['start']:.2f}-{x['end']:.2f} "
                f"sim={x['similarity']:.3f}\\n"
                f"  turbo: {x['turbo_text']}\\n"
                f"  full-v3: {x['full_v3_text']}"
            )
            for x in recovery_aligned_items
        )

        recovery_single_text = "\\n".join(
            (
                f"{x['start']:.2f}-{x['end']:.2f} "
                f"[{x['source']}]: {x['text']}"
            )
            for x in recovery_single_items
        )

        recovery_support_text = " | ".join(
            recovery_support_phrases
        )

        auto_novel_words = unsupported_words(
            semantic_text,
            [
                timed_text,
                third_text,
                third_2_text,
            ],
        )

        semantic_timed_similarity = (
            SequenceMatcher(
                None,
                normalized_text(
                    semantic_text
                ),
                normalized_text(
                    timed_text
                ),
            ).ratio()
            if timed_text
            else 0.0
        )

        # If transcripts already strongly agree,
        # don't spend a Claude call.
        #
        # The semantic transcript must also avoid
        # substantial words unsupported by the
        # independent ASR evidence.
        if (
            not region[
                "needs_review"
            ]
            and
            not suspicious_text(
                semantic_text
            )
            and
            (
                not auto_novel_words
                or
                semantic_timed_similarity >= 0.82
            )
        ):

            resolved.append({
                "start": start,
                "end": end,
                "resolved_text":
                    semantic_text,
                "confidence": "high",
                "status": "approved",
                "source":
                    "automatic_agreement",
            })

            continue

        prompt = f"""
Audio region:
{start:.2f}s -> {end:.2f}s

ASR A — semantic/full clip:
{semantic_text}

ASR B — timed/chunked:
{timed_text}

ASR C — selective third pass:
{third_text or "[none]"}

ASR D — independent second third pass:
{third_2_text or "[none]"}

VAD-triggered recovery evidence — TIME-ALIGNED SUPPORT ONLY:

Aligned turbo/full-v3 evidence:
{recovery_aligned_text or "[none]"}

Single-source recovery clues:
{recovery_single_text or "[none]"}

Repeated shifted-crop stable recovery:
{recovery_stable_text or "[none]"}

Resolve the most likely spoken wording.

STRICT RESOLUTION RULES:
- Recovery evidence is SUPPORT ONLY. Never treat it as automatic truth.
- Turbo and full-v3 are correlated Whisper-family evidence, not independent confirmation.
- Use recovery wording only for the exact timestamped subspan shown.
- Do NOT extend recovered wording outside its timestamped subspan.
- Do NOT use single-source recovery to invent or replace uncertain wording.
- If turbo and full-v3 differ, preserve only wording clearly supported by the other ASRs or return lower confidence.
- Do NOT combine unrelated fragments from different candidates into a new sentence.
- Do NOT invent connecting words to make bad ASR sound grammatical.
- Prefer wording that appears directly in at least one candidate.
- When two candidates share a clear phrase, preserve that shared phrase.
- If the ending is uncertain, return only the portion that is strongly supported rather than guessing.
- A shorter accurate caption is better than a longer invented caption.
- If no clean wording is strongly supported, use confidence "low".
"""
        prompt += names_prompt()

        response = (
            client.messages.create(
                model=MODEL,
                max_tokens=400,
                temperature=0,
                system=SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        )

        raw_response = (
            response.content[0].text
            if response.content
            else ""
        )

        try:
            result = clean_json(
                raw_response
            )

        except Exception as first_err:
            # Usual cause: analysis written before the JSON, cut off by
            # max_tokens. Ask once more for the JSON only, with more room.
            print()
            print(
                "CLAUDE PARSE RETRY:",
                f"{start:.2f} -> {end:.2f}",
                repr(first_err),
            )
            result = None
            retry_raw = ""
            try:
                retry = client.messages.create(
                    model=MODEL,
                    max_tokens=1000,
                    temperature=0,
                    system=SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                            + "\n\nIMPORTANT: reply with ONLY the JSON object "
                              "described above. No analysis, no markdown, "
                              "nothing before or after the JSON.",
                        }
                    ],
                )
                retry_raw = (
                    retry.content[0].text
                    if retry.content
                    else ""
                )
                result = clean_json(retry_raw)
                print("  retry parsed OK")
            except Exception as e:
                result = (
                    salvage_fields(retry_raw)
                    or salvage_fields(raw_response)
                )
                if result is not None:
                    print("  salvaged answer from the cut-off reply")
                else:
                    print()
                    print(
                        "CLAUDE PARSE FAILURE:",
                        f"{start:.2f} -> {end:.2f}",
                    )
                    print(
                        "error:",
                        repr(e),
                    )
                    print(
                        "raw:",
                        repr(retry_raw or raw_response),
                    )

                    result = {
                        "resolved_text":
                            semantic_text,
                        "confidence":
                            "low",
                        "reason":
                            "Claude response could not be parsed.",
                    }

        confidence = result["confidence"]

        novel_words = unsupported_words(
            result["resolved_text"],
            [
                semantic_text,
                timed_text,
                third_text,
                third_2_text,
                recovery_text,
            ],
        )

        novel_words = [
            w for w in novel_words
            if not name_supported(
                w,
                [semantic_text, timed_text, third_text, third_2_text],
            )
        ]

        if novel_words:
            confidence = "low"

            result["reason"] = (
                result.get(
                    "reason",
                    "",
                )
                + " Unsupported generated words: "
                + ", ".join(
                    novel_words
                )
            ).strip()

        elif confidence == "medium":
            support_count = direct_support_count(
                result["resolved_text"],
                [
                    semantic_text,
                    timed_text,
                    third_text,
                    third_2_text,
                ],
            )

            word_count = len(
                normalized_text(
                    result["resolved_text"]
                ).split()
            )

            candidate_word_counts = [
                len(
                    normalized_text(
                        candidate or ""
                    ).split()
                )
                for candidate in [
                    semantic_text,
                    timed_text,
                    third_text,
                    third_2_text,
                ]
                if candidate
            ]

            max_candidate_words = max(
                candidate_word_counts,
                default=0,
            )

            region_duration = (
                end - start
            )

            compressed_long_region = (
                region_duration >= 6.0
                and
                max_candidate_words
                >= max(
                    12,
                    word_count * 2,
                )
            )

            if (
                support_count >= 2
                and
                word_count <= 8
                and
                not compressed_long_region
            ):
                confidence = "high"

                result["reason"] = (
                    result.get(
                        "reason",
                        "",
                    )
                    + " Promoted by direct agreement "
                    + f"across {support_count} ASR candidates."
                ).strip()

        status = (
            "approved"
            if confidence == "high"
            else "review_required"
        )

        resolved.append({
            "start": start,
            "end": end,
            "resolved_text":
                result[
                    "resolved_text"
                ],
            "confidence":
                confidence,
            "status":
                status,
            "reason":
                result.get(
                    "reason",
                    "",
                ),
            "source":
                "claude_resolution",
            "recovery_text":
                recovery_text,
            "recovery_support":
                recovery_support_phrases,
            "stable_recovery":
                recovery_stable_items,
            "stable_recovery_policy":
                "support_only",
        })

        print()
        print(
            f"{start:.2f}s -> "
            f"{end:.2f}s"
        )
        print(
            "A:",
            semantic_text,
        )
        print(
            "B:",
            timed_text,
        )
        print(
            "C:",
            third_text,
        )
        print(
            "D:",
            third_2_text,
        )
        print(
            "=>",
            result[
                "resolved_text"
            ],
        )
        print(
            "confidence:",
            result[
                "confidence"
            ],
        )

    fixed = 0
    for x in resolved:
        new_text = apply_name_fix(x.get("resolved_text", ""))
        if new_text != x.get("resolved_text", ""):
            x["resolved_text"] = new_text
            x["name_fix_applied"] = True
            fixed += 1
    if NAME_FIX:
        print("Name fixes applied to", fixed, "regions")

    json.dump(
        resolved,
        open(
            args.output,
            "w",
        ),
        indent=2,
    )

    review = [
        x
        for x in resolved
        if x.get("status")
        == "review_required"
    ]

    print()
    print(
        "===== CAPTION RESOLUTION COMPLETE ====="
    )
    print(
        "Regions:",
        len(resolved),
    )
    print(
        "Review required:",
        len(review),
    )
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
