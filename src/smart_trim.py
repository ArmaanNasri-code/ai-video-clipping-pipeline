#!/usr/bin/env python3
"""
smart_trim.py -- test-only "TikTok editor" trim pass (start/end trimming only).

For each clip it gathers, with timestamps:
  - the transcript, word by word (transcribe_groq.py, cached)
  - audio + motion energy every 0.5s
  - scene changes
  - frames every 2 seconds
and asks Claude, as an experienced TikTok editor, to identify the SETUP, the
PAYOFF and the REACTION, and to propose a start and end. Then:
  - pass 2 looks again with dense frames (every 0.5s) around the proposed
    start and end only, and confirms or adjusts each cut
  - cuts are snapped so nobody is cut off mid-sentence (start moves back to the
    beginning of the sentence it lands in, end moves forward to the end of it)
  - SAFETY: the original clip is kept when the editor is unsure, when the payoff
    or reaction would be cut, when less than 3s would be removed, or when the
    result would be shorter than 12s
Repetition / dragging moments in the middle are REPORTED (for future internal
cuts) but not cut.

Nothing is overwritten. Writes only into --out (default trim_test/):
  <clip>_trim.mp4         trimmed preview (only when a trim is applied)
  <clip>_trim.json        everything the editor decided, and why
  _summary.txt            table: original vs trim vs your --compare cuts

  python3 smart_trim.py clipA.mp4 clipB.mp4 --compare 0:06-0:52 0:05-1:06
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
COARSE_STEP = 2.0      # seconds between frames in pass 1
FINE_STEP = 0.5        # seconds between frames around each proposed cut (pass 2)
FINE_WINDOW = 3.0      # +/- seconds looked at around each cut in pass 2
VOTES = 3              # independent editor opinions per clip (frames shifted slightly each time)
SPREAD_MAX = 10.0      # if their ends/starts disagree by more than this, it's not a clear call
MIN_REMOVED = 3.0      # trims that remove less than this aren't worth it
MIN_LENGTH = 12.0      # never produce a clip shorter than this
SENTENCE_GAP = 0.6     # a pause this long between words ends a sentence
END_PAD = 0.35         # keep this much after the last word / reaction
BLUR_FRAC = 0.35       # frames this much blurrier than the clip's median = transition/graphic
TRANSCRIBE_TIMEOUT = 240  # seconds; a stuck transcription keeps the clip as is
SENSITIVE_END_S = 8.0  # look this far back from the end for lines we won't end on
ALLEGATION = re.compile(r"wom[ae]n[- ]?beat|wife[- ]?beat|"
                        r"\b(?:beats?|beating|hits?|hitting|abuses?|abusing|chokes?|choking)\s+"
                        r"(?:on\s+)?(?:a\s+|his\s+|her\s+)?(?:wom[ae]n|wife|girls?|girlfriend|kids?|children)|"
                        r"rapist|\brape|pedo|groomer|abuser|domestic violence", re.I)
SLURS = re.compile(r"nigg|fagg?|retard", re.I)
# clips never END on either; internal cuts remove only allegations (slurs are
# masked on screen by the caption renderer, and cutting them would break up action)
SENSITIVE = re.compile(ALLEGATION.pattern + "|" + SLURS.pattern, re.I)

SYSTEM = """You are an experienced TikTok editor who clips livestreams for a large
clipping account. You decide where a clip should START and END so a viewer who has
never seen the stream understands it and stays to the end.

You think in three parts:
- SETUP: the minimum context a new viewer needs (who/what/why it matters)
- PAYOFF: the moment the clip exists for (the knockdown, the reveal, the punchline)
- REACTION: the response that completes the payoff (people reacting, the aftermath)
Everything else is a candidate for removal: slow lead-in, repeated lines, reactions
that drag on after the point has landed, dead air.

Rules:
- Never cut the payoff or the core of the reaction.
- Keep the setup a new viewer needs; cut setup that only repeats or delays.
- Never start or end in the middle of someone's sentence.
- Do NOT shorten just to be shorter. If the clip already works, keep it as is.
- Loud is not the same as important: a quiet moment can be the payoff.
- The REACTION is the response to the payoff. It is OVER as soon as it starts repeating
  itself, the conversation moves to another topic, or the action stops. Later unrelated
  aftermath is not "reaction"; list it as dragging.
- A REPLAY of the payoff, or the winner/loser's immediate reaction, usually ADDS to a clip
  (viewers want to see it again) -- keep it unless it only repeats with nothing new.
- Never end on a logo card, a broadcast transition/wipe, a replay intro graphic, or a
  blurry/black frame. End on a clear human moment.
- If dragging parts run to the end of the clip, the clip should end before them.
- Do not end on (and avoid including near the end) accusations or allegations about
  real people (e.g. "he's a woman-beater"), slurs, or threats -- end before that line.
- If you are unsure, say confidence "low" -- the original clip will be kept.
Return ONLY JSON."""

PASS1_ASK = """Clip duration: {dur:.1f}s.

Return ONLY this JSON:
{{
  "setup": {{"start": <s>, "end": <s>, "what": "<...>"}},
  "payoff": {{"start": <s>, "end": <s>, "what": "<...>"}},
  "reaction": {{"start": <s>, "end": <s>, "what": "<...>"}},
  "proposed_start": <s>,
  "proposed_end": <s>,
  "keep_original": <true if the clip already works as is>,
  "confidence": "high|medium|low",
  "dragging_parts": [{{"start": <s>, "end": <s>, "why": "<repetition / reaction drags / dead air>"}}],
  "reason": "<one or two sentences>"
}}"""

PASS2_ASK = """You proposed to {which} this clip at {t:.1f}s.
Below are frames every {step}s from {a:.1f}s to {b:.1f}s, and the words spoken there.
Pick the best exact {which} point in this window: {rule}
Return ONLY JSON: {{"time": <s>, "confidence": "high|medium|low", "why": "<...>"}}"""

RULE_START = ("the first moment a new viewer needs; not mid-sentence; prefer the start of "
              "a sentence or a scene change; never later than the setup a viewer needs")
RULE_END = ("just after the reaction has landed; not mid-sentence; never before the payoff "
            "and its immediate reaction are complete")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", path]).stdout.strip())


def frame_at(video, t, path, width=320):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video,
                    "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "5", "-y", path],
                   check=True)
    return base64.b64encode(open(path, "rb").read()).decode()


def energy_curve(video, dur, step=0.5):
    n = max(1, int(dur / step))
    sr = 16000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * step)
    loud = np.array([20 * np.log10(np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) + 1e-6)
                     if a[i * hop:(i + 1) * hop].size else -120 for i in range(n)])
    w, h = 90, 160
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf",
                          f"fps={1 / step},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    fr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).astype(np.float32)
    mot = np.zeros(n)
    for i in range(1, min(n, len(fr))):
        mot[i] = np.mean(np.abs(fr[i] - fr[i - 1]))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
    e = 0.6 * z(loud) + 0.4 * z(mot)
    e = np.convolve(e, np.ones(4) / 4, mode="same")
    cuts = [round(i * step, 1) for i in range(1, len(mot))
            if mot[i] > np.median(mot) * 4 and mot[i] > 25]
    return e, cuts


def transition_times(video, dur, step=0.5):
    """Times (s) that look like a wipe / logo card / blur: very low sharpness, or a
    nearly flat frame. Used so a clip never ends on one."""
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vals, t = [], 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (160, 284)), cv2.COLOR_BGR2GRAY)
        vals.append((t, float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.std())))
        t += step
    cap.release()
    if not vals:
        return []
    med = float(np.median([v[1] for v in vals]))
    return [t for t, sh, sd in vals if sh < BLUR_FRAC * med or sd < 12]


def get_words(video, cache):
    if not os.path.exists(cache):
        try:
            r = subprocess.run([sys.executable, "transcribe_groq.py", video, cache],
                               capture_output=True, text=True, timeout=TRANSCRIBE_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"transcription took over {TRANSCRIBE_TIMEOUT}s")
        if r.returncode != 0 or not os.path.exists(cache):
            # without word timings we can't guarantee clean sentence cuts -> keep original
            raise RuntimeError("transcription failed: " + r.stderr[-300:])
    d = json.load(open(cache))
    words = []
    for w in d.get("words", []) or []:
        txt = (w.get("word") or w.get("text") or "").strip()
        if txt:
            words.append((float(w["start"]), float(w["end"]), txt))
    segs = [(float(s["start"]), float(s["end"]), s.get("text", "").strip())
            for s in d.get("segments", [])]
    if not words:   # fall back to segments as "words"
        words = segs
    return words, segs


def sentences(words):
    out, cur = [], []
    for w in words:
        if cur and (w[0] - cur[-1][1] > SENTENCE_GAP or re.search(r"[.!?]$", cur[-1][2])):
            out.append((cur[0][0], cur[-1][1]))
            cur = []
        cur.append(w)
    if cur:
        out.append((cur[0][0], cur[-1][1]))
    return out


def snap_start(t, sents):
    for a, b in sents:
        if a < t < b:
            return max(0.0, a - 0.1)
    return t


def snap_end(t, sents, dur):
    for a, b in sents:
        if a < t < b:
            return min(dur, b + END_PAD)
    return min(dur, t)


def ask(client, content, max_tokens=900):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, temperature=0,
                               system=SYSTEM, messages=[{"role": "user", "content": content}])
    txt = r.content[0].text.strip()
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0) if m else txt)


def fmt(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def parse_cut(s):
    def sec(x):
        p = x.strip().split(":")
        return float(p[0]) * 60 + float(p[1]) if len(p) == 2 else float(p[0])
    a, b = s.split("-")
    return sec(a), sec(b)


def pass1(client, video, name, dur, segs, energy, cuts, trans, tmp, offset=0.5):
    """One editor opinion on the whole clip (frames every COARSE_STEP from offset)."""
    content = [{"type": "text", "text": f"CLIP: {name}\nDURATION: {dur:.1f}s\n"}]
    t, k = offset, 0
    while t < dur:
        content.append({"type": "text", "text": f"FRAME {t:.1f}s"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                        "data": frame_at(video, t, os.path.join(tmp, f"c{k}.jpg"))}})
        t += COARSE_STEP
        k += 1
    tr = "\n".join(f"[{a:.1f}-{b:.1f}] {x}" for a, b, x in segs) or "(no speech)"
    en = " ".join(f"{i * 2:.0f}s:{energy[int(i * 4)]:+.1f}" for i in range(int(len(energy) / 4)))
    content.append({"type": "text", "text":
                    f"\nTRANSCRIPT (timestamped):\n{tr}\n\nENERGY every 2s (loudness+motion, "
                    f"z-score; loud is NOT the same as important):\n{en}\n\nSCENE CHANGES at: "
                    f"{', '.join(str(c) for c in cuts) or 'none detected'}\n\nTRANSITION / LOGO / BLUR "
                    f"frames at: {', '.join(f'{x:.1f}' for x in trans) or 'none detected'}\n\n"
                    + PASS1_ASK.format(dur=dur)})
    p = ask(client, content)
    start = max(0.0, float(p.get("proposed_start", 0)))
    end = min(dur, float(p.get("proposed_end", dur)))
    rea = p.get("reaction") or {}
    for d in p.get("dragging_parts") or []:           # drags running to the end pull it in
        try:
            d0, d1 = float(d["start"]), float(d["end"])
        except Exception:
            continue
        if d1 >= end - 2.0 and d0 < end and d0 > start + MIN_LENGTH:
            if rea.get("start") is None or d0 >= float(rea["start"]) + 2.0:
                end = d0
    p["proposed_start"], p["proposed_end"] = start, end
    return p


def combine_votes(votes, dur, res):
    """Combine several editor opinions. Clear agreement -> the median cut.
    Disagreement -> the LONGEST of them (keep more when unsure). Most say keep /
    low confidence -> None (keep the original)."""
    if sum(1 for v in votes if v.get("keep_original")) * 2 > len(votes):
        res["vote_problem"] = "most editor opinions say the clip already works as is"
        return None
    if sum(1 for v in votes if v.get("confidence") == "low") * 2 > len(votes):
        res["vote_problem"] = "most editor opinions are low confidence"
        return None
    starts = sorted(float(v["proposed_start"]) for v in votes)
    ends = sorted(float(v["proposed_end"]) for v in votes)
    res["vote_starts"], res["vote_ends"] = starts, ends
    if ends[-1] - ends[0] > SPREAD_MAX or starts[-1] - starts[0] > SPREAD_MAX:
        start, end = starts[0], ends[-1]
        res["vote_agreement"] = "disagree -> kept the longest version"
    else:
        start, end = starts[len(starts) // 2], ends[len(ends) // 2]
        res["vote_agreement"] = "agree -> median"
    base = dict(votes[len(votes) // 2])

    def span(key):
        xs = [v.get(key) or {} for v in votes]
        st = [float(x["start"]) for x in xs if x.get("start") is not None]
        en = [float(x["end"]) for x in xs if x.get("end") is not None]
        if not st:
            return base.get(key)
        return {"start": min(st), "end": max(en) if en else min(st),
                "what": (base.get(key) or {}).get("what", "")}
    base["payoff"], base["reaction"] = span("payoff"), span("reaction")
    base["proposed_start"], base["proposed_end"] = start, end
    base["confidence"] = "medium" if "disagree" in res["vote_agreement"] else base.get("confidence", "medium")
    return base


def trim_one(client, video, out_dir, tmp):
    name = os.path.splitext(os.path.basename(video))[0]
    dur = duration(video)
    words, segs = get_words(video, os.path.join(out_dir, f"{name}_words.json"))
    sents = sentences(words)
    energy, cuts = energy_curve(video, dur)
    trans = transition_times(video, dur)

    # ---------- pass 1: whole clip, frame every 2s ----------
    votes = []
    for vote in range(VOTES):
        votes.append(pass1(client, video, name, dur, segs, energy, cuts, trans, tmp,
                           offset=0.5 + vote * COARSE_STEP / VOTES))
    res = {"clip": name, "duration": round(dur, 2), "votes": votes}
    p1 = combine_votes(votes, dur, res)
    res["pass1"] = p1
    if p1 is None:
        res.update(final_start=0.0, final_end=round(dur, 2), applied=False,
                   kept_original_because=res.get("vote_problem", "editor opinions disagree"))
        json.dump(res, open(os.path.join(out_dir, f"{name}_trim.json"), "w"), indent=2)
        return res

    start, end = float(p1.get("proposed_start", 0)), float(p1.get("proposed_end", dur))
    start, end = max(0.0, start), min(dur, end)

    # ---------- pass 2: dense frames around each cut ----------
    for which, t0, rule in (("start", start, RULE_START), ("end", end, RULE_END)):
        if (which == "start" and t0 < 1.0) or (which == "end" and t0 > dur - 1.0):
            continue
        a, b = max(0.0, t0 - FINE_WINDOW), min(dur, t0 + FINE_WINDOW)
        c2 = []
        t = a
        k = 0
        while t <= b:
            c2.append({"type": "text", "text": f"FRAME {t:.1f}s"})
            c2.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": frame_at(video, t, os.path.join(tmp, f"{which}{k}.jpg"))}})
            t += FINE_STEP
            k += 1
        near = "\n".join(f"[{x:.2f}-{y:.2f}] {w}" for x, y, w in words if y >= a - 1 and x <= b + 1)
        c2.append({"type": "text", "text":
                   f"PAYOFF: {p1.get('payoff')}\nREACTION: {p1.get('reaction')}\n"
                   f"WORDS:\n{near or '(none)'}\n\n" +
                   PASS2_ASK.format(which=which, t=t0, step=FINE_STEP, a=a, b=b, rule=rule)})
        p2 = ask(client, c2, max_tokens=300)
        res[f"pass2_{which}"] = p2
        if p2.get("confidence") != "low":
            v = float(p2.get("time", t0))
            v = max(t0 - FINE_WINDOW, min(t0 + FINE_WINDOW, v))   # stay inside the window looked at
            if which == "start":
                start = max(0.0, v)
            else:
                end = min(dur, v)

    # ---------- snap to sentences ----------
    start, end = snap_start(start, sents), snap_end(end, sents, dur)
    # don't end inside a transition / logo card: back up to just before it
    tset = sorted(x for x in trans if start + MIN_LENGTH < x <= end)
    if tset:
        run0 = tset[-1]
        for x in reversed(tset[:-1]):
            if run0 - x <= 0.6:
                run0 = x
            else:
                break
        if end - tset[-1] < 2.0:          # the clip ends in/right after a transition
            end = run0 - 0.1
            for a, b in sents:            # don't stop mid-sentence: end before that sentence
                if a < end < b:
                    end = max(start + MIN_LENGTH, a - 0.05)
                    break
            res["ended_before_transition_at"] = round(run0, 2)
    # never end on an allegation / slur: end before the sentence that contains it
    for a, b in reversed(sents):
        if b <= end - SENSITIVE_END_S:
            break
        if a >= end:
            continue
        line = " ".join(w for x, y, w in words if x >= a - 0.01 and y <= b + 0.01)
        if SENSITIVE.search(line):
            new_end = max(start + MIN_LENGTH, a - 0.05)
            res["ended_before_sensitive_line"] = line[:120]
            end = min(end, new_end)
    # the original clip may end mid-sentence: end on the last complete one instead
    if end >= dur - 0.3 and sents and sents[-1][1] >= dur - 0.3 and len(sents) > 1:
        end = min(end, sents[-2][1] + END_PAD)
        res["dropped_cut_off_last_sentence"] = True

    # ---------- safety checks ----------
    why_not = None
    pay = p1.get("payoff") or {}
    rea = p1.get("reaction") or {}
    if p1.get("keep_original"):
        why_not = "editor says the clip already works as is"
    elif p1.get("confidence") == "low":
        why_not = "editor not confident"
    elif pay.get("start") is not None and not (start <= float(pay["start"]) and float(pay.get("end", pay["start"])) <= end):
        why_not = "trim would cut into the payoff"
    elif rea.get("start") is not None and end < float(rea["start"]) + min(
            2.0, max(0.0, float(rea.get("end", rea["start"])) - float(rea["start"]))):
        why_not = "trim would cut the reaction too early"
    elif (start + (dur - end)) < MIN_REMOVED:
        why_not = f"would remove under {MIN_REMOVED:.0f}s"
    elif end - start < MIN_LENGTH:
        why_not = f"result shorter than {MIN_LENGTH:.0f}s"

    res["final_start"], res["final_end"] = round(start, 2), round(end, 2)
    res["applied"] = why_not is None
    res["kept_original_because"] = why_not
    if why_not is None:
        outp = os.path.join(out_dir, f"{name}_trim.mp4")
        if not os.path.exists(outp):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                            "-i", video, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", outp], check=True)
        res["preview"] = outp
    json.dump(res, open(os.path.join(out_dir, f"{name}_trim.json"), "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="trim_test")
    ap.add_argument("--compare", nargs="*", default=[], help="your cuts, same order, e.g. 0:06-0:52")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")
    if not os.path.exists("transcribe_groq.py"):
        sys.exit("ERROR: transcribe_groq.py not found -- run from ~/Downloads")
    os.makedirs(args.out, exist_ok=True)
    client = Anthropic()
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, v in enumerate(args.videos):
            print(f"\n== {os.path.basename(v)}")
            try:
                r = trim_one(client, v, args.out, tmp)
            except Exception as e:
                print(f"   FAILED: {e!r} -- original kept")
                lines.append(f"{os.path.basename(v)[:40]:40s} FAILED ({e.__class__.__name__}) -> original kept")
                continue
            p1 = r["pass1"] or {}
            if r.get("vote_ends"):
                print(f"   editor votes: starts {r['vote_starts']} ends {r['vote_ends']} "
                      f"-> {r.get('vote_agreement')}")
            print(f"   setup:    {p1.get('setup')}")
            print(f"   payoff:   {p1.get('payoff')}")
            print(f"   reaction: {p1.get('reaction')}")
            if p1.get("dragging_parts"):
                print(f"   drags (not cut yet): {p1.get('dragging_parts')}")
            cut = f"{fmt(r['final_start'])}-{fmt(r['final_end'])}"
            verdict = (f"TRIM {cut} ({r['final_end'] - r['final_start']:.0f}s of {r['duration']:.0f}s)"
                       if r["applied"] else f"KEEP ORIGINAL ({r['kept_original_because']}; proposed {cut})")
            yours = ""
            if i < len(args.compare):
                a, b = parse_cut(args.compare[i])
                yours = (f" | yours {fmt(a)}-{fmt(b)}: start off by {r['final_start'] - a:+.1f}s, "
                         f"end off by {r['final_end'] - b:+.1f}s")
            print("   =>", verdict + yours)
            lines.append(f"{r['clip'][:40]:40s} {verdict}{yours}")
    open(os.path.join(args.out, "_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "=" * 78)
    print("\n".join(lines))
    print(f"\npreviews and decisions in {args.out}/ (originals untouched)")


if __name__ == "__main__":
    main()
