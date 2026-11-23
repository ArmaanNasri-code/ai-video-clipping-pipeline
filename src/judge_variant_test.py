#!/usr/bin/env python3
"""
judge_variant_test.py -- try a revised judge prompt SIDE BY SIDE with the real one.

Changes NOTHING in the pipeline. visual_clip_judge.py, run folders and judgments are
only read. Results go to judge_variant_results.json (cached, so re-runs are free).

What the new prompt changes (everything else is identical: model, frames, transcript, scale):
  1. Score the BEST 15-45s stretch inside the raw ~70s window (smart trim / internal cuts
     remove the dead time before posting), instead of the whole window.
  2. A moment can land through speech (punchline, roast, argument with a winner, shocking
     statement, big reaction), not only through a visual event.
  3. For viewer_* candidates: "Twitch viewers clipped this, N views" as a HINT only.
  Still strict: inside jokes, generic banter, routine gameplay stay low.

Test sets:
  viewer    every viewer_* candidate in any *_final_review folder          (recall)
  rejected  candidates near popular viewer clips that the judge rejected
            (from _viewer_clip_check.json; skipped if the video was cleaned up)  (recall)
  eval      your labelled clips (newest eval_set_labeled*.csv), judged with BOTH the old
            and new prompt on the same input, so the comparison is fair      (precision)

    python3 judge_variant_test.py                   # all three sets
    python3 judge_variant_test.py --set eval
    python3 judge_variant_test.py --limit 5         # try a few first
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import subprocess
import tempfile

import visual_clip_judge as vcj
from anthropic import Anthropic

PASS = 7                     # same approval threshold as the pipeline
VERSION = "v1"
CACHE = "judge_variant_results.json"

NEW_PROMPT = """
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


def judge_new(video_path, transcript_path, viewer=None, old=False):
    duration = vcj.get_duration(video_path)
    transcript = vcj.load_transcript(transcript_path)
    client = Anthropic()
    with tempfile.TemporaryDirectory() as tmp:
        frames = vcj.extract_frames(video_path, duration, tmp)
        content = [{"type": "text", "text": (
            f"VIDEO DURATION: {duration:.2f} seconds\n\n"
            "The following images are sampled chronologically from the raw candidate window.")}]
        for t, path in frames:
            content.append({"type": "text", "text": f"FRAME AT {t:.2f} SECONDS"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": vcj.encode_image(path)}})
        tail = "\nWHISPER TRANSCRIPT:\n\n" + transcript
        if viewer:
            tail += (f"\n\nVIEWER EVIDENCE: Twitch viewers clipped this moment live; the most-viewed "
                     f"viewer clip has {viewer['views']} views (title: {viewer.get('title')!r}). "
                     "Hint only -- the footage must earn the score.")
        tail += ("\n\nEvaluate the complete candidate." if old
                 else "\n\nFind the best 15-45 second stretch and score it.")
        content.append({"type": "text", "text": tail})
        resp = client.messages.create(model=vcj.MODEL, max_tokens=900, temperature=0,
                                      system=vcj.SYSTEM_PROMPT if old else NEW_PROMPT,
                                      messages=[{"role": "user", "content": content}])
        try:
            res = vcj.clean_json(resp.content[0].text)
        except Exception:
            m = re.search(r"\{.*\}", resp.content[0].text, re.S)
            res = json.loads(m.group(0)) if m else {"score": 0, "reason": "unparseable reply"}
        res["duration"] = round(duration, 2)
        return res


# ---------------------------------------------------------------- finding inputs
def old_judgment(run, name):
    try:
        return json.load(open(os.path.join(run, "visual_judgments", f"{name}_visual.json")))
    except Exception:
        return {}


def find_video(run, name):
    direct = os.path.join(run, f"{name}.mp4")
    if os.path.exists(direct):
        return direct
    hits = [h for h in glob.glob(os.path.join(run, "**", f"{name}.mp4"), recursive=True)
            if "vertical" not in h and "finished" not in h]
    return hits[0] if hits else None


def item(run, name, set_name, label=None, viewer=None, fallback_video=None):
    video = find_video(run, name) or fallback_video
    tr = os.path.join(run, "visual_judgments", f"{name}_transcript.json")
    return {"set": set_name, "run": run, "name": name, "video": video,
            "transcript": tr if os.path.exists(tr) else None,
            "old": old_judgment(run, name).get("score"), "label": label, "viewer": viewer,
            "note": "" if find_video(run, name) else ("used eval copy" if fallback_video else "video cleaned up")}


def viewer_set():
    out = []
    for meta_path in sorted(glob.glob("*_final_review/_viewer_candidates.json")):
        run = os.path.dirname(meta_path)
        meta = json.load(open(meta_path))
        for name, m in sorted(meta.items()):
            out.append(item(run, name, "viewer", label=f"{m['views']}v", viewer=m))
    return out


def rejected_set():
    out, seen = [], set()
    for p in sorted(glob.glob("*_final_review/_viewer_clip_check.json")):
        run = os.path.dirname(p)
        for r in json.load(open(p)).get("rows", []):
            if r.get("verdict") == "REJECTED" and r.get("candidate") and (run, r["candidate"]) not in seen:
                seen.add((run, r["candidate"]))
                if r["candidate"].startswith("viewer_"):
                    continue          # already in the viewer set
                out.append(item(run, r["candidate"], "rejected", label=f"{r['views']}v"))
    return out


LABEL_WORDS = {"strong", "borderline", "weak", "good", "bad", "great", "ok", "maybe", "yes", "no",
               "post", "skip", "reject", "approve"}


def find_transcript(video, cid):
    stem = os.path.splitext(os.path.basename(video))[0]
    for c in [os.path.splitext(video)[0] + "_transcript.json",
              os.path.join(os.path.dirname(video), "visual_judgments", f"{stem}_transcript.json"),
              os.path.join("eval_set", "_transcripts", f"{cid}.json")]:
        if os.path.exists(c):
            return c
    hits = [] if "recut" in video else (glob.glob(f"*/visual_judgments/{stem}_transcript.json") +
                                        glob.glob(f"*/*/visual_judgments/{stem}_transcript.json"))
    if hits:
        return hits[0]
    os.makedirs(os.path.join("eval_set", "_transcripts"), exist_ok=True)
    out = os.path.join("eval_set", "_transcripts", f"{cid}.json")
    print(f"  transcribing {stem} (no transcript found) ...", flush=True)
    r = subprocess.run([sys.executable, "transcribe_groq.py", video, out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"    transcription failed: {(r.stderr or r.stdout)[-200:]}")
        return None
    return out


def eval_set():
    files = sorted(glob.glob("eval_set_labeled*.csv"), key=os.path.getmtime, reverse=True)
    path = files[0] if files else "eval_set/eval_set.csv"
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    if not rows:
        print(f"eval set: {path} is empty -- skipped")
        return []
    cols = list(rows[0].keys())
    lcol = next((c for c in cols if re.search(r"label", c, re.I)), None)
    call = next((c for c in cols if re.search(r"final_call", c, re.I)), None)
    n_lab = sum(1 for r in rows if lcol and str(r.get(lcol, "")).strip())
    print(f"eval set: {path} -- {len(rows)} rows, {n_lab} labelled (label column {lcol!r})")
    out = []
    for r in rows:
        label = str(r.get(lcol, "")).strip().lower() if lcol else ""
        if not label:
            continue
        video = next((str(r.get(k, "")).strip() for k in ("clip_path", "watch_file")
                      if str(r.get(k, "")).strip() and os.path.exists(str(r.get(k, "")).strip())), None)
        if not video and str(r.get("clip_path", "")).strip():
            vod = (r.get("vod") or "").strip() or (re.match(r"(\d{9,})", r.get("id", "")) or [None, None])[1]
            rc = os.path.join("eval_set", "recut", str(vod),
                              os.path.basename(str(r["clip_path"]).strip()))
            if os.path.exists(rc):
                video = rc
        cid = re.sub(r"[^\w.-]", "_", r.get("id") or os.path.basename(video or "x"))
        fc = str(r.get(call, "")).strip().lower() if call else ""
        lab = label + ("/" + fc if fc else "")
        try:
            csv_old = int(float(r.get("judge_score", "")))
        except Exception:
            csv_old = None
        out.append({"set": "eval", "run": "eval", "name": cid, "video": video, "transcript": None,
                    "old": csv_old, "label": lab, "viewer": None,
                    "note": "" if video else "clip file not found"})
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="viewer,rejected,eval", help="viewer, rejected, eval (comma list)")
    ap.add_argument("--limit", type=int, default=0, help="judge at most N new clips this run")
    ap.add_argument("--no-viewer-context", action="store_true", help="test viewer clips without the views hint")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set. Nothing changed.")

    sets = [s.strip() for s in args.set.split(",")]
    items = []
    if "viewer" in sets:
        items += viewer_set()
    if "rejected" in sets:
        items += rejected_set()
    if "eval" in sets:
        items += eval_set()

    cache = {}
    if os.path.exists(CACHE) and not args.force:
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}

    todo = [it for it in items if it["video"]]
    print(f"\n{len(items)} clips found, {len(todo)} with video. "
          f"(~$0.03-0.05 per new judgement; cached ones are free)\n")
    judged_now = 0
    for it in items:
        ctx = it["viewer"] if (it["viewer"] and not args.no_viewer_context) else None
        key = f"{VERSION}|{'ctx' if ctx else 'noctx'}|{it['run']}|{it['name']}"
        it["key"] = key
        if not it["video"] or key in cache:
            continue
        if args.limit and judged_now >= args.limit:
            continue
        print(f"  judging [{it['set']}] {it['name']} ...", flush=True)
        try:
            if it["set"] == "eval":
                if it["transcript"] is None:
                    it["transcript"] = find_transcript(it["video"], it["name"])
                okey = key.replace("|noctx|", "|oldprompt|")
                if okey not in cache:
                    cache[okey] = judge_new(it["video"], it["transcript"], None, old=True)
            res = judge_new(it["video"], it["transcript"], ctx)
        except Exception as e:
            msg = str(e)
            print(f"    failed: {msg[:200]}")
            if "credit" in msg.lower() or "billing" in msg.lower():
                print("    Anthropic credits look empty -- stopping. Results so far are saved.")
                break
            continue
        cache[key] = res
        judged_now += 1
        json.dump(cache, open(CACHE, "w"), indent=2)

    # ------------------------------------------------ report
    for it in items:
        o = cache.get(str(it.get("key", "")).replace("|noctx|", "|oldprompt|"), {}).get("score")
        if it["set"] == "eval" and o is not None:
            it["old"] = o

    def row(it):
        res = cache.get(it.get("key"), {})
        new = res.get("score")
        old = it["old"]
        o = "-" if old is None else str(old)
        n = "-" if new is None else str(new)
        flag = ""
        if new is not None and old is not None:
            if new >= PASS > old:
                flag = "NOW PASSES"
            elif old >= PASS > new:
                flag = "now fails"
        why = (res.get("reason") or it["note"] or "")[:150]
        span = (f"{res.get('recommended_start')}-{res.get('recommended_end')}s"
                if res.get("recommended_start") is not None else "")
        return f"  {it['name'][:34]:34s} {str(it['label'])[:16]:>16s}  old {o:>2s}  new {n:>2s}  {flag:10s} {span:9s} {why}"

    for s in ("viewer", "rejected", "eval"):
        group = [it for it in items if it["set"] == s]
        if not group:
            continue
        title = {"viewer": "VIEWER-CLIP CANDIDATES (recall: more of these should pass)",
                 "rejected": "REJECTED NEAR POPULAR VIEWER CLIPS (recall)",
                 "eval": "YOUR LABELLED EVAL SET (precision: weak ones must stay below 7)"}[s]
        print(f"\n--- {title}")
        for it in group:
            print(row(it))
        done = [it for it in group if cache.get(it.get("key"), {}).get("score") is not None]
        if s in ("viewer", "rejected") and done:
            op = sum(1 for it in done if (it["old"] or 0) >= PASS)
            np_ = sum(1 for it in done if cache[it["key"]]["score"] >= PASS)
            print(f"  passing (score >= {PASS}): old {op}/{len(done)}  ->  new {np_}/{len(done)}")
        if s == "eval" and done:
            labels = sorted({it["label"].split("/")[0] for it in done})
            for lab in labels:
                g = [it for it in done if it["label"].split("/")[0] == lab]
                olds = [it["old"] for it in g if it["old"] is not None]
                news = [cache[it["key"]]["score"] for it in g]
                op = sum(1 for v in olds if v >= PASS)
                np_ = sum(1 for v in news if v >= PASS)
                avg_o = f"{sum(olds) / len(olds):.1f}" if olds else "-"
                print(f"  {lab:10s} n={len(g):2d}  avg old {avg_o:>4s} new {sum(news) / len(news):.1f}   "
                      f"passing old {op}  new {np_}")
            bad = [it for it in done if it["label"].split("/")[0] in ("weak", "bad", "no", "skip", "reject")
                   and cache[it["key"]]["score"] >= PASS]
            if bad:
                print("  WARNING -- weak clips the new prompt would approve:")
                for it in bad:
                    print(f"    {it['name']}  new {cache[it['key']]['score']}: {cache[it['key']].get('reason', '')[:140]}")
            else:
                print("  no weak clips pass with the new prompt")
    left = sum(1 for it in items if it["video"] and it.get("key") not in cache)
    if left:
        print(f"\n{left} clip(s) not judged yet (limit/failure) -- run again to continue.")
    print(f"\nResults cached in {CACHE}. Nothing in the pipeline was changed.")


if __name__ == "__main__":
    main()
