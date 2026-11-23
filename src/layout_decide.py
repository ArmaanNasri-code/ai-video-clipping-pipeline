#!/usr/bin/env python3
"""
layout_decide.py -- AUTO layout choice (split-screen vs full-screen) for finished clips.

Standalone test tool: reads finished clips, changes nothing in the pipeline.

For each clip:
  1. Claude looks at 6 frames from the FINAL clip's timeline, plus the visual judge's
     existing analysis and the captions, and answers: what does the viewer need to see?
       PERSON_FOCUSED -- talking, storytelling, facecam reactions      -> split candidate
       ACTION_FOCUSED -- fights, sports, IRL chaos, chases             -> full-screen
       SCREEN_FOCUSED -- chess, gameplay challenges, on-screen detail  -> full-screen
  2. Supporting signals / safety checks (not the main decision):
       on-camera share (can the split crop keep the person in frame?), speech share,
       motion, crop mode. Crop mode is NOT an automatic rejection.
  3. Uncertain (confidence < 0.7) -> full-screen.
Decisions + reasons are saved to split_previews/_layout_decisions.json (re-runs reuse
them; --force re-decides). Manual override per clip: --override split|full.

    python3 layout_decide.py *_final_review/production_ready/*_final.mp4
    python3 layout_decide.py <clip_final.mp4> --force
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile

import split_preview as sp

MODEL = "claude-sonnet-4-6"
MIN_CONFIDENCE = 0.7
MIN_ON_CAMERA = 0.6      # below this the split crop can't keep the person in frame
REPORT = os.path.join("split_previews", "_layout_decisions.json")

SYSTEM = """You help a TikTok clipping tool choose a layout for a finished livestream clip.

Option A, SPLIT-SCREEN: the top 45% shows a crop that follows the streamer; the bottom
55% shows unrelated muted Minecraft parkour to hold attention. Anything on the stream
that is not near the streamer is cut off or small.
Option B, FULL-SCREEN: the normal vertical crop of the stream, no gameplay added.

Classify what the VIEWER NEEDS TO SEE to understand and enjoy the moment:
- PERSON_FOCUSED: the person is the content -- talking, storytelling, reacting on camera,
  conversation, trying things on. Losing the rest of the screen costs nothing.
- ACTION_FOCUSED: physical action is the content -- fights, sports, IRL chaos, chases,
  crowds, stunts, events where where-things-are matters.
- SCREEN_FOCUSED: what is on the screen is the content -- chess, gameplay plays/fails,
  menus or numbers that matter, video being reacted to, on-screen text.

Be conservative: if the person AND the screen both matter, it is SCREEN_FOCUSED.
Return JSON only:
{"category": "PERSON_FOCUSED|ACTION_FOCUSED|SCREEN_FOCUSED",
 "confidence": 0.0-1.0,
 "what_viewer_watches": "<one short sentence>",
 "would_split_lose_something": true/false,
 "reason": "<one or two sentences>"}"""


def frames_b64(src, parts, n=6):
    total = sum(b - a for a, b in parts) if parts else sp.probe(src)[3]
    spans = parts or [(0.0, total)]
    out = []
    with tempfile.TemporaryDirectory() as td:
        for k in range(n):
            t = (k + 0.5) * total / n
            for a, b in spans:          # map final-clip time -> source time
                if t <= b - a:
                    st = a + t
                    break
                t -= b - a
            else:
                st = spans[-1][1] - 0.1
            fp = os.path.join(td, f"{k}.jpg")
            subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-ss", f"{st:.2f}", "-i", src,
                            "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", fp],
                           stdin=subprocess.DEVNULL)
            if os.path.exists(fp):
                out.append(base64.b64encode(open(fp, "rb").read()).decode())
    return out


def judge_notes(src, name):
    run = os.path.dirname(os.path.dirname(src))
    fp = os.path.join(run, "visual_judgments", f"{name}_visual.json")
    try:
        j = json.load(open(fp))
    except Exception:
        return ""
    keep = {k: j[k] for k in ("reason", "reasoning", "summary", "context", "what_happens",
                              "event_type", "verdict", "notes") if k in j}
    return json.dumps(keep or j)[:1500]


def caption_text(finish_dir, limit=900):
    import glob
    cap = sorted(glob.glob(os.path.join(finish_dir, "captions", "*.ass")))
    if not cap:
        return ""
    words = []
    for ln in open(cap[0], encoding="utf-8", errors="ignore"):
        if ln.startswith("Dialogue:"):
            t = ln.split(",", 9)[-1]
            import re
            words.append(re.sub(r"\{[^}]*\}", "", t).strip())
    return " / ".join(words)[:limit]


def ask_claude(frames, notes, captions, signals):
    from anthropic import Anthropic
    content = [{"type": "text", "text": f"Six frames from the clip, in order:"}]
    for b in frames:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
    content.append({"type": "text", "text":
                    f"Visual judge's earlier analysis: {notes or '(none)'}\n"
                    f"Captions (what is said): {captions or '(none)'}\n"
                    f"Measured: {json.dumps(signals)}"})
    r = Anthropic().messages.create(model=MODEL, max_tokens=400, temperature=0, system=SYSTEM,
                                    messages=[{"role": "user", "content": content}])
    txt = r.content[0].text
    a, b = txt.find("{"), txt.rfind("}")
    return json.loads(txt[a:b + 1])


def decide(cls, sig, override=None):
    if override:
        return override, f"manual override: {override}"
    cat, conf = cls.get("category"), float(cls.get("confidence", 0) or 0)
    if cat != "PERSON_FOCUSED":
        return "full", f"{cat} -- the viewer needs the full picture"
    if conf < MIN_CONFIDENCE:
        return "full", f"PERSON_FOCUSED but unsure ({conf:.2f} < {MIN_CONFIDENCE}) -- full-screen to be safe"
    if cls.get("would_split_lose_something"):
        return "full", "PERSON_FOCUSED, but Claude says split would lose something important"
    oc = sig.get("on_camera")
    if oc is not None and oc < MIN_ON_CAMERA:
        return "full", f"PERSON_FOCUSED, but the person is trackable only {oc:.0%} of the time"
    notes = []
    if sig.get("crop_mode") == "action":
        notes.append("crop mode was 'action' (not used as a rejection)")
    return "split", "PERSON_FOCUSED and safe to split" + (f"; {'; '.join(notes)}" if notes else "")


def load_report():
    try:
        return json.load(open(REPORT))
    except Exception:
        return {}


def save_report(report):
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump(report, open(REPORT, "w"), indent=2)


def measure(src, name, track, parts):
    dur = sum(b - a for a, b in parts) if parts else sp.probe(src)[3]
    vdir = os.path.join(os.path.dirname(src), "vertical")
    finish_dir = os.path.join(vdir, "finished", name, "finish")
    sig = {"speech": sp.speech_share(finish_dir, dur)}
    if sig["speech"] is not None:
        sig["speech"] = round(sig["speech"], 2)
    mmed, _m80 = sp.motion_level(src, parts)
    sig["motion"] = None if mmed is None else round(mmed, 1)
    sig["crop_mode"] = sp.crop_mode_of(vdir, name)
    if track:
        ofps = sp.probe(src)[2]
        ft = sp.load_track(track, int(round(sp.probe(src)[3] * ofps)) + 2)
        if parts:
            tr = []
            for a, b in parts:
                tr += ft[int(a * ofps):int(b * ofps)]
            ft = tr or ft
        bad = sum(1 for d in ft if d.get("lost") or d.get("force_full"))
        sig["on_camera"] = round(1 - bad / max(1, len(ft)), 2)
    return sig, finish_dir


def decide_clip(final, report, force=False, override=None, ask=True):
    """Return the decision record for one finished clip (cached in report unless force).
    ask=False: no Claude call -- only the safety checks (used by --layout split)."""
    src, name, track = sp.find_source(final)
    if not src:
        return None
    key = os.path.basename(os.path.realpath(final))
    if key in report and not force and not override and ask:
        return report[key]
    parts, _ = sp.final_parts(src, name)
    sig, finish_dir = measure(src, name, track, parts)
    if not ask and not override:
        oc = sig.get("on_camera")
        if oc is not None and oc < MIN_ON_CAMERA:
            layout, why = "full", f"forced split refused: person trackable only {oc:.0%} of the time"
        else:
            layout, why = "split", "forced split (--layout split); safety checks passed"
        cls = {"category": "NOT_ASKED", "confidence": None}
    else:
        cls = {"category": "UNKNOWN", "confidence": 0}
        if not override:
            try:
                cls = ask_claude(frames_b64(src, parts), judge_notes(src, name), caption_text(finish_dir), sig)
            except Exception as e:
                if "credit balance" in str(e).lower():
                    raise SystemExit("STOPPED: Anthropic credits are used up -- top up and re-run.")
                cls = {"category": "UNKNOWN", "confidence": 0, "reason": f"classification failed: {e}"}
        layout, why = decide(cls, sig, override)
    rec = {"clip": name, "layout": layout, "why": why, "category": cls.get("category"),
           "confidence": cls.get("confidence"), "what_viewer_watches": cls.get("what_viewer_watches"),
           "claude_reason": cls.get("reason"), "signals": sig, "final": final}
    if ask or override:
        report[key] = rec
        save_report(report)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("finals", nargs="+")
    ap.add_argument("--force", action="store_true", help="re-decide clips already in the report")
    ap.add_argument("--override", choices=["split", "full"], help="force this layout for the given clips")
    args = ap.parse_args()
    report = load_report()
    for final in args.finals:
        rec = decide_clip(final, report, force=args.force, override=args.override)
        if rec is None:
            print(f"SKIP {final}: landscape original not found")
            continue
        print(f"{rec['layout'].upper():5s}  {rec['clip']:28s} {str(rec['category']):15s} "
              f"conf={rec['confidence']}  {rec['why']}\n       viewer watches: {rec['what_viewer_watches']}")
    print(f"\nreport: {REPORT}")


if __name__ == "__main__":
    main()
