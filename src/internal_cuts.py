#!/usr/bin/env python3
"""
internal_cuts.py -- test-only "editor" pass that may also cut OUT dull parts in
the middle of a clip (not just trim the start and end).

Reuses smart_trim.py's evidence (word-level transcript, energy, scene changes,
transition/logo frames, frames every 2s). Claude, as a TikTok editor, returns
the parts to KEEP, in their original order, and why each removed part can go
(repetition, tangent, unrelated conversation, dead air). Then:
  - every cut is snapped to sentence boundaries (nobody is cut off mid-sentence)
  - any sentence with an allegation about a real person or a slur is cut out
  - kept parts closer than 1s are joined; kept parts under 1.5s are dropped
  - SAFETY: no cuts at all (original kept) when the editor is unsure, when the
    payoff isn't fully kept, when there would be more than MAX_PARTS parts, or
    when the cuts would remove less than 3s
  - joins get a short crossfade (0.15s) so they don't pop
Order is never changed (no re-arranging events).

Nothing is overwritten. Writes only into --out (default cuts_test/):
  <clip>_cuts.mp4   preview (only when cuts are applied)
  <clip>_cuts.json  the editor's decision and every safety step

  python3 internal_cuts.py clip.mp4 [more clips] [--out cuts_test]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import smart_trim as st

MAX_PARTS = 4
MIN_PART = 1.5
JOIN_GAP = 1.0
XFADE = 0.15
MIN_REMOVED = 3.0
MIN_TOTAL = 12.0

ASK = """Clip duration: {dur:.1f}s.

Decide which parts to KEEP so the clip plays like a tight, human-edited TikTok.
You may remove parts from the MIDDLE as well as the start/end.

Rules:
- Keep the original order. Never rearrange events.
- Keep the setup a new viewer needs, the full payoff, and the reaction.
- A replay of the payoff can be kept if it adds something.
- Only remove a middle part if it is genuinely unnecessary: repetition, a tangent or
  unrelated conversation, dead air, a long lull. Never remove something the viewer
  needs to understand what comes after it.
- Cuts must fall between sentences, never inside one.
- Do NOT cut in the middle of continuous on-screen action (a fight exchange, a chase,
  an argument in progress) just because the talk is off-topic: the cut would show a
  visible jump in the action. Cut where the picture changes (scene change, replay,
  transition) or where nothing is happening on screen.
- Remove allegations about real people (e.g. "he's a woman-beater") and threats.
  (Slurs are masked in the captions; don't cut action just to remove one.)
- At most {maxp} kept parts, each at least 3 seconds. Fewer cuts is better.
- If the clip already flows well, return ONE kept part covering what should stay.
- If unsure, confidence "low".

Return ONLY JSON:
{{
  "keep": [{{"start": <s>, "end": <s>, "what": "<...>"}}],
  "removed": [{{"start": <s>, "end": <s>, "why": "<...>"}}],
  "payoff": {{"start": <s>, "end": <s>, "what": "<...>"}},
  "confidence": "high|medium|low",
  "reason": "<one or two sentences>"
}}"""


def snap_parts(parts, sents, words, dur):
    out = []
    for a, b in parts:
        a, b = st.snap_start(a, sents), st.snap_end(b, sents, dur)
        out.append([max(0.0, a), min(dur, b)])
    # cut out sentences with allegations about real people (slurs are masked on screen instead)
    bad = []
    for sa, sb in sents:
        line = " ".join(w for x, y, w in words if x >= sa - 0.01 and y <= sb + 0.01)
        if st.ALLEGATION.search(line):
            bad.append((sa - 0.05, sb + 0.05, line[:100]))
    carved = []
    dropped = []
    for a, b in out:
        inside = [x for x in bad if not (x[1] <= a or x[0] >= b)]
        covered = sum(min(b, x[1]) - max(a, x[0]) for x in inside)
        if len(inside) >= 2 or (b > a and covered / (b - a) > 0.15):
            # the conversation in this part IS the allegation -> drop the whole part
            dropped.append([round(a, 2), round(b, 2)])
            continue
        pieces = [[a, b]]
        for ba, bb, _ in bad:
            nxt = []
            for pa, pb in pieces:
                if bb <= pa or ba >= pb:
                    nxt.append([pa, pb])
                else:
                    if ba > pa:
                        nxt.append([pa, ba])
                    if bb < pb:
                        nxt.append([bb, pb])
            pieces = nxt
        carved += pieces
    carved.sort()
    merged = []
    for a, b in carved:
        if merged and a - merged[-1][1] < JOIN_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    hit = [x[2] for x in bad if any(not (x[1] <= a or x[0] >= b) for a, b in out)]
    if dropped:
        hit.append(f"dropped whole kept part(s) {dropped}: the talk there is the allegation")
    return [p for p in merged if p[1] - p[0] >= MIN_PART], hit


def guard_action_cuts(parts, energy, cuts, trans, step=0.5, near=1.2):
    """Only allow a cut in the middle of the clip where the picture changes anyway
    (scene change / replay / transition near either side of the cut) or where both
    sides are quiet. Otherwise the cut would show a visible jump in ongoing action,
    so the gap is filled back in."""
    import numpy as np
    marks = sorted(list(cuts) + list(trans))
    quiet = float(np.percentile(energy, 30)) if len(energy) else 0.0

    def e_at(t):
        i = min(len(energy) - 1, max(0, int(t / step)))
        return float(energy[i]) if len(energy) else 0.0

    def picture_changes(t):
        return any(abs(m - t) <= near for m in marks)

    out, refused = [list(parts[0])] if parts else [], []
    for a, b in parts[1:]:
        cut_out, cut_in = out[-1][1], a
        ok = (picture_changes(cut_out) or picture_changes(cut_in)
              or (e_at(cut_out) < quiet and e_at(cut_in) < quiet))
        if ok:
            out.append([a, b])
        else:
            refused.append([round(cut_out, 2), round(cut_in, 2)])
            out[-1][1] = b          # keep the gap: no jump in the middle of action
    return out, refused


def render(video, parts, outp):
    n = len(parts)
    fc, cur_v, cur_a, t_total = [], None, None, 0.0
    for i, (a, b) in enumerate(parts):
        fc.append(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}];"
                  f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[a{i}]")
    if n == 1:
        maps = ["[v0]", "[a0]"]
    else:
        cur_v, cur_a = "v0", "a0"
        t_total = parts[0][1] - parts[0][0]
        for i in range(1, n):
            off = t_total - XFADE
            fc.append(f"[{cur_v}][v{i}]xfade=transition=fade:duration={XFADE}:offset={off:.3f}[xv{i}]")
            fc.append(f"[{cur_a}][a{i}]acrossfade=d={XFADE}[xa{i}]")
            cur_v, cur_a = f"xv{i}", f"xa{i}"
            t_total = t_total + (parts[i][1] - parts[i][0]) - XFADE
        maps = [f"[{cur_v}]", f"[{cur_a}]"]
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video, "-filter_complex",
                    ";".join(fc), "-map", maps[0], "-map", maps[1], "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", "-y", outp], check=True)


def cut_one(client, video, out_dir, tmp):
    name = os.path.splitext(os.path.basename(video))[0]
    dur = st.duration(video)
    print("   [1/4] transcript (cached after the first run)...", flush=True)
    words, segs = st.get_words(video, os.path.join(out_dir, f"{name}_words.json"))
    sents = st.sentences(words)
    print("   [2/4] measuring loudness, motion and transitions...", flush=True)
    energy, cuts = st.energy_curve(video, dur)
    trans = st.transition_times(video, dur)
    print("   [3/4] asking the editor (about 20-40s)...", flush=True)

    content = [{"type": "text", "text": f"CLIP: {name}\nDURATION: {dur:.1f}s\n"}]
    t, k = 0.5, 0
    while t < dur:
        content.append({"type": "text", "text": f"FRAME {t:.1f}s"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                        "data": st.frame_at(video, t, os.path.join(tmp, f"i{k}.jpg"))}})
        t += st.COARSE_STEP
        k += 1
    tr = "\n".join(f"[{a:.1f}-{b:.1f}] {x}" for a, b, x in segs) or "(no speech)"
    sent_txt = "\n".join(
        f"[{a:.1f}-{b:.1f}] " + " ".join(w for x, y, w in words if x >= a - 0.01 and y <= b + 0.01)
        for a, b in sents) or "(none)"
    en = " ".join(f"{i * 2:.0f}s:{energy[int(i * 4)]:+.1f}" for i in range(int(len(energy) / 4)))
    content.append({"type": "text", "text":
                    f"\nTRANSCRIPT (timestamped):\n{tr}\n\nSENTENCES (cut only between these):\n"
                    f"{sent_txt}\n\nENERGY every 2s (loud is NOT the same as important):\n{en}\n\n"
                    f"SCENE CHANGES at: {', '.join(str(c) for c in cuts) or 'none'}\n"
                    f"TRANSITION / LOGO / BLUR frames at: "
                    f"{', '.join(f'{x:.1f}' for x in trans) or 'none'}\n\n"
                    + ASK.format(dur=dur, maxp=MAX_PARTS)})
    d = st.ask(client, content, max_tokens=1200)
    res = {"clip": name, "duration": round(dur, 2), "editor": d}

    raw = [(float(p["start"]), float(p["end"])) for p in d.get("keep", []) if float(p["end"]) > float(p["start"])]
    parts, removed_lines = snap_parts(raw, sents, words, dur)
    parts, refused = guard_action_cuts(parts, energy, cuts, trans)
    if refused:
        res["refused_cuts_through_action"] = refused
    res["parts"] = [[round(a, 2), round(b, 2)] for a, b in parts]
    if removed_lines:
        res["cut_out_sensitive_lines"] = removed_lines

    why_not = None
    pay = d.get("payoff") or {}
    kept_total = sum(b - a for a, b in parts)
    if d.get("confidence") == "low":
        why_not = "editor not confident"
    elif not parts:
        why_not = "nothing usable to keep"
    elif len(parts) > MAX_PARTS:
        why_not = f"more than {MAX_PARTS} parts"
    elif pay.get("start") is not None and not any(
            a <= float(pay["start"]) and float(pay.get("end", pay["start"])) <= b for a, b in parts):
        why_not = "the payoff isn't fully inside one kept part"
    elif kept_total < MIN_TOTAL:
        why_not = f"what's left would be under {MIN_TOTAL:.0f}s"
    elif dur - kept_total < MIN_REMOVED:
        why_not = f"would remove under {MIN_REMOVED:.0f}s"
    res["applied"] = why_not is None
    res["kept_original_because"] = why_not
    res["kept_seconds"] = round(kept_total - XFADE * max(0, len(parts) - 1), 1)
    if why_not is None:
        print("   [4/4] rendering preview...", flush=True)
        outp = os.path.join(out_dir, f"{name}_cuts.mp4")
        render(video, parts, outp)
        res["preview"] = outp
    json.dump(res, open(os.path.join(out_dir, f"{name}_cuts.json"), "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="cuts_test")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")
    os.makedirs(args.out, exist_ok=True)
    client = st.Anthropic()
    with tempfile.TemporaryDirectory() as tmp:
        for v in args.videos:
            print(f"\n== {os.path.basename(v)}")
            try:
                r = cut_one(client, v, args.out, tmp)
            except Exception as e:
                print(f"   FAILED: {e!r} -- original kept")
                continue
            d = r["editor"]
            print(f"   payoff:  {d.get('payoff')}")
            for p in d.get("removed", []):
                print(f"   remove:  {p.get('start')}-{p.get('end')}s  {p.get('why')}")
            if r.get("refused_cuts_through_action"):
                print(f"   refused (would jump mid-action, kept that stretch): {r['refused_cuts_through_action']}")
            if r.get("cut_out_sensitive_lines"):
                print(f"   also cut (allegation): {r['cut_out_sensitive_lines']}")
            fmt = lambda x: f"{int(x // 60)}:{x % 60:04.1f}"
            parts = " + ".join(f"{fmt(a)}-{fmt(b)}" for a, b in r["parts"])
            if r["applied"]:
                print(f"   => CUT: keep {parts}  ({r['kept_seconds']}s of {r['duration']:.0f}s)")
            else:
                print(f"   => NO CUTS ({r['kept_original_because']}); proposed {parts}")
    print(f"\npreviews and decisions in {args.out}/ (originals untouched)")


if __name__ == "__main__":
    main()
