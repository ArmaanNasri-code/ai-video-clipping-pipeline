#!/usr/bin/env python3
"""
viewer_clip_check.py -- did the clipper find the moments real viewers clipped?

For each finished VOD it fetches the clips Twitch VIEWERS made from that stream (with
view counts and their position in the VOD) and compares them with what the clipper did:

  APPROVED   the clipper found this moment and approved a clip of it
  REVIEW     found, finished, but held for review
  REJECTED   found, but the judge rejected it            -> possible JUDGING miss
  MISSED     the clipper never had a candidate near it   -> possible DETECTION miss

Read-only: changes nothing, costs no Anthropic credits. Needs TWITCH_CLIENT_ID and
TWITCH_CLIENT_SECRET (saved in ~/.twitch_keys; new Terminal windows load them).

    python3 viewer_clip_check.py                 # every *_final_review run folder
    python3 viewer_clip_check.py 2934476444      # one VOD
    python3 viewer_clip_check.py --top 15 --min-views 20

Saves <run>/_viewer_clip_check.json and prints a summary per VOD plus a total.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import urllib.parse
import urllib.request

NEAR_BEFORE = 45     # a candidate "covers" a viewer clip if the clip's middle is within
NEAR_AFTER = 75      # [candidate time - 45s, candidate time + 75s]


def keys():
    cid, sec = os.environ.get("TWITCH_CLIENT_ID"), os.environ.get("TWITCH_CLIENT_SECRET")
    if not (cid and sec) and os.path.exists(os.path.expanduser("~/.twitch_keys")):
        for ln in open(os.path.expanduser("~/.twitch_keys")):
            m = re.match(r"export (\w+)=(.*)", ln.strip())
            if m:
                os.environ[m.group(1)] = m.group(2)
        cid, sec = os.environ.get("TWITCH_CLIENT_ID"), os.environ.get("TWITCH_CLIENT_SECRET")
    if not (cid and sec):
        sys.exit("ERROR: TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET not set (see ~/.twitch_keys).")
    return cid, sec


def token(cid, sec):
    data = urllib.parse.urlencode({"client_id": cid, "client_secret": sec,
                                   "grant_type": "client_credentials"}).encode()
    with urllib.request.urlopen("https://id.twitch.tv/oauth2/token", data, timeout=30) as r:
        return json.load(r)["access_token"]


def helix(path, params, cid, tok):
    url = "https://api.twitch.tv/helix/" + path + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"Client-Id": cid, "Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def parse_dur(s):
    tot = 0
    for n, u in re.findall(r"(\d+)([hms])", s or ""):
        tot += int(n) * {"h": 3600, "m": 60, "s": 1}[u]
    return tot


def vod_clips(vid, cid, tok, max_pages=5):
    v = helix("videos", {"id": vid}, cid, tok)["data"]
    if not v:
        return None, []
    v = v[0]
    start = datetime.datetime.fromisoformat(v["created_at"].replace("Z", "+00:00"))
    end = start + datetime.timedelta(seconds=parse_dur(v["duration"])) + datetime.timedelta(days=3)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    params = {"broadcaster_id": v["user_id"], "started_at": start.strftime(fmt),
              "ended_at": end.strftime(fmt), "first": 100}
    clips, cursor = [], None
    for _ in range(max_pages):
        if cursor:
            params["after"] = cursor
        page = helix("clips", params, cid, tok)
        clips += [c for c in page.get("data", []) if c.get("video_id") == str(vid)]
        cursor = (page.get("pagination") or {}).get("cursor")
        if not cursor:
            break
    return v, clips


def candidate_times(run):
    """name -> VOD seconds, from the candidate names (e.g. chat_01_58m15s_2.8x, visualdisc_06_155m21s)."""
    names = set()
    for fp in glob.glob(os.path.join(run, "visual_judgments", "*_visual.json")):
        names.add(os.path.basename(fp)[: -len("_visual.json")])
    rp = os.path.join(run, "_pipeline_report.txt")
    status = {}
    if os.path.exists(rp):
        for ln in open(rp, errors="ignore"):
            m = re.match(r"^\s+\[([A-Z_]+)\]\s+(\S+)", ln)
            if m:
                status[m.group(2)] = m.group(1)
                names.add(m.group(2))
    out = {}
    for n in names:
        m = re.search(r"_(\d+)m(\d+)s", n)
        if m:
            out[n] = int(m.group(1)) * 60 + int(m.group(2))
    return out, status


def judgment(run, name):
    try:
        j = json.load(open(os.path.join(run, "visual_judgments", f"{name}_visual.json")))
    except Exception:
        return None, ""
    reason = next((j[k] for k in ("reason", "reasoning", "summary", "verdict") if isinstance(j.get(k), str)), "")
    return j.get("score"), reason


def check(vid, cid, tok, top, min_views):
    run = f"{vid}_final_review"
    cands, status = candidate_times(run)
    v, clips = vod_clips(vid, cid, tok)
    if v is None:
        return None, f"{vid}: VOD not found on Twitch (deleted or expired)"
    usable = [c for c in clips if c.get("vod_offset") is not None and c.get("view_count", 0) >= min_views]
    usable.sort(key=lambda c: -c["view_count"])
    rows = []
    for c in usable[:top]:
        mid = c["vod_offset"] + c.get("duration", 30) / 2
        near = [(n, t) for n, t in cands.items() if t - NEAR_BEFORE <= mid <= t + NEAR_AFTER]
        near.sort(key=lambda nt: abs(nt[1] - mid))
        if not near:
            verdict, cand, score, reason = "MISSED", None, None, ""
        else:
            # best outcome among nearby candidates
            order = {"APPROVED": 0, "EDIT_REVIEW": 1, "CAPTION_REVIEW": 1, "LOW_PRESENCE": 2,
                     "FINISH_FAILED": 2, "CROP_FAILED": 2, "VISUAL_REJECTED": 3}
            near.sort(key=lambda nt: order.get(status.get(nt[0], "VISUAL_REJECTED"), 3))
            cand = near[0][0]
            st = status.get(cand, "VISUAL_REJECTED")
            verdict = {"APPROVED": "APPROVED", "EDIT_REVIEW": "REVIEW", "CAPTION_REVIEW": "REVIEW",
                       "VISUAL_REJECTED": "REJECTED"}.get(st, st)
            score, reason = judgment(run, cand)
        rows.append({"views": c["view_count"], "title": c["title"], "vod_time": int(mid),
                     "url": c["url"], "verdict": verdict, "candidate": cand,
                     "judge_score": score, "judge_reason": reason[:300]})
    summ = {"vod": vid, "channel": v.get("user_login"), "title": v.get("title"),
            "viewer_clips_total": len(clips), "checked": len(rows),
            "counts": {k: sum(1 for r in rows if r["verdict"] == k)
                       for k in ("APPROVED", "REVIEW", "REJECTED", "MISSED")},
            "candidates": len(cands), "rows": rows}
    if os.path.isdir(run):
        json.dump(summ, open(os.path.join(run, "_viewer_clip_check.json"), "w"), indent=2)
    return summ, None


def hms(s):
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vods", nargs="*")
    ap.add_argument("--top", type=int, default=10, help="check the N most-viewed viewer clips per VOD")
    ap.add_argument("--min-views", type=int, default=10)
    args = ap.parse_args()
    vods = args.vods or sorted(d.split("_")[0] for d in glob.glob("*_final_review") if d.split("_")[0].isdigit())
    cid, sec = keys()
    tok = token(cid, sec)
    tot = {"APPROVED": 0, "REVIEW": 0, "REJECTED": 0, "MISSED": 0}
    for vid in vods:
        try:
            s, err = check(vid, cid, tok, args.top, args.min_views)
        except Exception as e:
            print(f"\n{vid}: check failed ({e})")
            continue
        if err:
            print(f"\n{err}")
            continue
        print(f"\n=== {s['channel']} VOD {vid}  {s['title'][:50]!r}")
        print(f"    viewer clips: {s['viewer_clips_total']} | clipper candidates: {s['candidates']} | "
              f"top {s['checked']} checked: " + ", ".join(f"{k} {v}" for k, v in s["counts"].items()))
        for r in s["rows"]:
            extra = (f"  judge {r['judge_score']}: {r['judge_reason'][:90]}" if r["verdict"] == "REJECTED"
                     else f"  ({r['candidate']})" if r["candidate"] else "")
            print(f"    {r['verdict']:8s} {r['views']:>7} views @ {hms(r['vod_time'])}  {r['title'][:45]!r}{extra}")
        for k in tot:
            tot[k] += s["counts"][k]
    n = sum(tot.values())
    if n:
        print(f"\nTOTAL over {n} top viewer clips: " + ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in tot.items()))
        print("MISSED = detection never found it.  REJECTED = found but the judge said no.")
    print("Nothing was changed.")


if __name__ == "__main__":
    main()
