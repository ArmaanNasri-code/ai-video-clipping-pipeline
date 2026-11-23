#!/usr/bin/env python3
"""
post_queue.py -- the posting queue for @chuzzzclips. Uploads NOTHING.

    python3 post_queue.py            # refresh the queue + open the review page
    python3 post_queue.py --today    # also prepare today's posts (max --cap, default 3)

What it does
  1. Collects every APPROVED clip from the *_final_review run folders, using the version
     the layout step recommends (post_file in _layout_report.json: split or full).
  2. Writes a caption for each (hook + streamer credit + hashtags) -- editable.
  3. Builds post_queue/review.html. Watch each clip and mark it Approve / Skip / Posted,
     edit the caption, then press "Save decisions" (downloads post_decisions.json to
     Downloads). Next run of this script reads it in.
  4. --today puts up to --cap approved clips into post_queue/today/ as
     1_<streamer>.mp4, 2_..., plus captions.txt -- AirDrop them to your phone and post.
     Newest streams first, alternating streamers. Files are hard links (no extra disk).
     After posting, mark them Posted on the review page (or: --posted 1 2 3).

Nothing posts automatically. Later, a scheduler (e.g. Blotato) can read post_queue/queue.json.
"""
import argparse
import datetime
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys

QDIR = "post_queue"
QFILE = os.path.join(QDIR, "queue.json")
DOWNLOADS = os.path.expanduser("~/Downloads")


# ------------------------------------------------------------------ collecting clips
def statuses(run):
    out = {}
    p = os.path.join(run, "_pipeline_report.txt")
    if os.path.exists(p):
        for ln in open(p, errors="ignore"):
            m = re.match(r"^\s+\[([A-Z_]+)\]\s+(\S+)", ln)
            if m:
                out[m.group(2)] = m.group(1)
    return out


def layout_entries(run):
    """candidate stem -> post_file, from _layout_report.json (any of the shapes it may have)."""
    p = os.path.join(run, "_layout_report.json")
    try:
        data = json.load(open(p))
    except Exception:
        return {}
    items = data.get("clips", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = [dict(v, clip=k) if isinstance(v, dict) else {} for k, v in items.items()]
    out = {}
    for e in items if isinstance(items, list) else []:
        f = e.get("post_file") or e.get("normal_file")
        if f:
            if not os.path.isabs(f) and not os.path.exists(f):
                f = os.path.join(run, "production_ready", os.path.basename(f))
            out[stem_of(e.get("clip") or f)] = {"file": f, "layout": e.get("layout", "full")}
    return out


def stem_of(name):
    """2902737688_chat_02_185m00s_1.2x_final_split.mp4 -> chat_02_185m00s_1.2x"""
    b = os.path.splitext(os.path.basename(name))[0]
    b = re.sub(r"^\d{9,}_", "", b)
    return re.sub(r"_final(_split)?$", "", b)


def channels():
    """vod -> channel, from the batch review folders (<channel>_<vod>_... shortcuts)."""
    out = {}
    for fp in glob.glob("batch_runs/*/*/*.mp4"):
        m = re.match(r"([a-z0-9_]+?)_(\d{9,})_", os.path.basename(fp))
        if m:
            out[m.group(2)] = m.group(1)
    for fp in glob.glob("*_final_review/_viewer_clip_check.json"):
        try:
            j = json.load(open(fp))
            out.setdefault(str(j["vod"]), j.get("channel"))
        except Exception:
            pass
    return out


def find_hook(run, stem):
    for fp in glob.glob(os.path.join(run, "**", f"*{stem}*hook*.json"), recursive=True):
        try:
            h = json.load(open(fp)).get("hook")
            if h:
                return h
        except Exception:
            pass
    return None


def judge_info(run, stem):
    try:
        j = json.load(open(os.path.join(run, "visual_judgments", f"{stem}_visual.json")))
        return j.get("score"), j.get("caption_idea") or ""
    except Exception:
        return None, ""


def stream_start(run):
    try:
        return json.load(open(os.path.join(run, "_stream_start.json"))).get("stream_start", "")
    except Exception:
        return datetime.datetime.fromtimestamp(os.path.getmtime(run)).isoformat()


def caption_for(hook, channel):
    text = (hook or "").strip()
    tag = re.sub(r"[^a-z0-9]", "", (channel or "").lower())
    tags = " ".join(t for t in [f"#{tag}" if tag else "", f"#{tag}clips" if tag else "",
                                "#twitchclips", "#streamer", "#fyp"] if t)
    credit = f"🎥 {channel} on Twitch" if channel else ""
    return "\n\n".join(x for x in [text, credit, tags] if x)


def collect(queue):
    chans = channels()
    known = {e["id"] for e in queue}
    added = 0
    for run in sorted(glob.glob("*_final_review")):
        vod = run.split("_")[0]
        if not vod.isdigit():
            continue
        st = statuses(run)
        lay = layout_entries(run)
        for stem, status in st.items():
            if status != "APPROVED":
                continue
            cid = f"{vod}_{stem}"
            if cid in known:
                continue
            entry = lay.get(stem)
            f = entry["file"] if entry else None
            if not f or not os.path.exists(f):
                cands = sorted(glob.glob(os.path.join(run, "production_ready", f"*{stem}_final.mp4")))
                f = cands[0] if cands else None
            if not f:
                continue
            ch = chans.get(vod) or ""
            hook = find_hook(run, stem)
            score, idea = judge_info(run, stem)
            queue.append({
                "id": cid, "vod": vod, "channel": ch, "clip": stem, "file": f,
                "layout": entry["layout"] if entry else "full",
                "hook": hook or idea, "score": score, "stream_start": stream_start(run),
                "caption": caption_for(hook or idea, ch),
                "status": "pending", "added": datetime.date.today().isoformat(),
                "today_slot": None, "posted_on": None,
            })
            known.add(cid)
            added += 1
    return added


# ------------------------------------------------------------------ decisions from the page
def ingest(queue, state):
    files = sorted(glob.glob(os.path.join(DOWNLOADS, "post_decisions*.json")), key=os.path.getmtime)
    last = state.get("ingested_mtime", 0)
    n = 0
    byid = {e["id"]: e for e in queue}
    for fp in files:
        mt = os.path.getmtime(fp)
        if mt <= last:
            continue
        try:
            dec = json.load(open(fp))
        except Exception:
            print(f"  could not read {fp} -- skipped")
            continue
        for cid, d in dec.items():
            e = byid.get(cid)
            if not e:
                continue
            if d.get("status") in ("pending", "approved", "skipped", "posted"):
                if d["status"] == "posted" and e["status"] != "posted":
                    e["posted_on"] = datetime.date.today().isoformat()
                e["status"] = d["status"]
            if d.get("caption"):
                e["caption"] = d["caption"]
            n += 1
        state["ingested_mtime"] = mt
    return n


# ------------------------------------------------------------------ today's posts
def prepare_today(queue, cap):
    today = datetime.date.today().isoformat()
    posted_today = [e for e in queue if e.get("posted_on") == today]
    room = max(0, cap - len(posted_today))
    tdir = os.path.join(QDIR, "today")
    shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(tdir)
    ready = [e for e in queue if e["status"] == "approved" and os.path.exists(e["file"])]
    ready.sort(key=lambda e: (e["stream_start"], e.get("score") or 0), reverse=True)
    picked, last_ch = [], None
    while ready and len(picked) < room:      # newest first, avoid same streamer twice in a row
        nxt = next((e for e in ready if e["channel"] != last_ch), ready[0])
        ready.remove(nxt)
        picked.append(nxt)
        last_ch = nxt["channel"]
    for e in queue:
        if e.get("today_slot") and e["status"] != "posted":
            e["today_slot"] = None
    lines = []
    for i, e in enumerate(picked, 1):
        e["today_slot"] = i
        dest = os.path.join(tdir, f"{i}_{e['channel'] or 'clip'}.mp4")
        try:
            os.link(e["file"], dest)
        except OSError:
            shutil.copy2(e["file"], dest)
        lines.append(f"===== {i}  ({e['channel']}, {e['layout']}) =====\n{e['caption']}\n")
    open(os.path.join(tdir, "captions.txt"), "w").write("\n".join(lines) or "Nothing approved to post.\n")
    return picked, len(posted_today), room


# ------------------------------------------------------------------ review page
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Post queue</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1b1b1b;--mut:#6b6b6b;--line:#e2e0da;--acc:#1f6f5c;--skip:#a23b3b;--post:#3b5ba2}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1e;--ink:#eee;--mut:#9a9a9a;--line:#333;--acc:#4fb39a;--skip:#d86b6b;--post:#7d9be0}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
h1{font-size:18px;margin:0 auto 0 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer}
button.on.approved{background:var(--acc);color:#fff;border-color:var(--acc)}
button.on.skipped{background:var(--skip);color:#fff;border-color:var(--skip)}
button.on.posted{background:var(--post);color:#fff;border-color:var(--post)}
#save{background:var(--acc);color:#fff;border-color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:8px}
video{width:100%;aspect-ratio:9/16;background:#000;border-radius:8px}
.meta{color:var(--mut);font-size:13px}
textarea{width:100%;box-sizing:border-box;min-height:110px;font:inherit;font-size:13px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px}
.row{display:flex;gap:6px;flex-wrap:wrap}
.filter button.on{outline:2px solid var(--acc)}
</style></head><body>
<header><h1>Post queue</h1>
<span class="filter row" id="filters"></span>
<button id="save">Save decisions</button></header>
<p class="meta" style="padding:0 16px">Watch each clip, choose Approve / Skip (or Posted once it's up), edit the caption if needed,
then press <b>Save decisions</b> and run <code>python3 post_queue.py --today</code>. Nothing is uploaded from here.</p>
<div class="grid" id="grid"></div>
<script>
const Q = __DATA__;
const dec = {};
let filter = "pending";
const grid = document.getElementById("grid");
function counts(){const c={all:Q.length};for(const e of Q){const s=(dec[e.id]&&dec[e.id].status)||e.status;c[s]=(c[s]||0)+1}return c}
function drawFilters(){const c=counts();const f=document.getElementById("filters");f.innerHTML="";
 for(const k of ["pending","approved","skipped","posted","all"]){const b=document.createElement("button");
  b.textContent=k+" ("+(c[k]||0)+")";if(k===filter)b.className="on";b.onclick=()=>{filter=k;draw()};f.appendChild(b)}}
function draw(){drawFilters();grid.innerHTML="";
 for(const e of Q){const st=(dec[e.id]&&dec[e.id].status)||e.status;if(filter!=="all"&&st!==filter)continue;
  const d=document.createElement("div");d.className="card";
  const v=document.createElement("video");v.src=e.src;v.controls=true;v.preload="metadata";v.playsInline=true;
  const m=document.createElement("div");m.className="meta";
  m.textContent=(e.channel||"?")+" · "+e.layout+" · judge "+(e.score??"-")+" · "+e.clip;
  const t=document.createElement("textarea");t.value=(dec[e.id]&&dec[e.id].caption)||e.caption;
  t.oninput=()=>{dec[e.id]=Object.assign(dec[e.id]||{status:st},{caption:t.value})};
  const r=document.createElement("div");r.className="row";
  for(const s of ["approved","skipped","posted","pending"]){const b=document.createElement("button");
   b.textContent={approved:"Approve",skipped:"Skip",posted:"Posted",pending:"Undo"}[s];
   if(s===st&&s!=="pending")b.className="on "+s;
   b.onclick=()=>{dec[e.id]=Object.assign(dec[e.id]||{},{status:s,caption:t.value});draw()};r.appendChild(b)}
  d.append(v,m,t,r);grid.appendChild(d)}}
document.getElementById("save").onclick=()=>{
 const blob=new Blob([JSON.stringify(dec,null,2)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="post_decisions.json";a.click()};
draw();
</script></body></html>"""


def write_page(queue):
    rows = []
    for e in queue:
        rel = os.path.relpath(os.path.abspath(e["file"]), os.path.abspath(QDIR))
        rows.append({**{k: e[k] for k in ("id", "channel", "clip", "layout", "score", "caption", "status")},
                     "src": rel})
    data = json.dumps(rows).replace("</", "<\\/")
    path = os.path.join(QDIR, "review.html")
    open(path, "w").write(PAGE.replace("__DATA__", data))
    return path


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--today", action="store_true", help="prepare today's posts in post_queue/today/")
    ap.add_argument("--cap", type=int, default=3, help="max posts per day (default 3)")
    ap.add_argument("--posted", nargs="*", type=int, help="mark today's slot numbers as posted, e.g. --posted 1 2")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    if not glob.glob("*_final_review"):
        sys.exit("ERROR: no *_final_review folders here -- run from ~/Downloads.")
    os.makedirs(QDIR, exist_ok=True)
    try:
        store = json.load(open(QFILE))
    except Exception:
        store = {"queue": [], "state": {}}
    queue, state = store["queue"], store["state"]

    added = collect(queue)
    n_dec = ingest(queue, state)
    if args.posted:
        today = datetime.date.today().isoformat()
        for e in queue:
            if e.get("today_slot") in args.posted:
                e["status"], e["posted_on"] = "posted", today
    picked = None
    if args.today:
        picked, done_today, room = prepare_today(queue, args.cap)
    json.dump(store, open(QFILE, "w"), indent=2)
    page = write_page(queue)

    c = {}
    for e in queue:
        c[e["status"]] = c.get(e["status"], 0) + 1
    print(f"queue: {len(queue)} clips (+{added} new) -- " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    if n_dec:
        print(f"read {n_dec} decision(s) from post_decisions.json")
    if picked is not None:
        if picked:
            print(f"\ntoday: {len(picked)} clip(s) ready in {QDIR}/today/ (posted today already: {done_today}, cap {args.cap})")
            for e in picked:
                print(f"  {e['today_slot']}. {e['channel']:14s} {e['layout']:5s} {e['hook'] or ''}")
            print(f"  captions: {QDIR}/today/captions.txt -- after posting: python3 post_queue.py --posted "
                  + " ".join(str(e['today_slot']) for e in picked))
        else:
            print(f"\ntoday: nothing to prepare (approved left: {c.get('approved', 0)}, posted today: {done_today}, cap {args.cap})")
    print(f"review page: {page}")
    print("Nothing was uploaded or posted.")
    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", page])
        if picked:
            subprocess.run(["open", os.path.join(QDIR, "today")])


if __name__ == "__main__":
    main()
