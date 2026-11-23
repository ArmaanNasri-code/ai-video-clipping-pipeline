#!/usr/bin/env python3
"""
add_streamer.py -- add a new streamer with one command (no manual face boxes).

  python3 add_streamer.py --channel adinross
  python3 add_streamer.py --channel adinross --vod 2918060463 2837447163   # pick the VODs yourself
  python3 add_streamer.py --channel kaicenat --name kai --append  # add more faces to an existing reference

What it does:
  1. picks 3 VODs spread across the channel's last 15 (yt-dlp, no Twitch API keys needed)
  2. downloads a few short snippets spread across the stream
  3. finds every face and groups them by person; on their own stream the
     streamer is almost always the person seen in the most snippets
  4. shows you a picture sheet of that person and asks "is this them?"
     (y = yes, n = show the next person, q = quit) -- nothing is saved until you say y
  5. writes target_reference_<name>.json in the SAME format as
     build_target_reference.py (dlib face encodings + OSNet embeddings), and adds
     the channel to reference_map.json (backed up first)

Safe: never overwrites an existing reference unless you pass --append (which adds
to it). Snippets are kept in streamer_refs/<name>/ and reused on reruns.
Run from ~/Downloads with the clipper_v8_env python.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

REID_WEIGHTS = (
    "~/.cache/huggingface/hub/models--kaiyangzhou--osnet/"
    "snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
CLUSTER_DIST = 0.50      # dlib distance to join a person group (dlib's own "same person" cutoff is 0.6)
KEEP_DIST = 0.45         # only faces this close to the group's centre go into the reference
MAX_REF_FACES = 30


def ytdlp(*args):
    r = subprocess.run([sys.executable, "-m", "yt_dlp", *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def find_vods(channel, n):
    code, out, err = ytdlp("--flat-playlist", "--playlist-end", "15", "--print", "%(id)s|%(title)s",
                           f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time")
    if code != 0 or not out:
        sys.exit(f"ERROR: could not list VODs for '{channel}'. Check the channel name, "
                 f"or pass --vod <id>.\n{err[-400:]}")
    allv = [(l.split("|", 1)[0].lstrip("v"), l.split("|", 1)[1]) for l in out.splitlines() if "|" in l]
    if len(allv) <= n:
        return allv
    # spread the picks across the list (newest, middle, oldest...) -- streamers often
    # run the same kind of stream several days in a row, so consecutive VODs look alike
    return [allv[round(i * (len(allv) - 1) / (n - 1))] for i in range(n)] if n > 1 else allv[:1]


def vod_info(vod):
    code, out, err = ytdlp("--skip-download", "--print", "%(duration)s|%(uploader)s|%(title)s",
                           f"https://www.twitch.tv/videos/{vod}")
    if code != 0 or not out:
        sys.exit(f"ERROR: could not read VOD {vod}.\n{err[-400:]}")
    dur, uploader, title = out.split("|", 2)
    return float(dur), uploader, title


def get_snippets(vod, duration, folder, n, secs):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i in range(n):
        start = int(duration * (0.08 + 0.84 * i / max(1, n - 1)))
        path = os.path.join(folder, f"snip_{vod}_{i:02d}.mp4")
        if not os.path.exists(path):
            print(f"  downloading snippet {i + 1}/{n} at {start // 3600}:{start % 3600 // 60:02d}:{start % 60:02d} ...")
            code, _, err = ytdlp("--quiet", "--no-warnings", "-f", "best[height<=720]/best",
                                 "--download-sections", f"*{start}-{start + secs}",
                                 "-o", path, f"https://www.twitch.tv/videos/{vod}")
            if code != 0 or not os.path.exists(path):
                print(f"    snippet {i + 1} failed, skipping ({err[-160:]})")
                continue
        paths.append(path)
    return paths


def collect_faces(paths, every_s):
    import face_recognition
    faces = []  # dict(enc, snip, frame_img_path?, box, frame)
    for si, p in enumerate(paths):
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(fps * every_s))
        idx, got = 0, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
                    locs = [l for l in locs if (l[2] - l[0]) >= 40]   # skip tiny background faces
                    if locs:
                        encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
                        for loc, enc in zip(locs, encs):
                            faces.append({"enc": np.asarray(enc), "snip": si, "loc": loc, "frame": frame})
                            got += 1
            idx += 1
        cap.release()
        print(f"  snippet {si + 1}/{len(paths)}: {got} faces")
    return faces


def cluster(faces):
    groups = []  # dict(members=[i], centre=vec)
    for i, f in enumerate(faces):
        best, bd = None, 9
        for g in groups:
            d = np.linalg.norm(g["centre"] - f["enc"])
            if d < bd:
                best, bd = g, d
        if best is not None and bd < CLUSTER_DIST:
            best["members"].append(i)
            best["centre"] = np.mean([faces[j]["enc"] for j in best["members"]], axis=0)
        else:
            groups.append({"members": [i], "centre": f["enc"].copy()})
    for g in groups:
        g["snips"] = len({faces[j]["snip"] for j in g["members"]})
    groups.sort(key=lambda g: (g["snips"], len(g["members"])), reverse=True)
    return groups


def contact_sheet(faces, members, path, cols=6, size=140):
    pick = members[:: max(1, len(members) // (cols * 4))][: cols * 4]
    tiles = []
    for j in pick:
        t, r, b, l = faces[j]["loc"]
        h, w = b - t, r - l
        fr = faces[j]["frame"]
        crop = fr[max(0, t - h // 3): b + h // 3, max(0, l - w // 3): r + w // 3]
        tiles.append(cv2.resize(crop, (size, size)) if crop.size else np.zeros((size, size, 3), np.uint8))
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))


def torso_crop_like_original(frame, x, y, w, h):
    """Identical to build_target_reference.py's crop_box_torso, applied to the
    face box exactly as that script does, so the files match."""
    x1, y1, x2 = int(max(0, x)), int(max(0, y)), int(x + w)
    crop = frame[int(y1 + 0.10 * h):int(y1 + 0.55 * h), x1:x2]
    return crop if crop.size > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, help="Twitch login, e.g. adinross")
    ap.add_argument("--name", default=None, help="name for target_reference_<name>.json (default: channel)")
    ap.add_argument("--vod", nargs="+", default=None,
                    help="one or more VOD ids to use (best: different settings, e.g. one IRL + one at home)")
    ap.add_argument("--vods", type=int, default=3, help="how many VODs to use (spread across the last 15) when --vod is not given")
    ap.add_argument("--snippets", type=int, default=12, help="total snippets, split across the VODs")
    ap.add_argument("--snippet-seconds", type=int, default=20)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--append", action="store_true", help="add to an existing reference")
    ap.add_argument("--no-map", action="store_true",
                    help="don't touch reference_map.json (for testing)")
    args = ap.parse_args()

    channel = args.channel.strip().lower()
    name = (args.name or channel).strip().lower()
    out_file = f"target_reference_{name}.json"
    if os.path.exists(out_file) and not args.append:
        sys.exit(f"{out_file} already exists. Use --append to add more faces to it. Nothing changed.")

    if args.vod:
        vods = [(v.rstrip("/").split("/")[-1].lstrip("v"), "") for v in args.vod]
    else:
        vods = find_vods(channel, args.vods)
    folder = os.path.join("streamer_refs", name)
    per = max(2, args.snippets // len(vods))
    print("\n[1/3] getting snippets")
    paths = []
    for vod, _ in vods:
        duration, uploader, title = vod_info(vod)
        print(f"VOD {vod}: {uploader} | {duration / 3600:.1f}h | {title}")
        if uploader.strip().lower() not in (channel, ""):
            print(f"  note: this VOD belongs to '{uploader}', not '{channel}'")
        paths += get_snippets(vod, duration, folder, per, args.snippet_seconds)
    if not paths:
        sys.exit("ERROR: no snippets downloaded. Nothing changed.")

    print("\n[2/3] finding faces")
    faces = collect_faces(paths, args.every)
    groups = [g for g in cluster(faces) if len(g["members"]) >= 4]
    if not groups:
        sys.exit("ERROR: no face appeared often enough. Try another VOD with --vod. Nothing changed.")

    print(f"\n[3/3] {len(groups)} recurring people found. Showing the most frequent first.")
    chosen = None
    for k, g in enumerate(groups[:5]):
        sheet = os.path.join(folder, f"person_{k + 1}.jpg")
        contact_sheet(faces, g["members"], sheet)
        subprocess.run(["open", sheet])
        print(f"\nperson {k + 1}: seen in {g['snips']} of {len(paths)} snippets, {len(g['members'])} faces "
              f"-> picture opened ({sheet})")
        ans = input(f"Is this {channel}? [y = yes / n = next person / q = quit]: ").strip().lower()
        if ans == "y":
            chosen = g
            break
        if ans == "q":
            break
    if chosen is None:
        sys.exit("No person confirmed. Nothing saved.")

    # clean, spread-out set of faces for the reference
    mem = [j for j in chosen["members"] if np.linalg.norm(faces[j]["enc"] - chosen["centre"]) < KEEP_DIST]
    mem.sort(key=lambda j: faces[j]["snip"])
    if len(mem) > MAX_REF_FACES:
        mem = [mem[int(i * len(mem) / MAX_REF_FACES)] for i in range(MAX_REF_FACES)]

    print("\nLoading OSNet Re-ID extractor...")
    from torchreid.reid.utils import FeatureExtractor
    reid = FeatureExtractor(model_name="osnet_x1_0", model_path=REID_WEIGHTS, device="cpu")

    face_encodings, reid_embeddings = [], []
    if args.append and os.path.exists(out_file):
        old = json.load(open(out_file))
        face_encodings = old.get("face_encodings", [])
        reid_embeddings = old.get("reid_embeddings", [])
        shutil.copy2(out_file, out_file.replace(".json", "_before_append.json"))
    n0 = len(face_encodings)
    for j in mem:
        f = faces[j]
        face_encodings.append(f["enc"].tolist())
        t, r, b, l = f["loc"]
        crop = torso_crop_like_original(f["frame"], l, t, r - l, b - t)
        if crop is not None:
            reid_embeddings.append(reid([crop])[0].cpu().numpy().tolist())

    with open(out_file, "w") as fh:
        json.dump({"name": name, "face_encodings": face_encodings,
                   "reid_embeddings": reid_embeddings}, fh)
    print(f"saved {out_file}: {len(face_encodings)} face encodings "
          f"({len(face_encodings) - n0} new), {len(reid_embeddings)} reid embeddings")

    mp = "reference_map.json"
    if args.no_map:
        print("--no-map: reference_map.json left unchanged")
        return
    mapping = json.load(open(mp)) if os.path.exists(mp) else {}
    if mapping.get(channel) != out_file:
        if os.path.exists(mp):
            shutil.copy2(mp, "reference_map_backup.json")
        mapping[channel] = out_file
        with open(mp, "w") as fh:
            json.dump(mapping, fh, indent=2)
        print(f"reference_map.json: {channel} -> {out_file}  (backup: reference_map_backup.json)")
    print(f"\nDone. Run the pipeline with --channel {channel}; the tracker builds the ArcFace "
          f"file for {name} by itself on the first run.")


if __name__ == "__main__":
    main()
