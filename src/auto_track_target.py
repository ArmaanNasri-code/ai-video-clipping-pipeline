#!/usr/bin/env python3
"""
auto_track_target.py

Fully automated, zero-manual-touch person tracker for one clip. Loads
a persistent identity reference (built once via build_target_reference.py)
and locks onto that target for the whole clip -- no per-clip seed file,
no GUI interaction.

Carries over the gating logic v21 proved out (tight spatial gate + IOU
check + identity threshold + velocity-hold + EMA smoothing) but drops
the two-way Hungarian assignment against a second named identity
("Haddy"), since that identity isn't known in advance for a new clip.
Single-target matching: for each frame, find the box (if any) that
passes the identity gate with the lowest cost. When ambiguous/no
confident match, the tracker holds its last known position (frozen
crop) rather than guessing -- same fail-safe behavior as v21.

Usage:
    python3 auto_track_target.py --clip some_clip.mp4 --reference target_reference_kai.json \\
        --output some_clip_track.json

Runs YOLO+ByteTrack itself (no separate caching step needed for a
one-off clip) -- if you're re-running against the same clip repeatedly
while iterating, pass --tracks-cache to reuse detection output across runs.
"""

import argparse
import json
import os

import cv2
import torch
import face_recognition
import numpy as np
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor
from pathlib import Path

REID_WEIGHTS = (
    "~/.cache/huggingface/hub/models--kaiyangzhou--osnet/"
    "snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)

# Same values v21 converged on, EXCEPT REID_SIM_THRESHOLD -- raised after a
# real failure: during a lean-back/occlusion moment, ByteTrack lost Kai's
# track id, ID search fell back to OSNet body-similarity alone (no face
# visible), and a same-outfit bystander qualified at the old 0.54 threshold
# once the search radius expanded during recovery. Raised to make body-only
# matches meaningfully harder to satisfy on their own.
MATCH_THRESHOLD = 0.48
REID_SIM_THRESHOLD = 0.62
SPATIAL_GATE_PX = 55
MAX_RECOVERY_PX = 400
MIN_IOU_THRESHOLD = 0.15
LOST_GRACE_FRAMES = 30
LOST_HIDE_FRAMES = 15
MAX_ASSIGN_COST = 0.6
IDENTITY_WEIGHT = 0.8
SPATIAL_WEIGHT = 0.2
VELOCITY_HOLD_FRAMES = 3

# Expensive face/ReID recovery:
# - try every lost frame for the first few frames
# - after that, retry periodically instead of on every frame
RECOVERY_INTERVAL_FRAMES = 10

EMA_ALPHA = 0.3


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    tid, x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def crop_box_torso(frame, box):
    tid, x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(x2), int(y2)
    h = y2 - y1
    torso_y1 = int(y1 + 0.10 * h)
    torso_y2 = int(y1 + 0.55 * h)
    c = frame[torso_y1:torso_y2, x1:x2]
    return c if c.size > 0 else None


def run_yolo_tracking(clip_path, tracks_cache=None):
    """Runs YOLO+ByteTrack over the whole clip, or loads cached results
    if a cache path is given and already exists."""
    if tracks_cache and os.path.exists(tracks_cache):
        print(f"Loading cached YOLO tracks from {tracks_cache} ...")
        with open(tracks_cache) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    print("Loading YOLOv8...")
    model = YOLO("yolov8n.pt")

    if torch.backends.mps.is_available():
        yolo_device = "mps"
    else:
        yolo_device = "cpu"

    print(
        f"Running detection + ByteTrack on {clip_path} "
        f"(every frame, device={yolo_device})..."
    )

    results = model.track(
        source=clip_path,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
        device=yolo_device,
    )

    all_frames = {}
    frame_idx = 0
    for r in results:
        boxes_this_frame = []
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.tolist()
            for tid, box in zip(ids, xyxy):
                boxes_this_frame.append([tid] + [round(v, 1) for v in box])
        all_frames[frame_idx] = boxes_this_frame
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  ...{frame_idx} frames processed")

    if tracks_cache:
        with open(tracks_cache, "w") as f:
            json.dump(all_frames, f)
        print(f"Cached YOLO tracks to {tracks_cache}")

    return all_frames


def find_best_seed_track(tracks_by_frame, face_refs, reid_ref, reid_extractor, cap):
    """No manual seed frame this time -- scan the first ~150 frames (or
    however many exist) for the first frame where a box confidently
    matches the reference (face match preferred, reid fallback), and
    use that as the anchor to start tracking from. Returns
    (anchor_frame, tid, pos) or (None, None, None) if nothing matched."""
    candidate_frames = sorted(tracks_by_frame.keys())[:150]

    for f in candidate_frames:
        boxes = tracks_by_frame.get(f, [])
        if not boxes:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face_locs = face_recognition.face_locations(rgb, model="hog")
        except Exception:
            face_locs = []

        for floc in face_locs:
            ftop, fright, fbottom, fleft = floc
            fcx, fcy = (fleft + fright) / 2, (ftop + fbottom) / 2
            for box in boxes:
                tid, x1, y1, x2, y2 = box
                if not (x1 <= fcx <= x2 and y1 <= fcy <= y2):
                    continue
                enc = face_recognition.face_encodings(rgb, known_face_locations=[floc])
                if enc and face_refs:
                    dist = float(np.min(face_recognition.face_distance(face_refs, enc[0])))
                    if dist <= MATCH_THRESHOLD:
                        print(f"Seed found via face match at frame {f}, track_id={tid} (dist={dist:.3f})")
                        return f, tid, box_center(box)

        if reid_ref is not None:
            for box in boxes:
                crop = crop_box_torso(frame, box)
                if crop is None:
                    continue
                feat = reid_extractor([crop])[0].cpu().numpy()
                sim = cosine_sim(reid_ref, feat)
                if sim >= REID_SIM_THRESHOLD:
                    tid = box[0]
                    print(f"Seed found via reid match at frame {f}, track_id={tid} (sim={sim:.3f})")
                    return f, tid, box_center(box)

    return None, None, None


def identity_cost(target_pos, target_face_refs, target_reid_ref, target_last_box,
                  box, frame_rgb, reid_extractor, effective_gate_px, enforce_iou,
                  cached_faces=None):
    bc = box_center(box)
    spatial_dist = ((bc[0] - target_pos[0]) ** 2 +
                    (bc[1] - target_pos[1]) ** 2) ** 0.5
    tid, x1, y1, x2, y2 = box

    if cached_faces is None:
        try:
            face_locs = face_recognition.face_locations(frame_rgb, model="hog")
        except Exception:
            face_locs = []
        cached_faces = []
        for floc in face_locs:
            try:
                encs = face_recognition.face_encodings(
                    frame_rgb, known_face_locations=[floc]
                )
            except Exception:
                encs = []
            if encs:
                cached_faces.append((floc, encs[0]))

    # Check face identity FIRST, even after a camera cut.
    for floc, face_enc in cached_faces:
        top, right, bottom, left = floc
        cx = (left + right) / 2
        cy = (top + bottom) / 2

        if x1 <= cx <= x2 and y1 <= cy <= y2 and target_face_refs:
            dist = float(np.min(
                face_recognition.face_distance(target_face_refs, face_enc)
            ))

            # Distant recovery requires a stronger face match.
            threshold = 0.43 if spatial_dist > effective_gate_px else MATCH_THRESHOLD
            if dist > threshold:
                return None

            spatial_cost = min(1.0, spatial_dist / max(effective_gate_px, 1))
            return (
                IDENTITY_WEIGHT * (dist / MATCH_THRESHOLD) * 0.5
                + SPATIAL_WEIGHT * spatial_cost
            )

    # Without a face match, retain strict spatial safeguards.
    if spatial_dist > effective_gate_px:
        return None

    if enforce_iou and target_last_box is not None:
        if compute_iou(target_last_box, (x1, y1, x2, y2)) < MIN_IOU_THRESHOLD:
            return None

    if spatial_dist > SPATIAL_GATE_PX:
        return None

    if target_reid_ref is not None:
        crop = crop_box_torso(
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), box
        )
        if crop is not None:
            feat = reid_extractor([crop])[0].cpu().numpy()
            sim = cosine_sim(target_reid_ref, feat)
            if sim < REID_SIM_THRESHOLD:
                return None
            spatial_cost = min(1.0, spatial_dist / SPATIAL_GATE_PX)
            return (
                IDENTITY_WEIGHT * (1.0 - sim)
                + SPATIAL_WEIGHT * spatial_cost
            )

    return None



def main():
    ap = argparse.ArgumentParser(description="Automated single-target tracker using a persistent identity reference.")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--reference", required=True, help="Path to target_reference_<name>.json")
    ap.add_argument("--output", required=True, help="Path to write track positions JSON")
    ap.add_argument("--tracks-cache", default=None, help="Optional path to cache/reuse raw YOLO+ByteTrack output")
    args = ap.parse_args()

    with open(args.reference) as f:
        ref = json.load(f)
    face_refs = [np.array(e) for e in ref.get("face_encodings", [])]
    reid_refs = [np.array(e) for e in ref.get("reid_embeddings", [])]
    # Average multiple reid reference embeddings into one centroid --
    # face matching already handles multiple refs natively via min-distance.
    reid_ref = np.mean(reid_refs, axis=0) if reid_refs else None

    if not face_refs and reid_ref is None:
        print(f"ERROR: reference file {args.reference} has no usable face or reid embeddings.")
        return

    print(f"Loaded reference '{ref.get('name', '?')}': {len(face_refs)} face encodings, "
          f"{len(reid_refs)} reid embeddings.")

    tracks_cache = args.tracks_cache
    if not tracks_cache:
        output_path = Path(args.output)
        tracks_cache = str(
            output_path.with_name(
                output_path.stem + "_yolo_tracks.json"
            )
        )

    print(f"YOLO tracks cache: {tracks_cache}")
    tracks_by_frame = run_yolo_tracking(args.clip, tracks_cache)
    total_frames = max(tracks_by_frame.keys()) + 1 if tracks_by_frame else 0
    if total_frames == 0:
        print("ERROR: no frames detected by YOLO tracking.")
        return

    cap = cv2.VideoCapture(args.clip)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("Loading OSNet Re-ID extractor...")
    reid_extractor = FeatureExtractor(model_name="osnet_x1_0", model_path=REID_WEIGHTS, device="cpu")

    print("Searching for an initial confident match to seed tracking...")
    anchor_frame, anchor_tid, anchor_pos = find_best_seed_track(
        tracks_by_frame, face_refs, reid_ref, reid_extractor, cap
    )
    if anchor_tid is None:
        print("ERROR: could not find a confident match to the reference anywhere in the "
              "first 150 frames. This clip may not contain the target early on, or the "
              "reference may need more/better sample frames via build_target_reference.py.")
        cap.release()
        return

    positions = {}
    lost_flags = {}
    assignments_made = 0
    velocity_holds_used = 0

    for direction, frame_range in [("forward", range(anchor_frame, total_frames)),
                                    ("backward", range(anchor_frame, -1, -1))]:
        tid = anchor_tid
        pos = anchor_pos
        last_box = None
        velocity = (0.0, 0.0)
        lost_streak = 0
        prev_confirmed_pos = pos

        for f in frame_range:
            if direction == "backward" and f in positions and f != anchor_frame:
                continue
            boxes = tracks_by_frame.get(f, [])
            match = next((b for b in boxes if b[0] == tid), None)

            if match is None:
                if lost_streak < VELOCITY_HOLD_FRAMES:
                    recovery_due = True
                elif lost_streak < 60:
                    recovery_due = (
                        lost_streak % RECOVERY_INTERVAL_FRAMES == 0
                    )
                else:
                    recovery_due = (
                        lost_streak % RECOVERY_INTERVAL_FRAMES == 0
                    )

                # No candidate boxes means identity recovery cannot succeed.
                # Likewise, once the initial short grace period passes,
                # avoid expensive face/ReID work on every single frame.
                if recovery_due and boxes:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ret, frame = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        in_grace = lost_streak >= LOST_GRACE_FRAMES
                        gate = MAX_RECOVERY_PX if in_grace else SPATIAL_GATE_PX
    
                        # Detect + encode faces ONCE for this recovery frame.
                        try:
                            recovery_face_locs = face_recognition.face_locations(
                                rgb,
                                model="hog",
                            )
                        except Exception:
                            recovery_face_locs = []
    
                        recovery_faces = []
    
                        for floc in recovery_face_locs:
                            try:
                                encs = face_recognition.face_encodings(
                                    rgb,
                                    known_face_locations=[floc],
                                )
                            except Exception:
                                encs = []
    
                            if encs:
                                recovery_faces.append(
                                    (floc, encs[0])
                                )
    
                        best_box, best_cost = None, None
    
                        for box in boxes:
                            cost = identity_cost(
                                pos,
                                face_refs,
                                reid_ref,
                                last_box,
                                box,
                                rgb,
                                reid_extractor,
                                gate,
                                enforce_iou=not in_grace,
                                cached_faces=recovery_faces,
                            )
    
                            if (
                                cost is not None
                                and (
                                    best_cost is None
                                    or cost < best_cost
                                )
                            ):
                                best_cost = cost
                                best_box = box
    
                        if best_box is not None and best_cost <= MAX_ASSIGN_COST:
                            tid = best_box[0]
                            match = best_box
                            assignments_made += 1
                            lost_streak = 0
                        else:
                            lost_streak += 1
                else:
                    lost_streak += 1

            else:
                lost_streak = 0

            if match is not None:
                new_pos = box_center(match)
                velocity = (new_pos[0] - prev_confirmed_pos[0], new_pos[1] - prev_confirmed_pos[1])
                positions[f] = new_pos
                pos = new_pos
                last_box = tuple(match[1:])
                prev_confirmed_pos = new_pos
                lost_flags[f] = False
            elif lost_streak <= VELOCITY_HOLD_FRAMES:
                predicted = (pos[0] + velocity[0], pos[1] + velocity[1])
                positions[f] = predicted
                pos = predicted
                velocity_holds_used += 1
                lost_flags[f] = False
            else:
                positions[f] = pos
                lost_flags[f] = lost_streak > LOST_HIDE_FRAMES

    cap.release()

    print(f"\nTotal successful matches: {assignments_made}")
    print(f"Frames using velocity-prediction hold: {velocity_holds_used}")
    print(f"Frames flagged as 'lost': {sum(lost_flags.values())} / {total_frames}\n")

    sorted_frames = sorted(positions.keys())
    smoothed = {}
    prev = None
    for f in sorted_frames:
        raw = positions[f]
        prev = raw if prev is None else (
            EMA_ALPHA * raw[0] + (1 - EMA_ALPHA) * prev[0],
            EMA_ALPHA * raw[1] + (1 - EMA_ALPHA) * prev[1],
        )
        smoothed[f] = prev

    output = {str(k): {"pos": smoothed[k], "lost": lost_flags[k]} for k in sorted_frames}
    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
