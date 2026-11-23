# Evaluation

How clip quality and detection are measured. The aim is to separate three different failures:
**no good moments existed**, **good moments existed but weren't detected**, and **detected but wrongly rejected**.

## 1. Hand-labelled clip set (precision)

46 candidate clips from two VODs, each labelled by hand in a small browser labelling page (data built by `src/build_eval_set.py`):

- **label:** strong / borderline / weak
- 1–5 ratings for **hook** (does it stop scrolling in 2 s?), **payoff** (funny / surprising / satisfying?) and **context** (does a new viewer get it?)
- a final call: auto-post / needs editing / reject, plus the best trim range

Any change to the judge is re-run on this set with **both the old and the new prompt on identical inputs** (same frames, same transcript). That way a difference comes from the prompt alone (`src/judge_variant_test.py`). Clips whose original video had been cleaned up are re-cut from the VOD using the same windows (`src/recut_eval_clips.py`).

**Acceptance rule:** a judge change is rejected if any clip labelled *weak* passes the approval threshold.

| Label | Old prompt passing | New prompt passing |
|---|---|---|
| weak (18) | 0 | 0 |
| borderline (25) | 3 | 5 |
| strong (3) | 1 | 2 |

## 2. Viewer-clip recall audit (recall)

`src/viewer_clip_check.py` fetches the clips viewers made for each finished VOD (Twitch Helix `/clips`, with each clip's offset into the VOD). For the top N by views, it checks whether the pipeline had a candidate covering that moment, and what happened to it:

- **APPROVED / REVIEW:** found and finished
- **REJECTED:** found, but the judge said no (with the judge's score and reason)
- **MISSED:** no candidate near it (a detection gap)

| | Found & approved | Found & rejected | Never found |
|---|---|---|---|
| Baseline: 120 viewer clips, 12 VODs | 4 % | 21 % | 72 % |
| After viewer-clip candidates + revised judge: 30 clips, 3 VODs | 13 % | 50 % | 37 % |

Caveat: view counts measure what the *live* audience liked. Some popular moments are inside jokes or unsuitable for TikTok, so a "rejected" outcome isn't automatically wrong. The judge is told viewers clipped a moment only as a hint, never as a reason to approve.

## 3. Tracking diagnostics

- `src/evaluate_tracking.py` and `src/compare_tracks.py`: per-frame comparison of tracker outputs (lost frames, identity switches, jumps).
- `src/diagnose_tracking.py`: renders debug overlays for frame-by-frame review.
- `src/bench_faces.py`: compares face backends (InsightFace vs dlib) on speed and identity separation.

## 4. Pacing analysis

`src/analyze_retention.py` measures loudness and on-screen motion every half second of a finished clip, combines them into an energy curve, and reports slow setups, weak endings and a suggested tighter window. It's a read-only check on whether trimming leaves dead time at either end.
