#!/usr/bin/env python3

import json
import math
import os
import shutil
import subprocess
import sys
import urllib.parse
from datetime import timedelta
from pathlib import Path


def _log(log_fn, msg):
    if log_fn:
        log_fn(
            "visual-discovery",
            msg,
        )
    else:
        print(
            "[visual-discovery]",
            msg,
        )


def _run(cmd):
    subprocess.run(
        cmd,
        check=True,
    )


def _make_resolved_manifest(
    adapter,
    vod_id,
    quality,
    output_path,
):
    master = adapter.get_vod_manifest(
        vod_id
    )

    sub_path = (
        adapter.get_quality_submanifest(
            master,
            quality=quality,
        )
    )

    base_url = getattr(
        adapter,
        "_last_manifest_base_url",
        None,
    )

    if not base_url:
        raise RuntimeError(
            "TwitchAdapter did not expose "
            "_last_manifest_base_url"
        )

    src = Path(
        sub_path
    )

    lines = (
        src.read_text()
        .splitlines()
    )

    resolved = []

    for line in lines:
        s = line.strip()

        if (
            not s
            or s.startswith("#")
        ):
            resolved.append(
                line
            )
            continue

        if s.startswith(
            (
                "http://",
                "https://",
            )
        ):
            resolved.append(
                s
            )
            continue

        resolved.append(
            urllib.parse.urljoin(
                base_url,
                s,
            )
        )

    Path(
        output_path
    ).write_text(
        "\n".join(
            resolved
        )
        + "\n"
    )

    return (
        sub_path,
        output_path,
    )


def _ensure_visual_proxy(
    vod_id,
    adapter,
    proxy_quality,
    force,
    log_fn,
    local_video=None,
):
    proxy_path = (
        f"{vod_id}_visual_proxy.mp4"
    )

    if (
        os.path.exists(proxy_path)
        and not force
    ):
        _log(
            log_fn,
            (
                "reusing visual proxy: "
                f"{proxy_path}"
            ),
        )

        return proxy_path

    resolved = (
        f"twitch_vod_"
        f"{vod_id}_"
        f"{proxy_quality}_"
        f"visual_resolved.m3u8"
    )

    _log(
        log_fn,
        (
            "building low-resolution "
            "visual proxy..."
        ),
    )

    if local_video:
        proxy_source = local_video
        _log(
            log_fn,
            (
                "using local archived video "
                f"for visual proxy: {local_video}"
            ),
        )
    else:
        _make_resolved_manifest(
            adapter,
            vod_id,
            proxy_quality,
            resolved,
        )
        proxy_source = resolved

    _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file,http,https,tcp,tls,crypto,data",
            "-i",
            proxy_source,
            "-an",
            "-vf",
            "fps=2,scale=-2:360",
            "-c:v",
            "libx264", "-pix_fmt", "yuv420p",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-y",
            proxy_path,
        ]
    )

    _log(
        log_fn,
        (
            "visual proxy ready: "
            f"{proxy_path}"
        ),
    )

    return proxy_path


def _run_scout(
    scout_script,
    proxy_path,
    events_path,
    force,
    log_fn,
    label,
):
    if (
        os.path.exists(
            events_path
        )
        and not force
    ):
        _log(
            log_fn,
            (
                f"{label} events cached"
            ),
        )

        return

    if not os.path.exists(
        scout_script
    ):
        raise RuntimeError(
            f"Missing scout: "
            f"{scout_script}"
        )

    _log(
        log_fn,
        (
            f"running {label} scout"
        ),
    )

    _run(
        [
            sys.executable,
            scout_script,
            "--video",
            proxy_path,
            "--output",
            events_path,
            "--sample-every",
            "3",
            "--top-n",
            "80",
            "--min-gap",
            "8",
        ]
    )


def _cluster(
    events_path,
    clusters_path,
    force,
    log_fn,
    label,
):
    if (
        os.path.exists(
            clusters_path
        )
        and not force
    ):
        _log(
            log_fn,
            (
                f"{label} clusters cached"
            ),
        )

        return

    _run(
        [
            sys.executable,
            "cluster_visual_events.py",
            "--input",
            events_path,
            "--output",
            clusters_path,
            "--cluster-gap",
            "25",
            "--pre-roll",
            "12",
            "--post-roll",
            "15",
            "--top-n",
            "25",
        ]
    )


def _build_union(
    v1_path,
    v3_path,
    top_per_source=20,
    max_candidates=24,
    merge_gap=10,
    max_region=90,
):
    raw = []

    for source, path in [
        ("v1", v1_path),
        ("v3", v3_path),
    ]:
        data = json.load(
            open(path)
        )

        clusters = data[
            "clusters"
        ][:top_per_source]

        for rank, c in enumerate(
            clusters,
            1,
        ):
            raw.append(
                {
                    "start": float(
                        c["start_sec"]
                    ),
                    "end": float(
                        c["end_sec"]
                    ),
                    "source": source,
                    "rank": rank,
                }
            )

    raw.sort(
        key=lambda x: x["start"]
    )

    merged = []

    for item in raw:
        if not merged:
            merged.append(
                {
                    "start": item[
                        "start"
                    ],
                    "end": item[
                        "end"
                    ],
                    "sources": {
                        item[
                            "source"
                        ]
                    },
                    "ranks": [
                        (
                            item[
                                "source"
                            ],
                            item[
                                "rank"
                            ],
                        )
                    ],
                }
            )

            continue

        cur = merged[-1]

        proposed_end = max(
            cur["end"],
            item["end"],
        )

        proposed_duration = (
            proposed_end
            - cur["start"]
        )

        if (
            item["start"]
            <= cur["end"]
            + merge_gap
            and proposed_duration
            <= max_region
        ):
            cur["end"] = (
                proposed_end
            )

            cur[
                "sources"
            ].add(
                item["source"]
            )

            cur[
                "ranks"
            ].append(
                (
                    item["source"],
                    item["rank"],
                )
            )

        else:
            merged.append(
                {
                    "start": item[
                        "start"
                    ],
                    "end": item[
                        "end"
                    ],
                    "sources": {
                        item[
                            "source"
                        ]
                    },
                    "ranks": [
                        (
                            item[
                                "source"
                            ],
                            item[
                                "rank"
                            ],
                        )
                    ],
                }
            )

    def priority(c):
        best_rank = min(
            rank
            for _, rank
            in c["ranks"]
        )

        multi_source_penalty = (
            0
            if len(
                c["sources"]
            ) > 1
            else 10
        )

        return (
            multi_source_penalty,
            best_rank,
        )

    merged.sort(
        key=priority
    )

    return merged[
        :max_candidates
    ]


def _make_proxy_candidate(
    proxy_path,
    candidate,
    index,
    candidate_dir,
):
    start = max(
        0.0,
        float(
            candidate["start"]
        ),
    )

    end = float(
        candidate["end"]
    )

    duration = max(
        1.0,
        end - start,
    )

    mins, secs = divmod(
        int(start),
        60,
    )

    sources = "+".join(
        sorted(
            candidate[
                "sources"
            ]
        )
    )

    path = os.path.join(
        candidate_dir,
        (
            f"proxy_{index:02d}_"
            f"{mins:03d}m"
            f"{secs:02d}s_"
            f"{sources}.mp4"
        ),
    )

    _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            proxy_path,
            "-t",
            f"{duration:.3f}",
            "-an",
            "-c:v",
            "libx264", "-pix_fmt", "yuv420p",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-y",
            path,
        ]
    )

    return path


def _contact_filter(
    video_path,
    output_path,
):
    _run(
        [
            sys.executable,
            "visual_contactsheet_filter.py",
            "--video",
            video_path,
            "--output",
            output_path,
        ]
    )

    return json.load(
        open(output_path)
    )


def stage_visual_discovery(
    vod_id,
    adapter,
    stream_start,
    quality,
    output_dir,
    proxy_quality="360p",
    maybe_confidence=0,
    force=False,
    log_fn=None,
    local_video=None,
):
    """
    Independent visual-only discovery branch.

    V1 motion/scene scout
        +
    V3 YOLO semantic scout
        ->
    union / dedupe
        ->
    strict contact-sheet filter
        ->
    surviving moments fetched at full quality
        ->
    copied into output_dir for the existing
    stage_visual_judge() to review.
    """

    work_dir = (
        f"{vod_id}_visual_discovery"
    )

    os.makedirs(
        work_dir,
        exist_ok=True,
    )

    candidate_dir = os.path.join(
        work_dir,
        "proxy_candidates",
    )

    filter_dir = os.path.join(
        work_dir,
        "filters",
    )

    os.makedirs(
        candidate_dir,
        exist_ok=True,
    )

    os.makedirs(
        filter_dir,
        exist_ok=True,
    )

    proxy_path = (
        _ensure_visual_proxy(
            vod_id,
            adapter,
            proxy_quality,
            force,
            log_fn,
            local_video=local_video,
        )
    )

    v1_events = os.path.join(
        work_dir,
        "v1_events.json",
    )

    v3_events = os.path.join(
        work_dir,
        "v3_events.json",
    )

    v1_clusters = os.path.join(
        work_dir,
        "v1_clusters.json",
    )

    v3_clusters = os.path.join(
        work_dir,
        "v3_clusters.json",
    )

    # V1 = original instantaneous
    # scene/motion scout.
    _run_scout(
        (
            "visual_event_scout_"
            "before_temporal_v2.py"
        ),
        proxy_path,
        v1_events,
        force,
        log_fn,
        "V1",
    )

    # Current visual_event_scout.py
    # is our V3 YOLO semantic scout.
    _run_scout(
        "visual_event_scout.py",
        proxy_path,
        v3_events,
        force,
        log_fn,
        "V3",
    )

    _cluster(
        v1_events,
        v1_clusters,
        force,
        log_fn,
        "V1",
    )

    _cluster(
        v3_events,
        v3_clusters,
        force,
        log_fn,
        "V3",
    )

    candidates = _build_union(
        v1_clusters,
        v3_clusters,
    )

    _log(
        log_fn,
        (
            "candidate union: "
            f"{len(candidates)}"
        ),
    )

    kept = []

    for i, candidate in enumerate(
        candidates,
        1,
    ):
        start = float(
            candidate["start"]
        )

        mins, secs = divmod(
            int(start),
            60,
        )

        proxy_clip = (
            _make_proxy_candidate(
                proxy_path,
                candidate,
                i,
                candidate_dir,
            )
        )

        # keyed by the moment's time (not its position in the list) and
        # reused on re-runs, so the kept set and names stay stable
        filter_path = os.path.join(
            filter_dir,
            f"candidate_{mins:03d}m{secs:02d}s_{int(float(candidate['end']))}.json",
        )

        if os.path.exists(filter_path) and not force:
            result = json.load(open(filter_path))
            _log(log_fn, f"{mins:03d}:{secs:02d} screening cached")
        else:
            result = (
                _contact_filter(
                    proxy_clip,
                    filter_path,
                )
            )

        decision = (
            result.get(
                "decision",
                "NO",
            )
        )

        confidence = int(
            result.get(
                "confidence",
                0,
            )
        )

        keep = (
            decision == "YES"
            or (
                decision == "MAYBE"
                and confidence
                >= maybe_confidence
            )
        )

        _log(
            log_fn,
            (
                f"{mins:03d}:"
                f"{secs:02d} "
                f"{decision} "
                f"{confidence} "
                f"{'KEEP' if keep else 'discard'}"
            ),
        )

        if keep:
            kept.append(
                (
                    candidate,
                    result,
                )
            )

    manifest_path = os.path.join(
        work_dir,
        "kept.json",
    )

    json.dump(
        [
            {
                "start_sec": c[
                    "start"
                ],
                "end_sec": c[
                    "end"
                ],
                "sources": sorted(
                    c["sources"]
                ),
                "filter": result,
            }
            for c, result
            in kept
        ],
        open(
            manifest_path,
            "w",
        ),
        indent=2,
    )

    if not kept:
        _log(
            log_fn,
            "no visual candidates survived",
        )

        return []

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    full_sub = None

    if not local_video:
        master = (
            adapter.get_vod_manifest(
                vod_id
            )
        )

        full_sub = (
            adapter.get_quality_submanifest(
                master,
                quality=quality,
            )
        )

    created = []

    for i, (
        candidate,
        filter_result,
    ) in enumerate(
        kept,
        1,
    ):
        start = float(
            candidate["start"]
        )

        end = float(
            candidate["end"]
        )

        center = (
            start + end
        ) / 2.0

        duration = (
            end - start
        )

        half_window = max(
            20,
            math.ceil(
                duration / 2
            ) + 6,
        )

        if local_video:
            target = None
        else:
            target = (
                stream_start
                + timedelta(
                    seconds=center
                )
            )

        mins, secs = divmod(
            int(start),
            60,
        )

        dest = os.path.join(
            output_dir,
            (
                f"visualdisc_"
                f"{i:02d}_"
                f"{mins:03d}m"
                f"{secs:02d}s.mp4"
            ),
        )

        if (
            os.path.exists(
                dest
            )
            and not force
        ):
            _log(
                log_fn,
                (
                    "already fetched: "
                    f"{dest}"
                ),
            )

            created.append(
                dest
            )

            continue

        _log(
            log_fn,
            (
                "fetching survivor "
                f"{mins:03d}:"
                f"{secs:02d} "
                f"at {quality}"
            ),
        )

        if local_video:
            # Match the Twitch fetcher's centered window semantics:
            # half_window seconds before + half_window seconds after.
            cut_start = max(
                0.0,
                center - half_window,
            )
            cut_duration = (
                half_window * 2.0
            )

            _run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(cut_start),
                    "-i",
                    local_video,
                    "-t",
                    str(cut_duration),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-c:v",
                    "libx264", "-pix_fmt", "yuv420p",
                    "-preset",
                    "veryfast",
                    "-c:a",
                    "aac",
                    "-y",
                    dest,
                ]
            )

        else:
            clip = (
                adapter.fetch_segment_window(
                    full_sub,
                    target,
                    window_seconds=(
                        half_window
                    ),
                )
            )

            shutil.move(
                clip,
                dest,
            )

        created.append(
            dest
        )

    _log(
        log_fn,
        (
            f"added {len(created)} "
            "visual-discovered clip(s) "
            "to final review queue"
        ),
    )

    return created
