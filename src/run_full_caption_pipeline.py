#!/usr/bin/env python3

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    print()
    print("=" * 72)
    print("$", " ".join(str(x) for x in cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True)


def require(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )
    return path



def video_content_key(path):
    """
    Stable cache key based on the actual video bytes.
    Moving/renaming the same video still reuses ASR.
    """
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def seed_cached_asr(cache_path, run_path, label):
    cache_path = Path(cache_path)
    run_path = Path(run_path)

    if not cache_path.exists():
        return False

    run_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        cache_path,
        run_path,
    )

    print(
        f"Using persistent {label} ASR cache:",
        cache_path,
    )

    return True


def save_cached_asr(run_path, cache_path, label):
    run_path = Path(run_path)
    cache_path = Path(cache_path)

    if cache_path.exists():
        return

    if not run_path.exists():
        raise RuntimeError(
            f"Cannot cache missing {label} ASR: "
            f"{run_path}"
        )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        run_path,
        cache_path,
    )

    print(
        f"Saved persistent {label} ASR cache:",
        cache_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full autonomous caption evidence, "
            "resolution, recovery, and review pipeline."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    video = require(
        args.video,
        "Input video",
    )

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = video.stem

    timed_dir = out / "timed"
    semantic_dir = out / "semantic"
    third_turbo_dir = out / "third_pass_turbo"
    third_full_dir = out / "third_pass_full"
    recovery_dir = out / "recovery"

    consensus_json = out / "caption_consensus.json"
    resolved_initial = out / "caption_resolved_initial.json"
    resolved_final = out / "caption_resolved_final.json"

    timed_json = (
        timed_dir
        / f"{stem}_captions_transcript.json"
    )

    semantic_json = (
        semantic_dir
        / f"{stem}_captions_semantic.json"
    )

    # Persistent raw-ASR cache.
    #
    # Benchmark/output folders are disposable, but Whisper/Groq
    # segmentation can vary slightly between identical API calls.
    # Cache by VIDEO CONTENT so repeated processing of the same
    # exact source uses identical raw evidence.
    video_key = video_content_key(video)

    asr_cache_dir = (
        Path.home()
        / ".cache"
        / "chuzzzclips"
        / "asr"
        / video_key
    )

    timed_cache = (
        asr_cache_dir
        / "timed_transcribe_groq_v3.json"
    )

    semantic_cache = (
        asr_cache_dir
        / "semantic_transcribe_groq_turbo.json"
    )

    third_turbo_json = (
        third_turbo_dir
        / "third_pass_results.json"
    )

    third_full_json = (
        third_full_dir
        / "third_pass_results.json"
    )

    recovered_json = (
        recovery_dir
        / "captions_support_only.json"
    )

    recovery_asr = (
        out
        / "recovery_asr_rich.json"
    )

    print()
    print("===== FULL CAPTION PIPELINE START =====")
    print("Video:", video)
    print("Output:", out)

    # 1. Timed Whisper transcript
    seed_cached_asr(
        timed_cache,
        timed_json,
        "timed",
    )

    run([
        "python3",
        "caption_clip.py",
        "--video",
        str(video),
        "--output-dir",
        str(timed_dir),
    ])

    require(
        timed_json,
        "Timed transcript",
    )

    save_cached_asr(
        timed_json,
        timed_cache,
        "timed",
    )

    # 2. Semantic caption pass
    seed_cached_asr(
        semantic_cache,
        semantic_json,
        "semantic",
    )

    run([
        "python3",
        "caption_clip_v2.py",
        "--video",
        str(video),
        "--output-dir",
        str(semantic_dir),
    ])

    require(
        semantic_json,
        "Semantic transcript",
    )

    save_cached_asr(
        semantic_json,
        semantic_cache,
        "semantic",
    )

    # 3. Compare timed + semantic evidence
    run([
        "python3",
        "compare_caption_transcripts_v2.py",
        "--semantic",
        str(semantic_json),
        "--timed",
        str(timed_json),
        "--output",
        str(consensus_json),
    ])

    # 4. Initial resolver pass
    #
    # Use an empty third-pass file initially so the
    # resolver can identify which regions require review.
    empty_third = out / "third_pass_empty.json"
    empty_third.write_text("[]\n")

    run([
        "python3",
        "resolve_caption_text_v2.py",
        "--semantic",
        str(semantic_json),
        "--timed",
        str(timed_json),
        "--consensus",
        str(consensus_json),
        "--third-pass",
        str(empty_third),
        "--output",
        str(resolved_initial),
    ])

    # 5. Selective TURBO third pass on unresolved regions
    run([
        "python3",
        "third_pass_caption_review.py",
        "--video",
        str(video),
        "--resolved",
        str(resolved_initial),
        "--output-dir",
        str(third_turbo_dir),
        "--transcriber",
        "turbo",
    ])

    require(
        third_turbo_json,
        "Turbo third-pass results",
    )

    # 6. Selective FULL-V3 third pass on the same
    # preliminary unresolved regions.
    run([
        "python3",
        "third_pass_caption_review.py",
        "--video",
        str(video),
        "--resolved",
        str(resolved_initial),
        "--output-dir",
        str(third_full_dir),
        "--transcriber",
        "full",
    ])

    require(
        third_full_json,
        "Full-v3 third-pass results",
    )

    # 7. Final resolver pass using BOTH selective ASRs.
    #
    # This reproduces the frozen v16 architecture:
    # turbo = --third-pass
    # full-v3 = --third-pass-2
    run([
        "python3",
        "resolve_caption_text_v2.py",
        "--semantic",
        str(semantic_json),
        "--timed",
        str(timed_json),
        "--consensus",
        str(consensus_json),
        "--third-pass",
        str(third_turbo_json),
        "--third-pass-2",
        str(third_full_json),
        "--output",
        str(resolved_final),
    ])

    # 7. Dedicated rich ASR for VAD coverage recovery.
    #
    # Do NOT reuse caption_clip.py's timed transcript here.
    # Its broad segments can hide real ASR omissions.
    run([
        "python3",
        "transcribe_groq.py",
        str(video),
        str(recovery_asr),
    ])

    require(
        recovery_asr,
        "Recovery rich ASR",
    )

    # 8. VAD / ASR coverage recovery.
    run([
        "python3",
        "run_caption_recovery_pipeline.py",
        "--video",
        str(video),
        "--asr",
        str(recovery_asr),
        "--base-captions",
        str(resolved_final),
        "--output-dir",
        str(recovery_dir),
    ])

    require(
        recovered_json,
        "Recovered support-only captions",
    )

    print()
    print("=" * 72)
    print("FULL CAPTION PIPELINE COMPLETE")
    print("=" * 72)
    print("Timed:", timed_json)
    print("Semantic:", semantic_json)
    print("Consensus:", consensus_json)
    print("Initial resolved:", resolved_initial)
    print("Third pass turbo:", third_turbo_json)
    print("Third pass full:", third_full_json)
    print("Final resolved:", resolved_final)
    print("Recovery ASR:", recovery_asr)
    print("Recovery-safe output:", recovered_json)


if __name__ == "__main__":
    main()
