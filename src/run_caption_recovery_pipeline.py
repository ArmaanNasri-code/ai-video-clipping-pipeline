#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd):
    print()
    print("=" * 70)
    print("RUNNING:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 70)

    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
    )

    ap.add_argument(
        "--asr",
        required=True,
        help="Primary whole-clip ASR JSON",
    )

    ap.add_argument(
        "--base-captions",
        required=True,
        help="Baseline resolved captions JSON",
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact_dir = (
        outdir
        / "exact"
    )

    shifted_dir = (
        outdir
        / "shifted"
    )

    exact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shifted_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage_json = (
        outdir
        / "coverage_exact_spans.json"
    )

    exact_turbo_json = (
        exact_dir
        / "retry_results_absolute.json"
    )

    exact_full_json = (
        exact_dir
        / "fullv3_exact_results.json"
    )

    targeted_plan = (
        outdir
        / "targeted_shifted_plan.json"
    )

    shifted_turbo_json = (
        shifted_dir
        / "shifted_turbo_results.json"
    )

    shifted_full_json = (
        shifted_dir
        / "shifted_fullv3_results.json"
    )

    stable_json = (
        outdir
        / "stable_recovery_evidence.json"
    )

    annotated_json = (
        outdir
        / "captions_support_only.json"
    )

    # --------------------------------------------------
    # 1. Detect exact uncovered speech spans.
    # --------------------------------------------------

    run([
        "./clipper_v8_env/bin/python3",
        "detect_asr_coverage_gaps.py",
        "--video",
        args.video,
        "--asr",
        args.asr,
        "--output",
        coverage_json,
    ])

    coverage = json.load(
        open(coverage_json)
    )

    retry_count = int(
        coverage.get(
            "retry_window_count",
            0,
        )
    )

    if retry_count == 0:
        import shutil

        shutil.copyfile(
            args.base_captions,
            annotated_json,
        )

        print()
        print(
            "CAPTION RECOVERY PIPELINE COMPLETE"
        )
        print(
            "Coverage retry windows: 0"
        )
        print(
            "Targeted shifted windows: 0"
        )
        print(
            "Stable support items: 0"
        )
        print(
            "Final support-only captions:",
            annotated_json,
        )
        return


    # --------------------------------------------------
    # 2. Exact turbo retries.
    # --------------------------------------------------

    run([
        "python3",
        "retry_exact_turbo.py",
        "--video",
        args.video,
        "--coverage",
        coverage_json,
        "--output-dir",
        exact_dir,
    ])

    # --------------------------------------------------
    # 3. Exact full-v3 retries.
    #
    # For now run every exact retry window.
    # The evidence builder remains conservative.
    # --------------------------------------------------

    run([
        "python3",
        "retry_exact_fullv3.py",
        "--turbo-exact",
        exact_turbo_json,
        "--retry-dir",
        exact_dir,
        "--output",
        exact_full_json,
    ])

    # --------------------------------------------------
    # 4. Find local near-agreement ambiguities.
    # --------------------------------------------------

    run([
        "python3",
        "plan_targeted_shifted_retries.py",
        "--turbo-exact",
        exact_turbo_json,
        "--full-exact",
        exact_full_json,
        "--output",
        targeted_plan,
    ])

    plan = json.load(
        open(targeted_plan)
    )

    shifted_count = int(
        plan.get(
            "count",
            0,
        )
    )

    if shifted_count > 0:
        # ----------------------------------------------
        # 5. Shifted turbo.
        # ----------------------------------------------

        run([
            "python3",
            "run_shifted_turbo_retries.py",
            "--video",
            args.video,
            "--plan",
            targeted_plan,
            "--output-dir",
            shifted_dir,
        ])

        # ----------------------------------------------
        # 6. Shifted full-v3.
        # ----------------------------------------------

        run([
            "python3",
            "run_shifted_fullv3_retries.py",
            "--shifted-turbo",
            shifted_turbo_json,
            "--crop-dir",
            shifted_dir,
            "--output",
            shifted_full_json,
        ])

    else:
        # Builder expects files, so create valid empty
        # lists when there are no targeted ambiguities.
        json.dump(
            [],
            open(
                shifted_turbo_json,
                "w",
            ),
            indent=2,
        )

        json.dump(
            [],
            open(
                shifted_full_json,
                "w",
            ),
            indent=2,
        )

        print()
        print(
            "No targeted shifted retries needed."
        )

    # --------------------------------------------------
    # 7. Conservative stable recovery evidence.
    # --------------------------------------------------

    run([
        "python3",
        "build_stable_recovery_evidence_v2.py",
        "--turbo-exact",
        exact_turbo_json,
        "--full-exact",
        exact_full_json,
        "--shifted-turbo",
        shifted_turbo_json,
        "--shifted-full",
        shifted_full_json,
        "--output",
        stable_json,
    ])

    # --------------------------------------------------
    # 8. Attach evidence as SUPPORT-ONLY metadata.
    # --------------------------------------------------

    run([
        "python3",
        "apply_stable_caption_recovery.py",
        "--base",
        args.base_captions,
        "--evidence",
        stable_json,
        "--output",
        annotated_json,
    ])

    # --------------------------------------------------
    # Final summary.
    # --------------------------------------------------

    evidence = json.load(
        open(stable_json)
    )

    stable_count = sum(
        len(
            r.get(
                "stable_support",
                [],
            )
        )
        for r in evidence.get(
            "regions",
            [],
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CAPTION RECOVERY PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Coverage retry windows:",
        retry_count,
    )

    print(
        "Targeted shifted windows:",
        shifted_count,
    )

    print(
        "Stable support items:",
        stable_count,
    )

    print(
        "Final support-only captions:",
        annotated_json,
    )


if __name__ == "__main__":
    main()
