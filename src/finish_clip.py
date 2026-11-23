#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable


def run(cmd):
    print()
    print("=" * 72)
    print("$", " ".join(str(x) for x in cmd))
    print("=" * 72)

    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def require(path, label):
    path = Path(path)

    if not path.exists():
        raise SystemExit(
            f"ERROR: {label} not found: {path}"
        )

    return path


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--video",
        required=True,
        help="Finished vertical crop before captions",
    )

    ap.add_argument(
        "--resolved",
        required=True,
        help="Resolved caption JSON",
    )

    ap.add_argument(
        "--context",
        default="",
        help="Short factual clip context for hook generation",
    )

    ap.add_argument(
        "--clip-id",
        required=True,
    )

    ap.add_argument(
        "--vod",
        required=True,
    )

    ap.add_argument(
        "--timestamp",
        required=True,
    )

    ap.add_argument(
        "--score",
        required=True,
    )

    ap.add_argument(
        "--work-dir",
        default="finish_work",
    )

    ap.add_argument(
        "--review-dir",
        default="needs_caption_review",
    )

    ap.add_argument(
        "--production-dir",
        default=None,
        help="where finalize_clip.py writes the READY_TO_POST clip "
             "(default: finalize_clip.py's own default)",
    )

    args = ap.parse_args()

    SETTINGS = {"captions": True, "hook": True, "watermark": True}
    try:
        _s = json.load(open(Path(__file__).resolve().parent / "clip_settings.json"))
        SETTINGS.update({k: bool(v) for k, v in _s.items() if k in SETTINGS})
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARNING: clip_settings.json unreadable ({e}) -- using all ON")
    print("Settings:", ", ".join(f"{k} {'ON' if v else 'OFF'}" for k, v in SETTINGS.items()))

    video = require(
        args.video,
        "video",
    )

    resolved = require(
        args.resolved,
        "resolved captions",
    )

    # Make sure all frozen finishing scripts exist.
    scripts = {
        "caption_renderer":
            Path("render_resolved_captions.py"),
        "caption_gate":
            Path("caption_quality_gate.py"),
        "hook_generator":
            Path("generate_clip_hook.py"),
        "hook_renderer":
            Path("render_clip_hook.py"),
        "watermark_renderer":
            Path("render_watermark.py"),
        "finalizer":
            Path("finalize_clip.py"),
    }

    for name, path in scripts.items():
        require(
            path,
            name,
        )

    work = Path(
        args.work_dir
    )

    captions_dir = (
        work / "captions"
    )

    hook_dir = (
        work / "hook"
    )

    watermark_dir = (
        work / "watermark"
    )

    for d in [
        captions_dir,
        hook_dir,
        watermark_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    print()
    print(
        "===== FINISH CLIP START ====="
    )
    print(
        "Input video:",
        video,
    )
    print(
        "Resolved captions:",
        resolved,
    )

    # ---------------------------------------------------------
    # 1. RENDER RESOLVED CAPTIONS
    # ---------------------------------------------------------

    if SETTINGS["captions"]:
        run([
            PYTHON,
            scripts["caption_renderer"],
            "--video",
            video,
            "--resolved",
            resolved,
            "--output-dir",
            captions_dir,
        ])

        captioned_video = (
            captions_dir
            / f"{video.stem}_resolved_captioned.mp4"
        )

        require(
            captioned_video,
            "captioned video",
        )
    else:
        print("Captions: OFF (clip_settings.json)")
        captioned_video = Path(video)

    review_report = (
        captions_dir
        / "caption_review_report.txt"
    )

    # ---------------------------------------------------------
    # 2. CAPTION QUALITY GATE
    # ---------------------------------------------------------

    gate_json = (
        captions_dir
        / "caption_gate.json"
    )

    if SETTINGS["captions"]:
        run([
            PYTHON,
            scripts["caption_gate"],
            "--resolved",
            resolved,
            "--output",
            gate_json,
        ])

        gate = json.load(
            open(
                gate_json,
                encoding="utf-8",
            )
        )
    else:
        gate = {"status": "CAPTIONS_OFF", "ready_to_post": True,
                "review_required": 0, "low_confidence": 0}
        json.dump(gate, open(gate_json, "w", encoding="utf-8"), indent=2)

    caption_status = gate[
        "status"
    ]

    captions_approved = bool(
        gate.get(
            "ready_to_post",
            False,
        )
    )

    # ---------------------------------------------------------
    # 3. GENERATE HOOK
    # ---------------------------------------------------------

    hook_json = (
        work
        / "hook_generated.json"
    )

    hook_cmd = [
        PYTHON,
        scripts["hook_generator"],
        "--resolved",
        resolved,
        "--output",
        hook_json,
    ]

    if args.context.strip():

        hook_cmd.extend([
            "--context",
            args.context.strip(),
        ])

    if SETTINGS["hook"]:
        run(
            hook_cmd
        )

        hook_data = json.load(
            open(
                hook_json,
                encoding="utf-8",
            )
        )
    else:
        print("Hook title: OFF (clip_settings.json)")
        hook_data = {}
        json.dump(hook_data, open(hook_json, "w", encoding="utf-8"))

    # ---------------------------------------------------------
    # 4. RENDER HOOK
    # ---------------------------------------------------------

    if SETTINGS["hook"]:
        run([
            PYTHON,
            scripts["hook_renderer"],
            "--video",
            captioned_video,
            "--hook-json",
            hook_json,
            "--output-dir",
            hook_dir,
        ])

        hooked_video = (
            hook_dir
            / f"{captioned_video.stem}_hooked.mp4"
        )

        require(
            hooked_video,
            "hooked video",
        )
    else:
        hooked_video = captioned_video

    # ---------------------------------------------------------
    # 5. WATERMARK
    # ---------------------------------------------------------

    if SETTINGS["watermark"]:
        run([
            PYTHON,
            scripts["watermark_renderer"],
            "--video",
            hooked_video,
            "--output-dir",
            watermark_dir,
        ])

        branded_video = (
            watermark_dir
            / f"{hooked_video.stem}_watermarked.mp4"
        )

        require(
            branded_video,
            "branded video",
        )
    else:
        print("Watermark: OFF (clip_settings.json)")
        branded_video = hooked_video

    # ---------------------------------------------------------
    # 6. FINISHING MANIFEST
    # ---------------------------------------------------------

    manifest = {
        "clip_id":
            args.clip_id,
        "vod":
            args.vod,
        "timestamp":
            args.timestamp,
        "score":
            args.score,
        "caption_status":
            caption_status,
        "caption_review_required":
            gate.get(
                "review_required",
                0,
            ),
        "caption_low_confidence":
            gate.get(
                "low_confidence",
                0,
            ),
        "hook":
            hook_data.get(
                "hook",
                "",
            ),
        "hook_confidence":
            hook_data.get(
                "confidence",
                "",
            ),
        "captions":
            SETTINGS["captions"],
        "hook_rendered":
            SETTINGS["hook"],
        "watermark":
            SETTINGS["watermark"],
        "watermark_text":
            "@chuzzzclips",
        "final_video":
            str(
                branded_video
            ),
    }

    manifest_path = (
        work
        / "finish_manifest.json"
    )

    json.dump(
        manifest,
        open(
            manifest_path,
            "w",
            encoding="utf-8",
        ),
        indent=2,
    )

    # ---------------------------------------------------------
    # 7A. APPROVED → PRODUCTION READY
    # ---------------------------------------------------------

    if captions_approved:

        print()
        print(
            "CAPTIONS APPROVED"
        )
        print(
            "Continuing to final packaging..."
        )

        run([
            PYTHON,
            scripts["finalizer"],
            "--video",
            branded_video,
            "--clip-id",
            args.clip_id,
            "--vod",
            args.vod,
            "--timestamp",
            args.timestamp,
            "--score",
            args.score,
        ] + (
            ["--output-dir", args.production_dir]
            if args.production_dir
            else []
        ))

        print()
        print(
            "===== FINISH CLIP COMPLETE ====="
        )
        print(
            "STATUS: READY_TO_POST"
        )
        print(
            "Final video:",
            branded_video,
        )
        print(
            "Manifest:",
            manifest_path,
        )

        return

    # ---------------------------------------------------------
    # 7B. REVIEW REQUIRED → DO NOT MARK READY TO POST
    # ---------------------------------------------------------

    review_root = Path(
        args.review_dir
    )

    review_package = (
        review_root
        / args.clip_id
    )

    review_package.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_video = (
        review_package
        / f"{args.clip_id}_review.mp4"
    )

    shutil.copy2(
        branded_video,
        review_video,
    )

    shutil.copy2(
        gate_json,
        review_package
        / "caption_gate.json",
    )

    shutil.copy2(
        hook_json,
        review_package
        / "hook.json",
    )

    if review_report.exists():

        shutil.copy2(
            review_report,
            review_package
            / "caption_review_report.txt",
        )

    manifest[
        "status"
    ] = "NEEDS_CAPTION_REVIEW"

    manifest[
        "ready_to_post"
    ] = False

    manifest[
        "review_video"
    ] = str(
        review_video
    )

    review_manifest = (
        review_package
        / "manifest.json"
    )

    json.dump(
        manifest,
        open(
            review_manifest,
            "w",
            encoding="utf-8",
        ),
        indent=2,
    )

    print()
    print(
        "===== FINISH CLIP COMPLETE ====="
    )
    print(
        "STATUS: NEEDS_CAPTION_REVIEW"
    )
    print(
        "Review-required regions:",
        gate.get(
            "review_required",
            0,
        ),
    )
    print(
        "Review video:",
        review_video,
    )
    print(
        "Review report:",
        review_package
        / "caption_review_report.txt",
    )
    print(
        "Manifest:",
        review_manifest,
    )
    print()
    print(
        "NOT sent to production_ready."
    )


if __name__ == "__main__":
    main()
