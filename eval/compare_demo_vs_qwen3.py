#!/usr/bin/env python3
"""
End-to-end comparison: DEMO (motion LLM) vs Qwen3-VL on a single video.

Pipeline
--------
  1. WHAM     — extract 3D skeleton from video         → wham_fragment_*.npz
  2. DEMO     — caption skeleton, attempt tier label   → almost always fails
  3. Qwen3-VL — classify directly from video pixels   → correct tier label

Purpose
-------
Demonstrates that off-the-shelf motion LLMs (DEMO, trained on AMASS/HumanML3D)
have no dance/NSFW vocabulary and cannot serve as content classifiers out of the
box. Qwen3-VL succeeds using visual appearance alone. This motivates KineGuard:
a skeleton-based classifier trained specifically on dance/NSFW movement data.

Transformers versions
---------------------
DEMO requires transformers==4.44.0; Qwen3-VL requires transformers==4.57.6.
This script handles the upgrade/restore automatically around the Qwen3 stage.

Usage
-----
    conda run -n wham python eval/compare_demo_vs_qwen3.py \\
        /path/to/video.mp4 \\
        --qwen3_repo /path/to/Qwen3-VL-Embedding \\
        [--output_dir results/]  \\
        [--demo_model /tmp/DEMO/weights/stage2] \\
        [--demo_repo /tmp/DEMO] \\
        [--skip_wham]
"""

import argparse
import csv
import json
import os
import subprocess
import sys

PYTHON = sys.executable  # stay inside the wham conda env


# ── helpers ────────────────────────────────────────────────────────────────────

def _run(label, cmd, **kwargs):
    """Run a subprocess, printing a stage header. Exits on failure."""
    print(f"\n{'─'*70}")
    print(f"  [{label}]")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n[!] Stage '{label}' failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def _pip(pkg):
    print(f"  pip install {pkg}")
    subprocess.run([PYTHON, "-m", "pip", "install", "-q", pkg], check=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DEMO vs Qwen3-VL end-to-end comparison on a single video."
    )
    parser.add_argument("video", help="Input video file (.mp4, .webm, …)")
    parser.add_argument(
        "--qwen3_repo", required=True,
        help="Path to cloned Qwen3-VL-Embedding repo"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Root output directory (default: <video_stem>_eval/)"
    )
    parser.add_argument("--demo_model", default="/tmp/DEMO/weights/stage2")
    parser.add_argument("--demo_repo", default="/tmp/DEMO")
    parser.add_argument(
        "--skip_wham", action="store_true",
        help="Skip WHAM if wham_fragment_*.npz files already exist in the output dir"
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video = os.path.abspath(args.video)
    video_stem = os.path.splitext(os.path.basename(video))[0]

    out = os.path.abspath(args.output_dir or f"{video_stem}_eval")
    os.makedirs(out, exist_ok=True)

    # WHAM writes to {wham_root}/{video_stem}/ — mirror that here
    wham_root = os.path.join(out, "wham")
    wham_video_dir = os.path.join(wham_root, video_stem)
    demo_json = os.path.join(out, "demo_results.json")
    qwen3_csv = os.path.join(out, "qwen3_results.csv")

    # ── Stage 1: WHAM ─────────────────────────────────────────────────────
    npz_exists = os.path.isdir(wham_video_dir) and any(
        f.startswith("wham_fragment_") and f.endswith(".npz")
        for f in os.listdir(wham_video_dir)
    )
    if args.skip_wham and npz_exists:
        print(f"\n[*] Skipping WHAM — fragments already exist in {wham_video_dir}")
    else:
        _run("WHAM — 3D skeleton extraction", [
            PYTHON,
            os.path.join(repo_root, "core", "wham_inference.py"),
            "--video", video,
            "--output_dir", wham_root,
        ])

    if not os.path.isdir(wham_video_dir):
        print(f"[!] WHAM output directory not found: {wham_video_dir}")
        sys.exit(1)

    # ── Stage 2: DEMO (transformers 4.44.0) ───────────────────────────────
    _run("DEMO — motion captioning + tier classification", [
        PYTHON,
        os.path.join(repo_root, "core", "demo_motion_classifier.py"),
        wham_video_dir,
        "--model_path", args.demo_model,
        "--demo_repo", args.demo_repo,
        "--output", demo_json,
    ])

    # ── Stage 3: Qwen3-VL (needs transformers 4.57.6) ─────────────────────
    print(f"\n{'─'*70}")
    print(f"  [Qwen3 — upgrading transformers to 4.57.6]")
    print(f"{'─'*70}")
    _pip("transformers==4.57.6")

    try:
        _run("Qwen3-VL — visual tier classification", [
            PYTHON,
            os.path.join(repo_root, "core", "qwen3_vl_tier_classifier.py"),
            video,
            "--qwen3-vl-repo", args.qwen3_repo,
            "--csv", qwen3_csv,
        ])
    finally:
        print(f"\n{'─'*70}")
        print(f"  [Qwen3 — restoring transformers to 4.44.0]")
        print(f"{'─'*70}")
        _pip("transformers==4.44.0")

    # ── Comparison table ──────────────────────────────────────────────────
    _print_comparison(video_stem, demo_json, qwen3_csv)


def _print_comparison(video_stem, demo_json, qwen3_csv):
    # Parse DEMO results
    demo_tier = "N/A"
    demo_caption = "(no output)"
    if os.path.exists(demo_json):
        with open(demo_json) as f:
            demo_data = json.load(f)
        if demo_data:
            entry = demo_data[0]
            t = entry.get("final_tier", -1)
            demo_tier = str(t) if t >= 0 else "N/A (unparseable)"
            frags = entry.get("fragments", [])
            demo_caption = frags[0]["caption"] if frags else "(too short / no caption)"

    # Parse Qwen3 results
    qwen3_tier = "N/A"
    qwen3_sims: dict = {}
    if os.path.exists(qwen3_csv):
        with open(qwen3_csv, newline="") as f:
            for row in csv.DictReader(f):
                qwen3_tier = row.get("best_tier", "N/A")
                qwen3_sims = {
                    k.replace("sim_", ""): float(row[k])
                    for k in row if k.startswith("sim_")
                }
                break  # single video

    print(f"\n\n{'='*70}")
    print(f"  COMPARISON  —  {video_stem}")
    print(f"{'='*70}\n")

    print(f"  {'Method':<14} {'Tier':^20} Details")
    print(f"  {'─'*14} {'─'*20} {'─'*32}")

    cap_preview = demo_caption[:60] + ("…" if len(demo_caption) > 60 else "")
    print(f"  {'DEMO':<14} {demo_tier:^20} \"{cap_preview}\"")

    sims_str = "  ".join(f"{k}={v:.3f}" for k, v in qwen3_sims.items())
    print(f"  {'Qwen3-VL':<14} {qwen3_tier:^20} {sims_str}")

    print(f"\n  {'─'*70}")
    print(f"  WHY DEMO FAILS")
    print(f"  {'─'*70}")
    print(f"  DEMO was trained on AMASS / HumanML3D — everyday activities and")
    print(f"  sports with zero dance or NSFW vocabulary. It maps every dance")
    print(f"  posture to the closest thing it knows: 'military crawl backwards',")
    print(f"  'swimming movements', 'laying down on back'. Tier classification")
    print(f"  fails because these phrases match no meaningful safety category.")
    print(f"\n  WHY THIS MATTERS FOR KINEGUARD")
    print(f"  {'─'*70}")
    print(f"  Qwen3-VL classifies from pixels — it works, but it's a heavyweight")
    print(f"  vision-language model (2B params). KineGuard's goal is a lightweight")
    print(f"  skeleton-based classifier that bridges this gap: dance/NSFW-aware,")
    print(f"  privacy-preserving, and deployable without raw video.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
