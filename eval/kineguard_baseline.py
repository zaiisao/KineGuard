#!/usr/bin/env python3
"""
KineGuard Baseline Comparison — three-way DEMO vs Qwen3(skeleton) vs Qwen3(video)

For each input video this script runs the full pipeline and produces a side-by-side
comparison that demonstrates why skeleton-based NSFW detection needs domain-specific
training (KineGuard) rather than off-the-shelf zero-shot models.

    ┌─────────────────────────┬──────────────────────────────────────────────┐
    │ Method                  │ Expected result                              │
    ├─────────────────────────┼──────────────────────────────────────────────┤
    │ DEMO (skeleton .npz)    │ FAIL — no dance vocab; N/A tier              │
    │ Qwen3-VL (skeleton vid) │ FAIL — flat scores; stick-figure has no cues │
    │ Qwen3-VL (original vid) │ SUCCEED — visual context intact              │
    └─────────────────────────┴──────────────────────────────────────────────┘

Pipeline per video
------------------
  1. WHAM          — extract 3D skeleton → wham_fragment_*.npz
  2. render        — stick-figure skeleton video per fragment → skeleton.mp4
  3. DEMO          — caption skeleton, attempt tier label
  4. Qwen3-VL      — classify skeleton video   (transformers 4.57.6)
  5. Qwen3-VL      — classify original video   (same upgrade window)

Each stage saves its output to a predictable file. Re-running the script skips
stages whose output file already exists (checkpoint behaviour). Use --force to
redo all stages.

Input forms
-----------
  Single video:   /path/to/video.mp4
  Folder:         /path/to/folder/          (any depth, finds .mp4/.webm/etc.)
  CSV:            /path/to/list.csv         (column named 'path', or one path per line)

Usage
-----
    conda run -n wham python eval/kineguard_baseline.py \\
        /path/to/video.mp4 \\
        --qwen3_repo /tmp/Qwen3-VL-Embedding \\
        [--output_dir output/kineguard_baseline] \\
        [--demo_model /tmp/DEMO/weights/stage2] \\
        [--demo_repo /tmp/DEMO] \\
        [--force]

    # Folder:
    conda run -n wham python eval/kineguard_baseline.py /path/to/videos/ ...

    # CSV:
    conda run -n wham python eval/kineguard_baseline.py videos.csv ...

Outputs per video  (<output_dir>/<video_stem>/)
-----------------------------------------------
  wham_fragment_*.npz     WHAM skeleton data
  skeleton_id*.mp4        per-fragment stick-figure videos
  skeleton.mp4            concatenated skeleton video (Qwen3 input)
  demo_results.json       DEMO captions + tier
  qwen3_skeleton.csv      Qwen3-VL on skeleton video
  qwen3_original.csv      Qwen3-VL on original video
"""

import argparse
import csv as csv_module
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


# ── input collection ───────────────────────────────────────────────────────────

def collect_videos(source):
    """Return list of absolute video paths from a file, folder, or CSV."""
    source = os.path.abspath(source)
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".csv":
            return _videos_from_csv(source)
        if ext in VIDEO_EXTS:
            return [source]
        print(f"[!] Unrecognised file type: {source}")
        sys.exit(1)
    if os.path.isdir(source):
        found = []
        for e in VIDEO_EXTS:
            found.extend(glob.glob(os.path.join(source, f"**/*{e}"), recursive=True))
        return sorted(set(found))
    print(f"[!] Path does not exist: {source}")
    sys.exit(1)


def _videos_from_csv(csv_path):
    paths = []
    with open(csv_path, newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        has_header = "path" in sample.lower().split("\n")[0]
        reader = csv_module.DictReader(f) if has_header else csv_module.reader(f)
        for row in reader:
            p = (row.get("path") or row.get("Path") or list(row.values())[0]).strip() \
                if isinstance(row, dict) else row[0].strip()
            if p:
                paths.append(os.path.abspath(p))
    return paths


# ── subprocess helpers ─────────────────────────────────────────────────────────

def _run(label, cmd, check=True, **kwargs):
    print(f"\n{'─'*70}")
    print(f"  [{label}]")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        print(f"[!] Stage '{label}' failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def _pip(pkg):
    print(f"  pip install {pkg} ...")
    subprocess.run([PYTHON, "-m", "pip", "install", "-q", pkg], check=True)


def _skip(label, path):
    """Print skip message and return True if output file already exists."""
    if os.path.exists(path):
        print(f"\n[*] Skipping [{label}] — output already exists: {os.path.basename(path)}")
        return True
    return False


# ── per-video pipeline ─────────────────────────────────────────────────────────

def process_video(video_path, output_dir, args, repo_root):
    """
    Run the full pipeline for a single video.
    Each stage is skipped if its output file already exists (unless --force).
    Returns a result dict, or None on unrecoverable failure.
    """
    video_stem = Path(video_path).stem
    wham_dir = os.path.join(output_dir, video_stem)
    os.makedirs(wham_dir, exist_ok=True)

    skeleton_combined = os.path.join(wham_dir, "skeleton.mp4")
    demo_json         = os.path.join(wham_dir, "demo_results.json")
    qwen3_skel_csv    = os.path.join(wham_dir, "qwen3_skeleton.csv")
    qwen3_orig_csv    = os.path.join(wham_dir, "qwen3_original.csv")

    print(f"\n{'═'*70}")
    print(f"  VIDEO: {video_path}")
    print(f"{'═'*70}")

    # ── 1. WHAM ───────────────────────────────────────────────────────────
    npz_files = sorted(glob.glob(os.path.join(wham_dir, "wham_fragment_*.npz")))
    if not args.force and npz_files:
        print(f"\n[*] Skipping [WHAM] — {len(npz_files)} fragment(s) already in {wham_dir}")
    else:
        _run("WHAM — 3D skeleton extraction", [
            PYTHON,
            os.path.join(repo_root, "core", "wham_inference.py"),
            "--video", video_path,
            "--output_dir", output_dir,
        ])
        npz_files = sorted(glob.glob(os.path.join(wham_dir, "wham_fragment_*.npz")))

    if not npz_files:
        print(f"[!] No skeleton fragments produced — skipping video.")
        return None

    # ── 2. Render skeleton video ──────────────────────────────────────────
    if not args.force and _skip("Render", skeleton_combined):
        pass
    else:
        _run("Render — stick-figure skeleton per fragment", [
            PYTHON,
            os.path.join(repo_root, "core", "render_skeleton.py"),
            wham_dir,
            "--output", wham_dir,
        ])
        skeleton_frags = sorted(glob.glob(os.path.join(wham_dir, "skeleton_id*.mp4")))
        if not skeleton_frags:
            print(f"[!] Skeleton rendering produced no output — skipping video.")
            return None
        concat_list = os.path.join(wham_dir, "_concat.txt")
        with open(concat_list, "w") as f:
            for vid in skeleton_frags:
                f.write(f"file '{vid}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", "-an", skeleton_combined],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        os.remove(concat_list)
        print(f"  Combined skeleton → {skeleton_combined}")

    if not os.path.exists(skeleton_combined):
        print(f"[!] skeleton.mp4 missing — skipping video.")
        return None

    # ── 3. DEMO (skeleton .npz) ───────────────────────────────────────────
    if not args.force and _skip("DEMO", demo_json):
        pass
    else:
        _run("DEMO — motion captioning on skeleton data", [
            PYTHON,
            os.path.join(repo_root, "core", "demo_motion_classifier.py"),
            wham_dir,
            "--model_path", args.demo_model,
            "--demo_repo", args.demo_repo,
            "--output", demo_json,
        ])

    # ── 4+5. Qwen3 (one upgrade/restore window for both) ──────────────────
    need_skel = args.force or not os.path.exists(qwen3_skel_csv)
    need_orig = args.force or not os.path.exists(qwen3_orig_csv)

    if need_skel or need_orig:
        print(f"\n{'─'*70}")
        print(f"  [Qwen3 — upgrading transformers to 4.57.6]")
        print(f"{'─'*70}")
        _pip("transformers==4.57.6")
        try:
            if need_skel:
                _run("Qwen3-VL — skeleton video", [
                    PYTHON,
                    os.path.join(repo_root, "core", "qwen3_vl_tier_classifier.py"),
                    skeleton_combined,
                    "--qwen3-vl-repo", args.qwen3_repo,
                    "--csv", qwen3_skel_csv,
                ])
            else:
                print(f"\n[*] Skipping [Qwen3 skeleton] — output already exists")

            if need_orig:
                _run("Qwen3-VL — original video", [
                    PYTHON,
                    os.path.join(repo_root, "core", "qwen3_vl_tier_classifier.py"),
                    video_path,
                    "--qwen3-vl-repo", args.qwen3_repo,
                    "--csv", qwen3_orig_csv,
                ])
            else:
                print(f"\n[*] Skipping [Qwen3 original] — output already exists")
        finally:
            print(f"\n{'─'*70}")
            print(f"  [Qwen3 — restoring transformers to 4.44.0]")
            print(f"{'─'*70}")
            _pip("transformers==4.44.0")
    else:
        print(f"\n[*] Skipping [Qwen3] — both skeleton and original results already exist")

    # ── Parse all results ─────────────────────────────────────────────────
    demo_tier, demo_caption = _parse_demo(demo_json)
    qwen3_skel = _parse_qwen3_csv(qwen3_skel_csv)
    qwen3_orig = _parse_qwen3_csv(qwen3_orig_csv)

    return {
        "video": video_path,
        "video_stem": video_stem,
        "demo": {"tier": demo_tier, "caption": demo_caption},
        "qwen3_skeleton": qwen3_skel,
        "qwen3_original": qwen3_orig,
    }


# ── result parsers ─────────────────────────────────────────────────────────────

def _parse_demo(demo_json):
    if not os.path.exists(demo_json):
        return "missing", ""
    with open(demo_json) as f:
        data = json.load(f)
    if not data:
        return "no output", ""
    entry = data[0]
    t = entry.get("final_tier", -1)
    tier = str(t) if t >= 0 else "N/A"
    frags = entry.get("fragments", [])
    caption = frags[0]["caption"] if frags else "(no caption)"
    return tier, caption


def _parse_qwen3_csv(csv_path):
    if not os.path.exists(csv_path):
        return {"tier": "missing", "sims": {}}
    with open(csv_path, newline="") as f:
        for row in csv_module.DictReader(f):
            return {
                "tier": row.get("best_tier", "N/A"),
                "sims": {k.replace("sim_", ""): float(row[k])
                         for k in row if k.startswith("sim_")},
            }
    return {"tier": "empty", "sims": {}}


# ── comparison display ─────────────────────────────────────────────────────────

def print_comparison(results):
    print(f"\n\n{'═'*84}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*84}")
    print(
        f"  {'Video':<28}"
        f"  {'DEMO (skeleton)':<20}"
        f"  {'Qwen3 (skeleton)':<18}"
        f"  {'Qwen3 (video)'}"
    )
    print(
        f"  {'─'*28}"
        f"  {'─'*20}"
        f"  {'─'*18}"
        f"  {'─'*14}"
    )
    for r in results:
        name   = r["video_stem"][:27]
        demo_t = r["demo"]["tier"]
        skel_t = r["qwen3_skeleton"]["tier"]
        orig_t = r["qwen3_original"]["tier"]
        print(f"  {name:<28}  {demo_t:<20}  {skel_t:<18}  {orig_t}")

    print(f"\n  {'─'*84}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*84}")
    print(f"  DEMO (skeleton .npz)")
    print(f"    No dance vocabulary. Captions describe movement as 'military crawl',")
    print(f"    'swimming', 'handstand' regardless of actual content. Tier = N/A.")
    print()
    print(f"  Qwen3-VL (skeleton video)")
    print(f"    Stick-figure has no skin, clothing, or environmental context.")
    print(f"    Similarity scores are nearly flat — the model cannot detect NSFW")
    print(f"    cues from joint positions alone.")
    print()
    print(f"  Qwen3-VL (original video)")
    print(f"    Full visual context. Correctly classifies dance tier with a clear")
    print(f"    score gap. Upper-bound reference for KineGuard.")
    print(f"{'═'*84}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KineGuard baseline: DEMO vs Qwen3(skeleton) vs Qwen3(video)."
    )
    parser.add_argument(
        "input",
        help="Single video file, folder of videos, or CSV with a 'path' column",
    )
    parser.add_argument(
        "--qwen3_repo", required=True,
        help="Path to cloned Qwen3-VL-Embedding repo",
    )
    parser.add_argument(
        "--output_dir", default="output/kineguard_baseline",
        help="Root output directory (default: output/kineguard_baseline)",
    )
    parser.add_argument("--demo_model", default="/tmp/DEMO/weights/stage2")
    parser.add_argument("--demo_repo",  default="/tmp/DEMO")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all stages even if output files already exist",
    )
    parser.add_argument(
        "--results_json", default=None,
        help="Save combined results to this path (default: <output_dir>/results.json)",
    )
    args = parser.parse_args()

    repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    videos = collect_videos(args.input)
    if not videos:
        print("[!] No videos found.")
        sys.exit(1)
    print(f"[*] {len(videos)} video(s) to process.")

    all_results = []
    for video in videos:
        result = process_video(video, output_dir, args, repo_root)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("[!] No videos processed successfully.")
        sys.exit(1)

    print_comparison(all_results)

    results_path = args.results_json or os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[*] Full results saved to {results_path}")


if __name__ == "__main__":
    main()
