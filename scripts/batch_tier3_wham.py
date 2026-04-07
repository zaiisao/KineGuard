#!/usr/bin/env python3
"""
Batch WHAM + LMA processing for Tier 3 (NPDI porn) dataset.
Designed to run in tmux — survives VSCode/SSH disconnects.

Usage:
    # GPU 1 — videos 0-332
    CUDA_VISIBLE_DEVICES=1 python scripts/batch_tier3_wham.py --start 0 --end 333 --gpu-id 1

    # GPU 2 — videos 333-665
    CUDA_VISIBLE_DEVICES=2 python scripts/batch_tier3_wham.py --start 333 --end 666 --gpu-id 2

    # GPU 3 — videos 666-999
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_tier3_wham.py --start 666 --end 1000 --gpu-id 3
"""

import argparse
import os
import sys
import glob
import time
import json
import subprocess
import traceback

# Project paths
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

T3_DIR = os.environ.get("KINEGUARD_T3_DIR", "")
OUT_ROOT = os.environ.get("KINEGUARD_OUTPUT_DIR", "output/tier3_processing")


def get_gpu_temp(gpu_id):
    """Get GPU temperature. Returns 0 if query fails."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
             f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def wait_for_cool(gpu_id, threshold=82, target=75, log=print):
    """Pause processing if GPU is too hot."""
    temp = get_gpu_temp(gpu_id)
    if temp >= threshold:
        log(f"[THERMAL] GPU {gpu_id} at {temp}C (threshold {threshold}C) — pausing...")
        while temp > target:
            time.sleep(60)
            temp = get_gpu_temp(gpu_id)
            log(f"[THERMAL] GPU {gpu_id} at {temp}C, waiting for {target}C...")
        log(f"[THERMAL] GPU {gpu_id} cooled to {temp}C — resuming")


def process_video(video_path, output_dir, log=print):
    """Run WHAM + LMA on a single video. Returns (success, message)."""
    vid_id = os.path.splitext(os.path.basename(video_path))[0]
    vid_out = os.path.join(output_dir, vid_id)

    # Skip if already processed
    if glob.glob(os.path.join(vid_out, "lma_dict_id*.npy")):
        return True, "already processed"

    # Skip if previously failed with a marker file
    fail_marker = os.path.join(vid_out, "_FAILED")
    if os.path.exists(fail_marker):
        return False, "previously failed"

    os.makedirs(vid_out, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO, "core", "wham_inference.py"),
             "--video", video_path,
             "--output_dir", output_dir],
            capture_output=True, text=True,
            timeout=1800,  # 30 min timeout per video
        )

        if result.returncode != 0:
            # Write failure marker with error
            with open(fail_marker, "w") as f:
                f.write(result.stderr[-500:] if result.stderr else "unknown error")
            return False, f"WHAM failed (exit {result.returncode})"

        # Check if LMA was produced
        if glob.glob(os.path.join(vid_out, "lma_dict_id*.npy")):
            return True, "OK"
        else:
            # WHAM ran but no LMA (no valid fragments detected)
            with open(fail_marker, "w") as f:
                f.write("no valid fragments / no LMA output")
            return False, "no LMA output"

    except subprocess.TimeoutExpired:
        with open(fail_marker, "w") as f:
            f.write("timeout (30 min)")
        return False, "timeout"
    except Exception as e:
        with open(fail_marker, "w") as f:
            f.write(str(e))
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=1000, help="End index (exclusive)")
    parser.add_argument("--gpu-id", type=int, default=0, help="Physical GPU ID for temp monitoring")
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Get all video files sorted
    all_videos = sorted(glob.glob(os.path.join(T3_DIR, "*.mp4")))
    selected = all_videos[args.start:args.end]

    # Log file per GPU
    log_path = os.path.join(OUT_ROOT, f"gpu{args.gpu_id}_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Starting batch: videos {args.start}-{args.end} ({len(selected)} videos) on GPU {args.gpu_id}")
    log(f"Output: {OUT_ROOT}")

    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for i, video_path in enumerate(selected):
        vid_id = os.path.splitext(os.path.basename(video_path))[0]

        # Thermal check every 5 videos
        if i % 5 == 0:
            wait_for_cool(args.gpu_id, log=log)

        # Progress
        elapsed = time.time() - start_time
        rate = (processed + skipped + failed) / (elapsed / 3600) if elapsed > 0 else 0
        remaining = (len(selected) - i) / rate if rate > 0 else 0
        log(f"[{i+1}/{len(selected)}] {vid_id} (rate: {rate:.1f} vid/hr, ETA: {remaining:.1f} hr)")

        success, msg = process_video(video_path, OUT_ROOT, log=log)

        if success and msg == "already processed":
            skipped += 1
        elif success:
            processed += 1
            log(f"  -> {msg}")
        else:
            failed += 1
            log(f"  -> FAILED: {msg}")

    total_time = (time.time() - start_time) / 3600
    log(f"\nBatch complete: {processed} processed, {skipped} skipped, {failed} failed in {total_time:.1f} hours")


if __name__ == "__main__":
    main()
