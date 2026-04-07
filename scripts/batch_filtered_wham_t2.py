#!/usr/bin/env python3
"""
Batch WHAM + LMA processing with YOLO pre-filtering for Tier 2 videos.
Reads video paths from a file list instead of a hardcoded directory.

Usage (in tmux):
    CUDA_VISIBLE_DEVICES=2 python scripts/batch_filtered_wham_t2.py --start 0 --end 458 --gpu-id 2
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_filtered_wham_t2.py --start 458 --end 916 --gpu-id 3
"""

import argparse
import os
import sys

# Reuse the filtered WHAM logic from the T3 script
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Import everything from the existing batch_filtered_wham module
# but override the paths
from scripts.batch_filtered_wham import (
    process_video, wait_for_cool, YOLO_MODEL_PATH,
)
import glob
import time

VIDEO_LIST = os.environ.get("KINEGUARD_T2_VIDEO_LIST", "t2_video_list.txt")
OUT_ROOT = os.environ.get("KINEGUARD_T2_OUTPUT_DIR", "output/tier2_processing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=916)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Load YOLO model
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)

    # Read video list
    with open(VIDEO_LIST) as f:
        all_videos = [l.strip() for l in f if l.strip()]
    selected = all_videos[args.start:args.end]

    log_path = os.path.join(OUT_ROOT, f"gpu{args.gpu_id}_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Starting T2 filtered batch: videos {args.start}-{args.end} ({len(selected)} videos) on GPU {args.gpu_id}")
    log(f"Output: {OUT_ROOT}")

    counts = {"processed": 0, "skipped": 0, "failed": 0, "filtered_out": 0}
    start_time = time.time()

    for i, video_path in enumerate(selected):
        vid_id = os.path.splitext(os.path.basename(video_path))[0]

        if i % 3 == 0:
            wait_for_cool(args.gpu_id, log=log)

        elapsed = time.time() - start_time
        done = sum(counts.values())
        rate = done / (elapsed / 3600) if elapsed > 0 else 0
        remaining = (len(selected) - i) / rate if rate > 0 else 0
        log(f"[{i+1}/{len(selected)}] {vid_id} (rate: {rate:.1f} vid/hr, ETA: {remaining:.1f} hr)")

        status, msg = process_video(video_path, OUT_ROOT, model, log=log)
        counts[status] += 1
        if status != "skipped":
            log(f"  -> {status}: {msg}")

    total_time = (time.time() - start_time) / 3600
    log(f"\nBatch complete in {total_time:.1f} hours:")
    for k, v in counts.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
