#!/usr/bin/env python3
"""
Retry failed T2 videos with SLAM disabled.
WHAM runs on the full original video (no clip extraction).

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/retry_no_slam.py --list /tmp/retry_gpu2.txt --gpu-id 2
    CUDA_VISIBLE_DEVICES=3 python scripts/retry_no_slam.py --list /tmp/retry_gpu3.txt --gpu-id 3
"""

import argparse
import os
import sys
import glob
import shutil
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(os.path.join(REPO, "external/WHAM"))

from core.wham_inference import KineGuardWHAMProcessor, process_single_video

# Monkey-patch to disable SLAM
_orig_preprocess = KineGuardWHAMProcessor.preprocess_video
def _no_slam_preprocess(self, video_path, output_pth, calib=None, use_slam=True):
    return _orig_preprocess(self, video_path, output_pth, calib=calib, use_slam=False)
KineGuardWHAMProcessor.preprocess_video = _no_slam_preprocess

OUT_DIR = os.environ.get("KINEGUARD_T2_OUTPUT_DIR", "output/tier2_processing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", required=True, help="File with vid_id,path per line")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    retries = []
    with open(args.list) as f:
        for l in f:
            l = l.strip()
            if l:
                vid_id, orig = l.split(",", 1)
                retries.append((vid_id, orig))

    log_path = os.path.join(OUT_DIR, f"retry_gpu{args.gpu_id}_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Retrying {len(retries)} videos with SLAM disabled on GPU {args.gpu_id}")

    success = 0
    fail = 0
    start_time = time.time()

    for i, (vid_id, orig) in enumerate(retries):
        vid_out = os.path.join(OUT_DIR, vid_id)

        # Skip if already recovered
        if glob.glob(os.path.join(vid_out, "**", "lma_dict_id*.npy"), recursive=True):
            log(f"[{i+1}/{len(retries)}] {vid_id}: already has LMA, skipping")
            success += 1
            continue

        # Clean old state
        fail_marker = os.path.join(vid_out, "_FAILED")
        if os.path.exists(fail_marker):
            os.remove(fail_marker)
        for old in glob.glob(os.path.join(vid_out, "_clip_seg*")):
            shutil.rmtree(old, ignore_errors=True)

        elapsed = time.time() - start_time
        rate = (success + fail) / (elapsed / 3600) if elapsed > 60 else 0
        log(f"[{i+1}/{len(retries)}] {vid_id} (rate: {rate:.1f}/hr)")

        try:
            ok, msg = process_single_video(orig, vid_out, visualize=False)
            lma = glob.glob(os.path.join(vid_out, "**", "lma_dict_id*.npy"), recursive=True)
            if ok and lma:
                success += 1
                log(f"  -> OK ({len(lma)} LMA)")
            else:
                fail += 1
                log(f"  -> FAIL: {str(msg)[:80]}")
        except Exception as e:
            fail += 1
            log(f"  -> ERROR: {str(e)[:80]}")

    total_hr = (time.time() - start_time) / 3600
    log(f"\nDone in {total_hr:.1f}hr: {success} recovered, {fail} failed")


if __name__ == "__main__":
    main()
