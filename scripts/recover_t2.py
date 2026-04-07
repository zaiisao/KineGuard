#!/usr/bin/env python3
"""
Recover failed T2 videos by running WHAM directly on the original video
(bypassing the clip extraction step that caused the failures).

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/recover_t2.py --start 0 --end 305 --gpu-id 2
    CUDA_VISIBLE_DEVICES=3 python scripts/recover_t2.py --start 305 --end 609 --gpu-id 3
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

from core.wham_inference import process_single_video

OUT_DIR = os.environ.get("KINEGUARD_T2_OUTPUT_DIR", "output/tier2_processing")
LIST_FILE = "/tmp/t2_recover_list.txt"


def get_gpu_temp(gpu_id):
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                           "--format=csv,noheader,nounits", f"--id={gpu_id}"],
                          capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip())
    except:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=609)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    retries = []
    with open(LIST_FILE) as f:
        for l in f:
            l = l.strip()
            if l:
                vid_id, orig = l.split(",", 1)
                retries.append((vid_id, orig))

    selected = retries[args.start:args.end]
    log_path = os.path.join(OUT_DIR, f"recover_gpu{args.gpu_id}_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Recovering {len(selected)} T2 videos on GPU {args.gpu_id}")

    success = 0
    fail = 0
    skip = 0
    start_time = time.time()

    for i, (vid_id, orig) in enumerate(selected):
        vid_out = os.path.join(OUT_DIR, vid_id)

        # Skip if already has LMA
        if glob.glob(os.path.join(vid_out, "**", "lma_dict_id*.npy"), recursive=True):
            skip += 1
            continue

        # Thermal check
        if i % 3 == 0:
            temp = get_gpu_temp(args.gpu_id)
            if temp >= 85:
                log(f"[THERMAL] {temp}C — pausing...")
                while get_gpu_temp(args.gpu_id) > 75:
                    time.sleep(60)
                log(f"[THERMAL] Cooled — resuming")

        elapsed = time.time() - start_time
        done = success + fail + skip
        rate = done / (elapsed / 3600) if elapsed > 60 else 0
        remaining = (len(selected) - i) / rate if rate > 0 else 0
        log(f"[{i+1}/{len(selected)}] {vid_id} (rate: {rate:.1f}/hr, ETA: {remaining:.1f}hr)")

        # Remove old failure state
        fail_marker = os.path.join(vid_out, "_FAILED")
        if os.path.exists(fail_marker):
            os.remove(fail_marker)

        try:
            # Run WHAM directly on original video — no clip extraction
            ok, msg = process_single_video(orig, vid_out, visualize=False)
            lma = glob.glob(os.path.join(vid_out, "**", "lma_dict_id*.npy"), recursive=True)
            if ok and lma:
                success += 1
                log(f"  -> OK ({len(lma)} LMA)")
            else:
                fail += 1
                # Re-mark as failed
                with open(fail_marker, "w") as f:
                    f.write(f"recover failed: {str(msg)[:200]}")
                log(f"  -> FAIL: {str(msg)[:80]}")
        except Exception as e:
            fail += 1
            with open(fail_marker, "w") as f:
                f.write(f"recover exception: {str(e)[:200]}")
            log(f"  -> ERROR: {str(e)[:80]}")

    total_hr = (time.time() - start_time) / 3600
    log(f"\nDone in {total_hr:.1f}hr: {success} recovered, {fail} failed, {skip} skipped")


if __name__ == "__main__":
    main()
