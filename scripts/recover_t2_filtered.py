#!/usr/bin/env python3
"""
Recover failed T2 videos using YOLO pre-filter + WHAM.
Uses the existing filter_meta.json to know which segments have humans,
extracts just those clips, runs WHAM on the short clips.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/recover_t2_filtered.py --start 0 --end 203 --gpu-id 1
    CUDA_VISIBLE_DEVICES=2 python scripts/recover_t2_filtered.py --start 203 --end 406 --gpu-id 2
    CUDA_VISIBLE_DEVICES=3 python scripts/recover_t2_filtered.py --start 406 --end 609 --gpu-id 3
"""

import argparse
import os
import sys
import glob
import json
import time
import subprocess
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OUT_DIR = os.environ.get("KINEGUARD_T2_OUTPUT_DIR", "output/tier2_processing")
LIST_FILE = os.environ.get("KINEGUARD_T2_RECOVER_LIST", "t2_recover_list.txt")
WHAM_SCRIPT = os.path.join(REPO, "core", "wham_inference.py")
PYTHON = sys.executable


def get_gpu_temp(gpu_id):
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                           "--format=csv,noheader,nounits", f"--id={gpu_id}"],
                          capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip())
    except:
        return 0


def extract_clip(video_path, start_sec, end_sec, output_path):
    duration = end_sec - start_sec
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
         "-i", video_path, "-c:v", "libx264", "-preset", "ultrafast",
         "-crf", "23", "-an", output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def run_wham(clip_path, output_dir, timeout=600):
    result = subprocess.run(
        [PYTHON, WHAM_SCRIPT, "--video", clip_path, "--output_dir", output_dir],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode == 0


def process_video(vid_id, orig_path, out_dir, log):
    vid_out = os.path.join(out_dir, vid_id)

    # Already has LMA?
    if glob.glob(os.path.join(vid_out, "**", "lma_dict_id*.npy"), recursive=True):
        return "skipped", "already has LMA"

    # Read existing filter metadata (YOLO already ran on these)
    fm = os.path.join(vid_out, "filter_meta.json")
    if not os.path.exists(fm):
        return "failed", "no filter metadata"

    with open(fm) as f:
        meta = json.load(f)

    segments = meta.get("segments", [])
    if not segments or (isinstance(segments, int) and segments == 0):
        return "filtered_out", "no segments"

    # Remove old failure marker
    fail_marker = os.path.join(vid_out, "_FAILED")
    if os.path.exists(fail_marker):
        os.remove(fail_marker)

    # Clean old clip dirs
    for old in glob.glob(os.path.join(vid_out, "_clip_seg*")):
        shutil.rmtree(old, ignore_errors=True)

    any_success = False
    for seg_idx, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        dur = seg["duration"]

        # Skip very short segments (WHAM needs at least ~1 second)
        if dur < 2:
            continue

        clip_dir = os.path.join(vid_out, f"_clip_seg{seg_idx}")
        os.makedirs(clip_dir, exist_ok=True)
        clip_path = os.path.join(clip_dir, f"clip.mp4")

        # Extract clip
        if not extract_clip(orig_path, start, end, clip_path):
            log(f"  seg{seg_idx}: ffmpeg failed")
            continue

        # Run WHAM on clip — output goes into clip_dir/clip/
        try:
            ok = run_wham(clip_path, clip_dir, timeout=600)
        except subprocess.TimeoutExpired:
            log(f"  seg{seg_idx}: timeout ({dur:.0f}s clip)")
            os.remove(clip_path)
            continue
        except Exception as e:
            log(f"  seg{seg_idx}: error {str(e)[:60]}")
            os.remove(clip_path)
            continue

        # Clean up clip to save disk
        if os.path.exists(clip_path):
            os.remove(clip_path)

        # Check if LMA was produced (search recursively)
        lma = glob.glob(os.path.join(clip_dir, "**", "lma_dict_id*.npy"), recursive=True)
        if lma:
            any_success = True
            log(f"  seg{seg_idx}: OK ({len(lma)} LMA, {dur:.0f}s)")
        else:
            log(f"  seg{seg_idx}: WHAM ran but no LMA ({dur:.0f}s)")

    if any_success:
        return "processed", "recovered"
    else:
        with open(fail_marker, "w") as f:
            f.write("recover: WHAM failed on all segments")
        return "failed", "all segments failed"


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
    log_path = os.path.join(OUT_DIR, f"recover_filtered_gpu{args.gpu_id}_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Recovering {len(selected)} T2 videos (filtered) on GPU {args.gpu_id}")

    counts = {"processed": 0, "skipped": 0, "failed": 0, "filtered_out": 0}
    start_time = time.time()

    for i, (vid_id, orig) in enumerate(selected):
        # Thermal check
        if i % 3 == 0:
            temp = get_gpu_temp(args.gpu_id)
            if temp >= 85:
                log(f"[THERMAL] {temp}C — pausing...")
                while get_gpu_temp(args.gpu_id) > 75:
                    time.sleep(60)

        elapsed = time.time() - start_time
        done = sum(counts.values())
        rate = done / (elapsed / 3600) if elapsed > 60 else 0
        remaining = (len(selected) - i) / rate if rate > 0 else 0
        log(f"[{i+1}/{len(selected)}] {vid_id} (rate: {rate:.1f}/hr, ETA: {remaining:.1f}hr)")

        status, msg = process_video(vid_id, orig, OUT_DIR, log)
        counts[status] += 1
        if status not in ("skipped",):
            log(f"  -> {status}: {msg}")

    total_hr = (time.time() - start_time) / 3600
    log(f"\nDone in {total_hr:.1f}hr:")
    for k, v in counts.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
