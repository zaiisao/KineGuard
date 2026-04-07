#!/usr/bin/env python3
"""Render WHAM .npz joint data as a stick-figure skeleton video.

Two modes:
  1. Standalone (default): skeleton on plain dark background, bias-free.
  2. Overlay (--video): skeleton drawn on top of the original video.

Usage:
    python render_skeleton.py output/dir/wham_fragment_id0.npz                     # standalone
    python render_skeleton.py output/dir/ --video original.mp4                     # overlay
    python render_skeleton.py output/dir/ --video original.mp4 -o overlay.mp4      # overlay with output path
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np


def reencode_h264(path):
    """Re-encode an mp4v video to H.264 for browser/IDE compatibility."""
    tmp = path + ".tmp.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-preset", "fast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-an", tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.replace(tmp, path)

# WHAM outputs 31 joints in a hybrid COCO+Common format.
# We use the first 17 (COCO keypoints) which are well-defined:
#  0: nose,  1: leye,  2: reye,  3: lear,  4: rear,
#  5: lshoulder,  6: rshoulder,  7: lelbow,  8: relbow,
#  9: lwrist, 10: rwrist, 11: lhip, 12: rhip,
# 13: lknee, 14: rknee, 15: lankle, 16: rankle
NUM_JOINTS = 17

# Skeleton bones as (joint_a, joint_b) pairs
BONES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes-ears
    # Torso
    (5, 6),   # shoulder to shoulder
    (5, 11),  # left shoulder to left hip
    (6, 12),  # right shoulder to right hip
    (11, 12), # hip to hip
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]

# Colors (BGR)
COLOR_TORSO = (200, 200, 200)
COLOR_HEAD  = (180, 180, 180)
COLOR_LEFT  = (255, 160, 80)   # blue-ish for left side
COLOR_RIGHT = (80, 160, 255)   # orange-ish for right side

BONE_COLORS = {}
for a, b in BONES:
    bone = (min(a,b), max(a,b))
    if a in (0,1,2,3,4) and b in (0,1,2,3,4):
        BONE_COLORS[bone] = COLOR_HEAD
    elif bone in ((5,6), (5,11), (6,12), (11,12)):
        BONE_COLORS[bone] = COLOR_TORSO
    elif a in (5,7,9,11,13,15) or b in (5,7,9,11,13,15):
        BONE_COLORS[bone] = COLOR_LEFT
    else:
        BONE_COLORS[bone] = COLOR_RIGHT

# Index of the mid-torso (average of hips) for centering
HIP_LEFT, HIP_RIGHT = 11, 12


def align_to_front(joints_3d):
    """Rotate each frame so the skeleton faces the camera (front view).

    Uses the left-hip (11) → right-hip (12) vector to determine facing direction,
    then rotates around the Y (up) axis so the person faces +Z.
    """
    result = np.zeros_like(joints_3d)
    for t in range(len(joints_3d)):
        j = joints_3d[t]
        hip_vec = j[HIP_RIGHT] - j[HIP_LEFT]  # lateral direction
        # Facing direction is perpendicular to hip vector in XZ plane
        facing = np.array([-hip_vec[2], 0, hip_vec[0]])
        facing_len = np.linalg.norm(facing)
        if facing_len < 1e-6:
            result[t] = j
            continue
        facing = facing / facing_len
        angle = np.arctan2(facing[0], facing[2])
        c, s = np.cos(-angle), np.sin(-angle)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        result[t] = j @ Ry.T
    return result


def render_skeleton_video(npz_path, output_path, width=640, height=640, bg_color=(30, 30, 30)):
    """Render a single .npz file to a skeleton video."""
    data = np.load(npz_path, allow_pickle=True)
    joints = data['joints'][:, :NUM_JOINTS, :]  # Use first 17 COCO joints
    fps = float(data['fps'])
    n_frames = len(joints)

    # Center on mid-hip to remove global translation
    mid_hip = (joints[:, HIP_LEFT:HIP_LEFT+1, :] + joints[:, HIP_RIGHT:HIP_RIGHT+1, :]) / 2.0
    joints = joints - mid_hip

    # Align skeleton to face camera, then use X (lateral) and Y (vertical)
    joints = align_to_front(joints)
    j2d_all = joints[:, :, :2]  # front-view orthographic: X, Y

    # Use median body height (ankle-to-nose) for stable scaling.
    heights = np.linalg.norm(j2d_all[:, 0, :] - (j2d_all[:, 15, :] + j2d_all[:, 16, :]) / 2.0, axis=1)
    median_height = np.median(heights)
    margin = 0.12
    scale = (1.0 - 2 * margin) * min(width, height) * 0.5 / max(median_height, 1e-6)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for t in range(n_frames):
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
        j2d = j2d_all[t].copy()

        # Center each frame on its own midpoint for stable positioning
        frame_center = (j2d.min(axis=0) + j2d.max(axis=0)) / 2.0
        j2d = (j2d - frame_center) * scale
        j2d[:, 1] = -j2d[:, 1]  # flip Y (image coords)
        j2d[:, 0] += width / 2
        j2d[:, 1] += height / 2
        pts = j2d.astype(np.int32)

        # Draw bones
        for a, b in BONES:
            bone = (min(a,b), max(a,b))
            color = BONE_COLORS.get(bone, COLOR_TORSO)
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 4, cv2.LINE_AA)

        # Draw joints
        for i in range(NUM_JOINTS):
            cv2.circle(frame, tuple(pts[i]), 6, (255, 255, 255), -1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    reencode_h264(output_path)
    print(f"  Wrote {n_frames} frames @ {fps:.0f} fps -> {output_path}")


CONF_THRESHOLD = 0.3  # ViTPose confidence threshold


def draw_skeleton(frame, pts, conf=None, line_thickness=3, joint_radius=5):
    """Draw skeleton bones and joints on a frame. Skips low-confidence joints."""
    for a, b in BONES:
        if conf is not None and (conf[a] < CONF_THRESHOLD or conf[b] < CONF_THRESHOLD):
            continue
        bone = (min(a, b), max(a, b))
        color = BONE_COLORS.get(bone, COLOR_TORSO)
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, line_thickness, cv2.LINE_AA)
    for i in range(NUM_JOINTS):
        if conf is not None and conf[i] < CONF_THRESHOLD:
            continue
        cv2.circle(frame, tuple(pts[i]), joint_radius, (255, 255, 255), -1, cv2.LINE_AA)


def render_overlay_video(npz_path, video_path, output_path):
    """Overlay skeleton on original video using ViTPose 2D keypoints."""
    data = np.load(npz_path, allow_pickle=True)
    if 'keypoints_2d' not in data:
        print(f"  SKIP (no keypoints_2d — re-run wham_inference.py) {npz_path}", file=sys.stderr)
        return
    kp2d = data['keypoints_2d']  # (T, 17, 3) — x, y, confidence
    frame_ids = data['frame_ids']
    fps = float(data['fps'])

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Scale line thickness based on video resolution
    diag = (width**2 + height**2) ** 0.5
    line_w = max(2, int(diag / 250))
    joint_r = max(3, int(diag / 180))

    # Build a mapping: video_frame_index -> skeleton_index
    frame_map = {int(fid): idx for idx, fid in enumerate(frame_ids)}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_map:
            skel_idx = frame_map[frame_idx]
            pts = kp2d[skel_idx, :NUM_JOINTS, :2].astype(np.int32)
            conf = kp2d[skel_idx, :NUM_JOINTS, 2]
            draw_skeleton(frame, pts, conf=conf, line_thickness=line_w, joint_radius=joint_r)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    reencode_h264(output_path)
    n_drawn = len(frame_ids)
    print(f"  Overlay {n_drawn}/{frame_idx} frames -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render WHAM joints as stick-figure skeleton video")
    parser.add_argument("input", nargs="+", help=".npz file(s) or directory containing them")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .mp4 path or directory (default: alongside input)")
    parser.add_argument("--video", default=None,
                        help="Original video for overlay mode (draws skeleton on top of video)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()

    # Gather all .npz files
    npz_files = []
    for inp in args.input:
        if os.path.isdir(inp):
            for f in sorted(os.listdir(inp)):
                if f.startswith("wham_fragment") and f.endswith(".npz"):
                    npz_files.append(os.path.join(inp, f))
        elif inp.endswith(".npz"):
            npz_files.append(inp)

    if not npz_files:
        print("No .npz files found.", file=sys.stderr)
        sys.exit(1)

    for npz_path in npz_files:
        suffix = "overlay" if args.video else "skeleton"
        if args.output and os.path.isdir(args.output):
            base = os.path.splitext(os.path.basename(npz_path))[0]
            out_path = os.path.join(args.output, base.replace("wham_fragment", suffix) + ".mp4")
        elif args.output and len(npz_files) == 1:
            out_path = args.output
        else:
            out_path = npz_path.replace("wham_fragment", suffix).replace(".npz", ".mp4")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        print(f"Rendering {npz_path} ...")

        if args.video:
            render_overlay_video(npz_path, args.video, out_path)
        else:
            render_skeleton_video(npz_path, out_path, args.width, args.height)

    print("Done.")


if __name__ == "__main__":
    main()
