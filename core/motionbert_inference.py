import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from transformers import VitPoseForPoseEstimation, AutoProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))
motionbert_path = os.path.abspath(os.path.join(script_dir, "../external/MotionBERT"))
sys.path.append(motionbert_path)

# Import MotionBERT visualization tool
from lib.utils.vismo import render_and_save

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def generate_comparison_video(input_2d, output_3d, out_path, fps=30):
    # Ensure we don't try to animate more frames than we have in either array
    num_frames = min(len(input_2d), len(output_3d))
    
    fig = plt.figure(figsize=(12, 6))
    
    # Setup 2D Plot (Before)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Before: 2D Polish (ViTPose++)")
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1)
    scatter2d = ax1.scatter([], [], c='green') # Using green to match our overlay logic

    # Setup 3D Plot (After)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("After: 3D MotionBERT")
    ax2.set_xlim3d([-1, 1]); ax2.set_zlim3d([-1, 1]); ax2.set_ylim3d([-1, 1])
    scatter3d = ax2.scatter([], [], [], c='blue')

    def update(frame):
        # Safety check inside update
        if frame >= num_frames:
            return scatter2d, scatter3d
            
        # Update 2D
        kpts2d = input_2d[frame]
        scatter2d.set_offsets(kpts2d[:, :2] * np.array([1, -1]))
        
        # Update 3D
        kpts3d = output_3d[frame]
        scatter3d._offsets3d = (kpts3d[:, 0], kpts3d[:, 2], -kpts3d[:, 1])
        return scatter2d, scatter3d

    # Use the calculated num_frames
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
    ani.save(out_path, writer='ffmpeg', fps=fps)
    plt.close()

# --- 1. JOINT MAPPING & SYNTHESIS (COCO TO H36M) ---
def coco_to_h36m(coco_kpts):
    """
    Transforms YOLO11 (COCO) keypoints to MotionBERT (H36M) format.
    coco_kpts: (17, 3) -> [x, y, confidence]
    """
    h36m = np.zeros((17, 3))
    # H36M Index 0: Pelvis (Root) - Synthesized as midpoint of Hips (11, 12)
    h36m[0] = (coco_kpts[11] + coco_kpts[12]) / 2
    
    # Direct Mappings
    h36m[1] = coco_kpts[12] # R-Hip
    h36m[2] = coco_kpts[14] # R-Knee
    h36m[3] = coco_kpts[16] # R-Ankle
    h36m[4] = coco_kpts[11] # L-Hip
    h36m[5] = coco_kpts[13] # L-Knee
    h36m[6] = coco_kpts[15] # L-Ankle
    
    # Synthesized Spine (7) and Neck (8)
    h36m[7] = (coco_kpts[5] + coco_kpts[6] + coco_kpts[11] + coco_kpts[12]) / 4
    h36m[8] = (coco_kpts[5] + coco_kpts[6]) / 2
    
    # Upper Body & Extremities
    h36m[9]  = coco_kpts[0]  # Nose
    h36m[10] = coco_kpts[0] + (coco_kpts[0] - h36m[8]) # Head Top (approx)
    h36m[11] = coco_kpts[5]  # L-Shoulder
    h36m[12] = coco_kpts[7]  # L-Elbow
    h36m[13] = coco_kpts[9]  # L-Wrist
    h36m[14] = coco_kpts[6]  # R-Shoulder
    h36m[15] = coco_kpts[8]  # R-Elbow
    h36m[16] = coco_kpts[10] # R-Wrist
    return h36m

# --- 2. THE KINEGUARD PROCESSOR ---
class KineGuardProcessor:
    def __init__(self, yolo_model="yolo11x.pt", motionbert_checkpoint="pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin", config="configs/pose3d/MB_ft_h36m.yaml"):
        # Load YOLO11-Pose
        print("Initializing YOLO11-Pose...")
        self.yolo = YOLO(yolo_model)

        self.vit_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
        self.vit_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge").cuda()
        
        # Load MotionBERT Backbone
        from lib.utils.tools import get_config
        from lib.utils.learning import load_backbone

        self.args = get_config(os.path.join(script_dir, "../external/MotionBERT", config))
        self.model_3d = load_backbone(self.args)
        
        if torch.cuda.is_available():
            self.model_3d = nn.DataParallel(self.model_3d).cuda()
        
        # Load MotionBERT Weights
        print(f"Loading MotionBERT Checkpoint: {motionbert_checkpoint}")
        chk = torch.load(os.path.join(script_dir, "../external/MotionBERT/checkpoint", motionbert_checkpoint), map_location='cpu')
        self.model_3d.load_state_dict(chk['model_pos'], strict=True)
        self.model_3d.eval()

    def run_pipeline(self, video_path, clip_len=243):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        filename = os.path.basename(video_path)
        debug_out_path = os.path.join(os.getcwd(), filename.replace(".mp4", "_overlay.mp4"))
        out = cv2.VideoWriter(debug_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        raw_frames = []
        self.saved_pixel_kpts = [] # Critical: to store 2D points for the final overlay

        print("Phase 1: 2D Extraction & Polishing...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            det_results = self.yolo(frame, classes=[0], verbose=False)
            if len(det_results[0].boxes) > 0:
                box = det_results[0].boxes.xyxy[0].cpu().numpy()
                inputs = self.vit_processor(images=frame, boxes=[[box]], return_tensors="pt").to("cuda")

                with torch.no_grad():
                    outputs_2d = self.vit_model(**inputs, dataset_index=torch.tensor([0]).to("cuda"))
                    processed = self.vit_processor.post_process_pose_estimation(outputs_2d, boxes=[[box]])
                    kpts_2d = processed[0][0]["keypoints"]
                    scores = processed[0][0]["scores"]
                    
                    kpts_np = kpts_2d.cpu().numpy() 
                    scores_np = scores.cpu().numpy().reshape(-1, 1)
                    yolo_style_kpts = np.concatenate([kpts_np, scores_np], axis=1)

                    h36m_frame_raw = coco_to_h36m(yolo_style_kpts)
                    self.saved_pixel_kpts.append(h36m_frame_raw[:, :2]) # Store for Phase 3
                    
                    # Draw Step 2 (Green)
                    debug_frame = self.draw_skeleton(frame.copy(), h36m_frame_raw[:, :2], color=(0, 255, 0))
                    out.write(debug_frame)

                h36m_frame = h36m_frame_raw.copy()
                h36m_frame[:, 0] = (h36m_frame[:, 0] - w / 2) / (min(w, h) / 2)
                h36m_frame[:, 1] = (h36m_frame[:, 1] - h / 2) / (min(w, h) / 2)
                raw_frames.append(h36m_frame)
            else:
                raw_frames.append(np.zeros((17, 3)))
                self.saved_pixel_kpts.append(np.zeros((17, 2)))
                out.write(frame) 

        cap.release()
        out.release()
        
        # Phase 2: Temporal Windowing & 3D Inference
        print("Phase 2: 3D Lifting (MotionBERT)...")
        seq = np.array(raw_frames)
        T = len(seq)
        if T < clip_len:
            seq_input = np.pad(seq, ((0, clip_len - T), (0, 0), (0, 0)), 'edge')
        else:
            seq_input = seq[:clip_len]
        
        input_tensor = torch.from_numpy(seq_input).float().unsqueeze(0)
        if torch.cuda.is_available(): input_tensor = input_tensor.cuda()
        
        with torch.no_grad():
            # FIX: This is where 'output' is defined
            output = self.model_3d(input_tensor)
            output[:, :, 0, :] = 0 
        
        motion_3d = output[0].cpu().numpy() 
        
        # Phase 3: Triple Overlay (Green = 2D Refined, Blue = 3D Re-projected)
        print("Phase 3: Triple Overlay Generation...")
        cap = cv2.VideoCapture(video_path)
        triple_path = os.path.join(os.getcwd(), filename.replace(".mp4", "_triple_overlay.mp4"))
        out_triple = cv2.VideoWriter(triple_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for i in range(min(len(motion_3d), T)):
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Get the "Green" 2D pixels (Stage 2)
            # We need this to find the anchor point (the pelvis)
            if i < len(self.saved_pixel_kpts):
                green_kpts = self.saved_pixel_kpts[i]
                frame = self.draw_skeleton(frame, green_kpts, color=(0, 255, 0))
                
                # The Anchor: This is where the person actually is in the video
                # In H36M, Index 0 is the Pelvis
                pelvis_2d = green_kpts[0] 
            else:
                pelvis_2d = np.array([w/2, h/2]) # Fallback to center

            # 2. Process the "Blue" 3D -> 2D (Stage 3)
            # motion_3d[i] is centered at (0,0). We need to scale then shift.
            kpts_3d_2d = motion_3d[i][:, :2].copy()
            
            # Scale first (to get the size right)
            kpts_3d_2d[:, 0] *= (min(w, h) / 2)
            kpts_3d_2d[:, 1] *= (min(w, h) / 2)
            
            # Shift to the Pelvis anchor (repositioning it back)
            kpts_3d_2d[:, 0] += pelvis_2d[0]
            kpts_3d_2d[:, 1] += pelvis_2d[1]
            
            # Draw Blue (now overlaying the Green)
            frame = self.draw_skeleton(frame, kpts_3d_2d, color=(255, 0, 0))
            out_triple.write(frame)

        cap.release()
        out_triple.release()
        return seq, output.cpu().numpy()

    def draw_skeleton(self, frame, kpts_2d, color=(0, 255, 0)):
        """
        Draws the H36M skeleton onto the frame.
        kpts_2d: (17, 2) in pixel coordinates.
        """
        # H36M connectivity (pairs of joint indices)
        skeleton_links = [
            (0, 1), (1, 2), (2, 3),      # Right Leg
            (0, 4), (4, 5), (5, 6),      # Left Leg
            (0, 7), (7, 8), (8, 9), (8, 10), # Spine & Head
            (8, 11), (11, 12), (12, 13), # Left Arm
            (8, 14), (14, 15), (15, 16)  # Right Arm
        ]
        
        for start, end in skeleton_links:
            pt1 = tuple(kpts_2d[start].astype(int))
            pt2 = tuple(kpts_2d[end].astype(int))
            cv2.line(frame, pt1, pt2, color, 2)
            
        for pt in kpts_2d:
            cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 0, 255), -1)
        
        return frame

# --- 3. MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--output", type=str, default="./polished_3d.npy")
    parser.add_argument("--viz", action='store_true', default=True, help="Generate 3D visualization mp4")
    opts = parser.parse_args()

    processor = KineGuardProcessor()
    input_2d, polished_motion = processor.run_pipeline(opts.video)
    
    # Save NumPy Result
    np.save(opts.output, polished_motion)
    print(f"Success! Polished 3D motion data saved to {opts.output}")

    # Visualization Step
    if opts.viz:
        print("Generating side-by-side comparison video...")
        # input_2d: (T, 17, 3), polished_motion: (1, T, 17, 3)
        viz_3d = polished_motion[0] if polished_motion.ndim == 4 else polished_motion
        
        comp_path = opts.output.replace(".npy", "_comparison.mp4")
        generate_comparison_video(input_2d, viz_3d, comp_path)
        print(f"Comparison video saved to {comp_path}")