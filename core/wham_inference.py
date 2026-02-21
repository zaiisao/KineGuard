import numpy as np
np.float = float

import sys
import os
import cv2
import torch
import joblib
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict
from progress.bar import Bar
from loguru import logger
from scipy.spatial import ConvexHull

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.vis.run_vis import run_vis_on_demo

script_dir = os.path.dirname(os.path.abspath(__file__))
lma_path = os.path.abspath(os.path.join(script_dir, "../external/dance-style-recognition/src"))
sys.path.append(lma_path)

from process_lma_features import compute_lma_descriptor, IdentityFloor

import subprocess

sys.path.append('external/WHAM/third-party/DPVO')

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except ImportError: 
    logger.warning('DPVO (SLAM) is not installed. Global trajectory will default to local camera space!')
    _run_global = False

class KineGuardWHAMProcessor:
    def __init__(self, cfg_path='configs/yamls/demo.yaml'):
        print("[*] Initializing KineGuard WHAM Processor...")
        self.cfg = get_cfg_defaults()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        wham_root = os.path.join(script_dir, '..', 'external', 'WHAM')
        full_cfg_path = os.path.join(wham_root, cfg_path)
        
        self.cfg.merge_from_file(full_cfg_path)
        
        original_cwd = os.getcwd()
        os.chdir(wham_root)
        
        # Build WHAM SMPL Model & Network
        smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        self.smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
        self.network = build_network(self.cfg, self.smpl)
        self.network.eval()
        
        # Detector & Extractor (Replaces YOLO & ViTPose)
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower(), self.cfg.FLIP_EVAL)

    def preprocess_video(self, video_path, output_pth, calib=None, use_slam=True):
        """Replaces Phase 1: 2D Extraction."""

        with torch.no_grad():
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            use_slam = use_slam and _run_global
            slam = SLAMModel(video_path, output_pth, width, height, calib) if use_slam else None
            
            bar = Bar('Preprocessing: Tracking and SLAM', fill='#', max=length)
            while cap.isOpened():
                flag, img = cap.read()
                if not flag: break
                
                self.detector.track(img, fps, length)
                if slam is not None: slam.track()
                bar.next()
            cap.release()

            tracking_results = self.detector.process(fps)
            slam_results = slam.process() if slam is not None else np.zeros((length, 7))
            if slam is None: slam_results[:, 3] = 1.0
            
            tracking_results = self.extractor.run(video_path, tracking_results)
            return CustomDataset(self.cfg, tracking_results, slam_results, width, height, fps), fps

    def run_pipeline(self, video_path, output_dir, visualize=False):
        """Replaces Phase 2: 3D Lifting (MotionBERT) -> Now using WHAM"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n[*] Phase 1: Preprocessing Video & Extracting Features...")
        dataset, fps = self.preprocess_video(video_path, output_dir)
        
        print("\n[*] Phase 2: WHAM 3D Inference & Global Optimization...")
        results = defaultdict(dict)
        n_subjs = len(dataset)
        
        for subj in range(n_subjs):
            with torch.no_grad():
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # WHAM Inference
                pred = self.network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # 1. Align temporal dimensions (T) and combine root + body poses
                # Force shape to (Batch=1, Time=T, Joints, 3, 3)
                root_pose = pred['poses_root_world'].reshape(1, -1, 1, 3, 3)
                body_pose = pred['poses_body'].reshape(1, -1, 23, 3, 3)
                poses_world_mat = torch.cat([root_pose, body_pose], dim=2)
                
                # 2. Extract the 6D rotation tensor AND preserve the 3D shape (1, T, 144)
                pred_rot6d_world = poses_world_mat[..., :3, :2].contiguous().reshape(1, -1, 144)
                
                # 3. Call WHAM's custom SMPL wrapper with the exact arguments it requires
                smpl_output = self.network.smpl(
                    pred_rot6d=pred_rot6d_world,
                    betas=pred['betas']
                )
                
                # 4. Extract 3D data, apply world translation, and strip the dummy batch dimension
                trans_world = pred['trans_world'].reshape(1, -1, 1, 3) # (1, T, 1, 3)
                joints_world = (smpl_output.joints + trans_world).cpu().squeeze(0).numpy() # -> (T, 45, 3)
                verts_world = (smpl_output.vertices + trans_world).cpu().squeeze(0).numpy() # -> (T, 6890, 3)
                
                # 5. Restore all original WHAM dictionary keys for the visualizer
                root_world_aa = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
                root_cam_aa = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
                body_aa = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
                
                results[_id]['frame_ids'] = frame_id
                results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
                results[_id]['pose'] = np.concatenate((root_cam_aa, body_aa), axis=-1)
                results[_id]['pose_world'] = np.concatenate((root_world_aa, body_aa), axis=-1)
                
                # Foolproof trans and verts handling for the visualizer
                trans_cam = pred['trans_cam'].cpu().squeeze(0).numpy()
                results[_id]['trans'] = trans_cam - self.network.output.offset.cpu().numpy()
                results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
                
                verts_cam = pred['verts_cam'].cpu().squeeze(0).numpy()
                results[_id]['verts'] = verts_cam + trans_cam[:, None, :] # Broadcast trans to (T, 1, 3)
                
                # 6. Store our LMA-specific parameters!
                results[_id]['joints_world'] = joints_world 
                results[_id]['verts_world'] = verts_world

        if not results:
            print("[!] No subjects detected.")
            return None, fps
            
        processed_fragments = {}
        
        for _id, data in results.items():
            frames = data['frame_ids']
            
            # Optional: Skip noise/glitches (e.g., tracks shorter than 1 second)
            if len(frames) < 30:
                print(f"[*] Skipping ID {_id} (Too short: {len(frames)} frames)")
                continue
                
            print(f"\n[*] Processing Fragment ID {_id} with {len(frames)} frames...")

            # 1. Save specific NPZ for this fragment
            out_npz = osp.join(output_dir, f"wham_fragment_id{_id}.npz")
            np.savez(
                out_npz,
                joints=data['joints_world'],
                verts=data['verts_world'],
                frame_ids=frames,
                fps=fps
            )
            print(f"    -> Saved kinematics: {out_npz}")
            processed_fragments[_id] = data

            # 2. Render and Crop Video for this fragment
            if visualize:
                # Create a temp folder for WHAM's native renderer
                temp_dir = osp.join(output_dir, f"temp_vis_{_id}")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Render the full-length video but ONLY drawing this specific ID
                run_vis_on_demo(self.cfg, video_path, {_id: data}, temp_dir, self.network.smpl, vis_global=_run_global)
                
                generated_videos = glob(osp.join(temp_dir, '*.mp4'))
                if len(generated_videos) > 0:
                    raw_render = generated_videos[0]
                    final_cropped_video = osp.join(output_dir, f"preview_fragment_id{_id}.mp4")
                    
                    # Calculate timestamps based on frame indices
                    start_frame = int(np.min(frames))
                    end_frame = int(np.max(frames))
                    
                    start_time = start_frame / fps
                    duration = (end_frame - start_frame + 1) / fps
                    
                    print(f"    -> Cropping video from {start_time:.2f}s to {start_time+duration:.2f}s")
                    
                    # FFmpeg trims the dead space where the raw video was showing nothing
                    cmd = [
                        'ffmpeg', '-y', 
                        '-ss', str(start_time), 
                        '-t', str(duration),
                        '-i', raw_render, 
                        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', 
                        final_cropped_video
                    ]
                    
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Clean up the bloated temp files
                    os.remove(raw_render)
                    os.rmdir(temp_dir)
                    print(f"    -> Saved preview video: {final_cropped_video}")

        return processed_fragments, fps

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--output_dir", type=str, default="output/wham_kineguard", help="Directory for output files")
    parser.add_argument("--viz", action='store_true', help="Generate 3D visualization mp4 using WHAM renderer")
    opts = parser.parse_args()

    processor = KineGuardWHAMProcessor()
    
    # 1. Run WHAM to get all fragments
    fragments, fps = processor.run_pipeline(opts.video, opts.output_dir, visualize=opts.viz)
    
    if fragments:
        print("\n[+] WHAM Processing Complete. Starting LMA Integration...")
        
        for _id, data in fragments.items():
            print(f"[*] Extracting LMA features for Fragment {_id}...")
            
            joints = data['joints_world'][:, :24, :]
            verts_array = data['verts_world']
            
            # A. Calculate Volumes (ConvexHull) - Prerequisite for LMA 'Shape' features
            volumes = []
            last_v = 0.07 # Average human volume fallback
            for verts in verts_array:
                try:
                    v = ConvexHull(verts).volume
                    volumes.append(v)
                    last_v = v
                except Exception:
                    volumes.append(last_v)
            
            # B. Generate the Identity floors (WHAM is already grounded)
            floors = [IdentityFloor()] * len(joints)
            
            # C. Call the frozen LMA logic from your external script
            lma_dict, lma_matrix = compute_lma_descriptor(
                joints=joints, 
                volumes=volumes, 
                floors=floors, 
                fps=fps, 
                window_size=55
            )
            
            # D. Save the final LMA feature vector (N_frames, 55)
            out_lma_npy = osp.join(opts.output_dir, f"lma_features_id{_id}.npy")
            np.save(out_lma_npy, lma_matrix)
            print(f"    -> LMA Matrix saved: {out_lma_npy} | Shape: {lma_matrix.shape}")

    print("\n[SUCCESS] Pipeline end-to-end complete.")