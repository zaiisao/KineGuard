import os
import cv2
import json
import argparse
from tqdm import tqdm
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Folder containing raw videos")
    parser.add_argument("--save_dir", type=str, default="motion_metadata", help="Where to save JSON maps")
    parser.add_argument("--min_kpts", type=int, default=10, help="Min keypoints needed")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Min average confidence")
    parser.add_argument("--min_seconds", type=float, default=3.0, help="Min continuous seconds to keep")
    return parser.parse_args()

def process_video(video_path, model, args):
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    required_frames = int(args.min_seconds * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results = model.predict(source=video_path, stream=True, conf=0.25, verbose=False)
    
    segments = []
    current_segment = []
    max_p_in_seg = 0
    
    print(f"--- Mapping: {video_name} ({fps:.2f} FPS) ---")
    for frame_idx, r in enumerate(tqdm(results, total=total_frames)):
        is_good = False
        p_count = 0
        
        if r.keypoints and len(r.keypoints.conf) > 0:
            for person_conf in r.keypoints.conf:
                visible_count = (person_conf > 0.4).sum().item()
                avg_conf = person_conf.mean().item()
                
                if visible_count >= args.min_kpts and avg_conf >= args.min_conf:
                    is_good = True
                    p_count += 1 
        
        if is_good:
            current_segment.append(frame_idx)
            max_p_in_seg = max(max_p_in_seg, p_count)
        else:
            if len(current_segment) >= required_frames:
                segments.append({
                    "start_frame": current_segment[0],
                    "end_frame": current_segment[-1],
                    "max_persons": max_p_in_seg, # Store peak density
                    "start_time": round(current_segment[0] / fps, 3),
                    "end_time": round(current_segment[-1] / fps, 3),
                    "duration": round(len(current_segment) / fps, 3)
                })
            current_segment = []
            max_p_in_seg = 0
            
    if len(current_segment) >= required_frames:
        segments.append({
            "start_frame": current_segment[0],
            "end_frame": current_segment[-1],
            "max_persons": max_p_in_seg,
            "start_time": round(current_segment[0] / fps, 3),
            "end_time": round(current_segment[-1] / fps, 3),
            "duration": round(len(current_segment) / fps, 3)
        })

    if not segments:
        print(f"Skipping {video_name}: No segments meet criteria.")
        return

    metadata = {
        "source_video": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "valid_segments": segments,
        "segment_count": len(segments)
    }

    output_path = os.path.join(args.save_dir, f"{video_name}_map.json")
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Saved Metadata: {output_path} ({len(segments)} segments found)")
    cap.release()

def main():
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = YOLO("yolo11x-pose.pt") 
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    all_files = [f for f in os.listdir(args.source_dir) if f.lower().endswith(video_extensions)]
    
    for filename in all_files:
        process_video(os.path.join(args.source_dir, filename), model, args)

if __name__ == "__main__":
    main()