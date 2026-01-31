import cv2
import json
import os
import argparse
from tqdm import tqdm
from ultralytics import YOLO

def extract_rig_data(mapping_json_path, model, args):
    with open(mapping_json_path, 'r') as f:
        map_data = json.load(f)

    video_path = map_data['source_video']
    if not os.path.exists(video_path):
        print(f"Warning: Video not found at {video_path}. Skipping.")
        return

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    
    all_rig_segments = []

    for seg_idx, segment in enumerate(map_data['valid_segments']):
        start_f = segment['start_frame']
        end_f = segment['end_frame']
        
        print(f"Processing {video_name} Segment {seg_idx} ({start_f} -> {end_f})")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        segment_frames = []
        for current_f in tqdm(range(start_f, end_f + 1), leave=False):
            ret, frame = cap.read()
            if not ret: break

            # Run Pose Tracking
            results = model.track(frame, persist=True, conf=args.conf, verbose=False)
            
            frame_entry = {"frame_idx": current_f, "bodies": []}

            if results[0].keypoints is not None:
                # Get normalized coordinates (x, y, confidence)
                # xyn is [N, 17, 2], conf is [N, 17]
                kpts_xyn = results[0].keypoints.xyn.cpu().numpy()
                kpts_conf = results[0].keypoints.conf.cpu().numpy()
                
                # Get tracking IDs if available
                track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [0] * len(kpts_xyn)

                for body_idx in range(len(kpts_xyn)):
                    body_data = {
                        "track_id": int(track_ids[body_idx]),
                        "keypoints": []
                    }
                    for i in range(17):
                        body_data["keypoints"].append({
                            "id": i,
                            "x": float(kpts_xyn[body_idx][i][0]),
                            "y": float(kpts_xyn[body_idx][i][1]),
                            "conf": float(kpts_conf[body_idx][i])
                        })
                    frame_entry["bodies"].append(body_data)
            
            segment_frames.append(frame_entry)
        
        all_rig_segments.append({
            "segment_info": segment,
            "frames": segment_frames
        })

    cap.release()

    # Save output
    output_filename = os.path.join(args.output, f"{video_name}_rig.json")
    final_output = {
        "metadata": map_data,
        "rig_data": all_rig_segments
    }
    
    with open(output_filename, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved rig data to: {output_filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_maps", type=str, required=True, help="Folder with mapping JSONs")
    parser.add_argument("--output", type=str, default="rig_json_outputs", help="Folder for rig data")
    parser.add_argument("--model", type=str, default="yolo11x-pose.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = YOLO(args.model)

    map_files = [f for f in os.listdir(args.input_maps) if f.endswith(".json")]
    for map_file in map_files:
        extract_rig_data(os.path.join(args.input_maps, map_file), model, args)

if __name__ == "__main__":
    main()
