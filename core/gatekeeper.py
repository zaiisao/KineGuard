import sys
import os
import cv2
import json
import argparse
from nudenet import NudeDetector
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Graceful import of the black box
try:
    from internal_logic.constants_secret import EXPLICIT_LABELS
except ImportError:
    EXPLICIT_LABELS = []

class PrecisionGatekeeper:
    def __init__(self, batch_size=32, gap_tolerance=5):
        self.detector = NudeDetector()
        self.batch_size = batch_size
        self.gap_tolerance = gap_tolerance
        self.trigger_labels = EXPLICIT_LABELS

    def process_video(self, video_path, output_json):
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps) if fps > 0 else 0
        
        raw_hits = []

        print(f"--- Analyzing: {os.path.basename(video_path)} ---")

        for start_sec in range(0, duration, self.batch_size):
            batch_paths, batch_times = [], []
            
            # 1. Extraction phase
            for i in range(self.batch_size):
                curr_sec = start_sec + i
                if curr_sec >= duration: break
                
                cap.set(cv2.CAP_PROP_POS_MSEC, curr_sec * 1000)
                success, frame = cap.read()
                if success:
                    path = f"tmp_f_{curr_sec}.jpg"
                    cv2.imwrite(path, frame)
                    batch_paths.append(path)
                    batch_times.append(curr_sec)

            # 2. Inference & Real-time Shredding
            if batch_paths:
                results = self.detector.detect_batch(batch_paths)
                
                for timestamp, detections in zip(batch_times, results):
                    hits = [d for d in detections if d['class'] in self.trigger_labels and d['score'] > 0.45]
                    if hits:
                        raw_hits.append({"timestamp": timestamp, "detections": hits})

                for path in batch_paths:
                    if os.path.exists(path):
                        os.remove(path)

        cap.release()
        merged_blocks = self._merge_segments([h['timestamp'] for h in raw_hits])

        final_data = {
            "video_metadata": {"path": video_path, "duration": duration},
            "raw_frame_hits": raw_hits,
            "action_blocks": merged_blocks
        }

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(final_data, f, indent=4)
        
        print(f"Done. Found {len(raw_hits)} hits in {len(merged_blocks)} blocks.")
        return final_data

    def _merge_segments(self, timestamps):
        if not timestamps: return []
        segments = []
        start = timestamps[0]
        prev = timestamps[0]

        for current in timestamps[1:]:
            if current - prev > self.gap_tolerance:
                segments.append({"start": max(0, start-1), "end": prev+1})
                start = current
            prev = current
        segments.append({"start": max(0, start-1), "end": prev+1})
        return segments

def main():
    print(EXPLICIT_LABELS)
    parser = argparse.ArgumentParser(description="KineGuard Gatekeeper: Video Action Localization")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--out", type=str, default="./results.json", help="Path to save result JSON")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for GPU inference")
    parser.add_argument("--gpu", type=str, default="0", help="Target CUDA device ID")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    gatekeeper = PrecisionGatekeeper(batch_size=args.batch)
    gatekeeper.process_video(args.video, args.out)

if __name__ == "__main__":
    main()