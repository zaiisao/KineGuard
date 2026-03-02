import multiprocessing as mp
import os
import torch
import argparse
from glob import glob
import multiprocessing.pool
import cv2

class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwargs):
        return NoDaemonProcess(*args, **kwargs)

def init_worker(gpu_id_queue):
    gpu_id = gpu_id_queue.get()
    
    # Isolate the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # =========================================================================
    # THE FIX: STOP CPU THREAD EXPLOSION
    # Force underlying C++ libraries to only use 1 thread per worker
    # =========================================================================
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    import cv2
    cv2.setNumThreads(0) # 0 means "use only the calling thread"
    # =========================================================================

    print(f"[*] Worker {os.getpid()} successfully assigned to GPU {gpu_id} with restricted CPU threads")

def hunt_nan_in_video(video_path):
    """
    Runs ONLY the SLAM/DPVO module to check for NaN math explosions.
    Extremely fast compared to full WHAM inference.
    """
    try:
        # Import inside worker to ensure CUDA isolation is respected
        from lib.models.preproc.slam import SLAMModel
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            return (False, f"[ERROR] Failed to open {os.path.basename(video_path)}")
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[*] Worker {os.getpid()} starting scan on: {os.path.basename(video_path)}", flush=True)
        
        # Initialize SLAM directly (skipping the rest of WHAM)
        # Create a unique dummy output path for this worker to prevent race conditions
        dummy_dir = f"dummy_output_{os.getpid()}"
        os.makedirs(dummy_dir, exist_ok=True)
        slam = SLAMModel(video_path, dummy_dir, width, height, calib=None)
        
        # Initialize GPU metrics
        gpu_id = torch.cuda.current_device()
        mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3) # in GB
        frame_count = 0
        while cap.isOpened():
            flag, img = cap.read()
            if not flag: 
                break

            # =========================================================================
            # PROBE A: Frame Integrity (Did the CPU choke and drop the image?)
            # =========================================================================
            is_black_frame = (img is None or img.max() == 0)
            if is_black_frame:
                print(f"⚠️ [CPU STARVATION] Worker {os.getpid()} got a BLANK frame at {frame_count} in {os.path.basename(video_path)}", flush=True)

            # =========================================================================
            # PROBE B: VRAM Pressure (Are we hitting the ceiling?)
            # =========================================================================
            mem_used = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            if mem_used > (mem_total * 0.90): # Warn if we cross 90% capacity
                print(f"⚠️ [VRAM SPIKE] Worker {os.getpid()} hit {mem_used:.2f}GB / {mem_total:.2f}GB at frame {frame_count} in {os.path.basename(video_path)}", flush=True)
            
            # Step 1: Run the tracker
            slam.track()
            frame_count += 1
            
            # Step 2: Check for math explosions
            # Access DPVO's internal pose memory
            if hasattr(slam, 'dpvo') and slam.dpvo is not None:
                if torch.isnan(slam.dpvo.poses).any():
                    msg = f"🚨 [FOUND NaN] at frame {frame_count} in {os.path.basename(video_path)} | VRAM: {mem_used:.2f}GB | Blank Frame? {is_black_frame}"
                    print(msg, flush=True)
                    
                    if os.path.exists(os.path.join(dummy_dir, 'calib.txt')):
                        os.remove(os.path.join(dummy_dir, 'calib.txt'))
                    os.rmdir(dummy_dir)
                    return (True, msg)
                 
        # Force DPVO to finish its asynchronous queue
        import numpy as np
        slam_results = slam.process()
        
        # Clean up the dummy directory
        if os.path.exists(os.path.join(dummy_dir, 'calib.txt')):
            os.remove(os.path.join(dummy_dir, 'calib.txt'))
        os.rmdir(dummy_dir)
        
        # Check the final computed trajectory for NaNs
        if np.isnan(slam_results).any():
            msg = f"🚨 [FOUND NaN AFTER PROCESSING] in {os.path.basename(video_path)}"
            print(msg, flush=True)
            return (True, msg)

        return (False, f"[CLEAN] {os.path.basename(video_path)} processed {frame_count} frames safely.")
        
    except Exception as e:
        # If DPVO crashes entirely before we can check the tensor
        if "NaN" in str(e) or "nan" in str(e).lower():
             print(f"🚨 [FOUND NaN CRASH] in {os.path.basename(video_path)}: {str(e)}", flush=True)
             return (True, f"[FOUND NaN CRASH] at frame {frame_count} in {os.path.basename(video_path)}")
        return (False, f"[FAILED] {os.path.basename(video_path)} failed with: {str(e)}")

def run_nan_hunter(input_folder, num_processes):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    all_videos = []
    for ext in extensions:
        all_videos.extend(glob(os.path.join(input_folder, ext)))
        all_videos.extend(glob(os.path.join(input_folder, ext.upper())))

    if not all_videos:
        print(f"[!] No videos found in {input_folder}")
        return

    print(f"[*] Found {len(all_videos)} total videos to scan.")
    print(f"[*] Starting NaN Hunter pool with {num_processes} workers...")

    num_gpus = torch.cuda.device_count()
    manager = mp.Manager()
    gpu_id_queue = manager.Queue()
    
    for i in range(num_processes):
        gpu_id_queue.put(i % num_gpus)

    with NoDaemonPool(processes=num_processes, 
                 initializer=init_worker, 
                 initargs=(gpu_id_queue,)) as pool:
        
        # We only need the video path for the hunter
        results = pool.map(hunt_nan_in_video, all_videos)

    print("\n" + "="*50)
    print("FINAL NaN HUNTER REPORT")
    print("="*50)
    
    culprits = []
    for found_nan, msg in results:
        if found_nan:
            culprits.append(msg)
            print(f"🚨 {msg}")
        else:
            print(f"✅ {msg}")
            
    print("\n" + "="*50)
    print(f"Total NaN triggers found: {len(culprits)}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KineGuard DPVO NaN Hunter")
    parser.add_argument("--input", type=str, required=True, help="Path to folder containing video files")
    parser.add_argument("--jobs", type=int, default=2, help="Number of parallel processes (workers) to run")

    args = parser.parse_args()

    run_nan_hunter(
        input_folder=args.input, 
        num_processes=args.jobs
    )