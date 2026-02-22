import multiprocessing as mp
import os
import torch
import argparse
from glob import glob

def init_worker(gpu_id_queue):
    """
    This runs once per worker process immediately after it is spawned.
    It locks each worker to a specific physical GPU.
    """
    gpu_id = gpu_id_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Since only one GPU is visible, the worker will use 'cuda:0' internally
    print(f"[*] Worker {os.getpid()} successfully assigned to GPU {gpu_id}")

def run_multiprocess_pipeline(input_folder, output_root, num_processes, visualize):
    from wham_inference import process_single_video

    # CRITICAL: Must use 'spawn' for CUDA applications
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. GATHER VIDEOS (Restored this part!)
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_list = []
    for ext in extensions:
        video_list.extend(glob(os.path.join(input_folder, ext)))
        video_list.extend(glob(os.path.join(input_folder, ext.upper())))

    if not video_list:
        print(f"[!] No videos found in {input_folder}")
        return

    print(f"[*] Found {len(video_list)} videos. Starting pool with {num_processes} workers...")
    os.makedirs(output_root, exist_ok=True)

    # 2. PREPARE GPU QUEUE
    num_gpus = torch.cuda.device_count()
    manager = mp.Manager()
    gpu_id_queue = manager.Queue()
    
    # Fill the queue with GPU IDs (0, 1, 2, 3, 0, 1...) 
    # to be picked up by the workers
    for i in range(num_processes):
        gpu_id_queue.put(i % num_gpus)

    # 3. INITIALIZE POOL
    # initializer=init_worker ensures the GPU is isolated BEFORE wham_inference is imported
    with mp.Pool(processes=num_processes, 
                 initializer=init_worker, 
                 initargs=(gpu_id_queue,)) as pool:
        
        # Prepare tasks (No device_id needed here anymore, init_worker handles it)
        tasks = [(vid, output_root, visualize) for vid in video_list]
        results = pool.starmap(process_single_video, tasks)

    # 4. REPORT
    print("\n" + "="*30)
    print("FINAL REPORT")
    print("="*30)
    for success, msg in results:
        status = " [DONE]  " if success else " [FAILED] "
        print(f"{status} {msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KineGuard Multiprocessing Video Processor")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to folder containing video files")
    parser.add_argument("--output", type=str, default="output/multiprocess_results", 
                        help="Root directory for output fragments and features")
    parser.add_argument("--jobs", type=int, default=2, 
                        help="Number of parallel processes (workers) to run")
    parser.add_argument("--viz", action='store_true', 
                        help="Enable 3D visualization rendering")

    args = parser.parse_args()

    run_multiprocess_pipeline(
        input_folder=args.input, 
        output_root=args.output, 
        num_processes=args.jobs, 
        visualize=args.viz
    )