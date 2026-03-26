"""
Qwen3-VL-Embedding Tier Classifier
===================================
Classifies videos into a 4-tier ordinal content safety scale using
Qwen3-VL-Embedding-2B zero-shot cosine similarity.

  Tier 0: Normal (일상) — everyday non-dance movement
  Tier 1: Artistic (예술) — high-energy but wholesome dance/athletics
  Tier 2: Suggestive (암시) — sensual or sexually suggestive movement
  Tier 3: Explicit (노골적) — pornography or overtly sexual content

Setup:
    git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
    pip install torch>=2.2 transformers>=4.53 qwen-vl-utils decord

Usage:
    python qwen3_vl_tier_classifier.py /path/to/video_dir
    python qwen3_vl_tier_classifier.py video1.mp4 video2.mp4 video3.mp4
    python qwen3_vl_tier_classifier.py /path/to/video_dir --qwen3-vl-repo /path/to/Qwen3-VL-Embedding
    python qwen3_vl_tier_classifier.py /path/to/video_dir --csv results.csv
"""
import argparse
import csv
import glob
import os
import sys

import functools
import numpy as np
import torch

# Compatibility shim for torch <2.2 with transformers >=4.53.
# Not needed if torch>=2.2 is installed (recommended).
import torch.utils._pytree as _pytree
if not hasattr(_pytree, 'register_pytree_node') and hasattr(_pytree, '_register_pytree_node'):
    _orig = _pytree._register_pytree_node
    @functools.wraps(_orig)
    def _compat_register(*args, serialized_type_name=None, **kwargs):
        return _orig(*args, **kwargs)
    _pytree.register_pytree_node = _compat_register

# Text anchors for zero-shot classification.
#   Tier 0: Normal (일상) – 걷기, 손 흔들기, 대기 동작 등 일반적인 움직임.
#   Tier 1: Artistic (예술) – 스트릿 댄스, 기계 체조 등 에너지가 높지만 건전한 움직임.
#   Tier 2: Suggestive (암시) – 성적인 암시가 있거나 경계선에 있는 선정적인 움직임.
#   Tier 3: Explicit (노골적) – 명백히 부적절하거나 유해한 NSFW 동작.
TIER_PROMPTS = {
    "tier0": (
        "Non-dance everyday movement: walking, standing, sitting, waving, "
        "stretching, or ordinary physical activity with no dance intent"
    ),
    "tier1": (
        "Artistic non-sensual dance or athletics: street dance, hip-hop, "
        "gymnastics, breakdancing, K-pop group choreography, or other "
        "vigorous movement without sexual undertones"
    ),
    "tier2": (
        "Suggestive or sensual dance movement: hip isolations, body waves, "
        "twerking, sensual floorwork, pelvic rolls, slow grinding, wine/whine, "
        "or any choreography emphasizing sexual body movement — "
        "including in tutorial or class settings"
    ),
    "tier3": (
        "Explicit hardcore pornography, sexual intercourse, or overtly "
        "sexual physical actions"
    ),
}


def load_model(qwen3_vl_repo):
    """Load Qwen3-VL-Embedding-2B model."""
    sys.path.insert(0, qwen3_vl_repo)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    model = Qwen3VLEmbedder(
        model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_length=16384,
    )
    return model


TIER_NAMES = list(TIER_PROMPTS.keys())


def encode_tier_prompts(model):
    """Encode tier text prompts into embeddings."""
    tier_inputs = [{"text": TIER_PROMPTS[t]} for t in TIER_NAMES]
    with torch.no_grad():
        tier_embs = model.process(tier_inputs)
        if isinstance(tier_embs, torch.Tensor):
            tier_embs = tier_embs.cpu().float().numpy()
    return tier_embs


def classify_video(model, tier_embs, video_path, fps=0.5, max_frames=16):
    """Classify a single video. Returns dict of {tier: similarity} and the best-match tier."""
    with torch.no_grad():
        vid_emb = model.process([{"video": video_path, "fps": fps, "max_frames": max_frames}])
        if isinstance(vid_emb, torch.Tensor):
            vid_emb = vid_emb.cpu().float().numpy()

    sims = {}
    for i, tier in enumerate(TIER_NAMES):
        sims[tier] = float(np.dot(vid_emb[0], tier_embs[i]))
    best_tier = max(sims, key=sims.get)
    return sims, best_tier


def collect_videos(path):
    """Collect video files from a path (file or directory, supports nested dirs)."""
    video_exts = (".mp4", ".webm", ".mkv", ".avi", ".mov")
    if os.path.isfile(path):
        return [path]
    videos = []
    for ext in video_exts:
        videos.extend(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description="Classify dance videos using Qwen3-VL-Embedding")
    parser.add_argument("input", nargs="+", help="Video file(s) or directory of videos")
    parser.add_argument("--qwen3-vl-repo", required=True,
                        help="Path to cloned Qwen3-VL-Embedding repo")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Frame sampling rate for video encoding (default: 0.5)")
    parser.add_argument("--max-frames", type=int, default=16,
                        help="Max frames to sample per video (default: 16)")
    parser.add_argument("--csv", default=None,
                        help="Optional CSV output path for results")
    args = parser.parse_args()

    # Load model
    print("[*] Loading Qwen3-VL-Embedding-2B...")
    model = load_model(args.qwen3_vl_repo)
    print("[+] Model loaded.")

    # Encode tier prompts
    tier_embs = encode_tier_prompts(model)
    print("[+] Tier prompts encoded:")
    for t in TIER_NAMES:
        print(f"    {t}: {TIER_PROMPTS[t]}")

    # Collect and classify videos
    videos = []
    for inp in args.input:
        videos.extend(collect_videos(inp))
    print(f"\n[*] Classifying {len(videos)} video(s)...\n")

    results = []
    for vid_path in videos:
        vid_name = os.path.basename(vid_path)
        parent = os.path.basename(os.path.dirname(vid_path))
        try:
            sims, best_tier = classify_video(
                model, tier_embs, vid_path, fps=args.fps, max_frames=args.max_frames
            )
            results.append({
                "directory": parent,
                "filename": vid_name,
                "path": vid_path,
                "best_tier": best_tier,
                **{f"sim_{t}": sims[t] for t in TIER_NAMES},
            })
            sim_str = "  ".join(f"{t}={sims[t]:.4f}" for t in TIER_NAMES)
            print(f"  {best_tier.upper():<5}  {sim_str}  {parent}/{vid_name}")
        except Exception as e:
            print(f"  [!] ERROR {parent}/{vid_name}: {e}")

    # Summary table
    sim_headers = "  ".join(f"{'T'+t[-1]:>6}" for t in TIER_NAMES)
    print(f"\n{'='*100}")
    print(f"{'Directory':<20} {'Filename':<25} {sim_headers}  {'Best'}")
    print(f"{'='*100}")
    for r in sorted(results, key=lambda x: -x.get("sim_tier2", 0)):
        sim_vals = "  ".join(f"{r[f'sim_{t}']:>6.4f}" for t in TIER_NAMES)
        print(f"{r['directory']:<20} {r['filename']:<25} {sim_vals}  {r['best_tier']}")

    # Per-tier counts
    print()
    for t in TIER_NAMES:
        count = sum(1 for r in results if r["best_tier"] == t)
        print(f"  {t}: {count}/{len(results)}")

    # CSV export
    if args.csv and results:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[+] Results saved to {args.csv}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
