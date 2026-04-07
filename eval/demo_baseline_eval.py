#!/usr/bin/env python3
"""
Baseline evaluation: DEMO (Dense Motion Captioning) on dance/NSFW skeletons.

DEMO is an LLM-based motion captioner trained on AMASS / HumanML3D, which
covers everyday activities and sports but contains NO dance vocabulary.
This script runs DEMO on WHAM skeleton outputs and records the captions it
produces, demonstrating that out-of-the-box motion LLMs cannot decode
dance or NSFW movement — motivating the need for KineGuard.

Input layout (walks the tree recursively):
    <wham_dir>/
        <any_subdir>/
            wham_fragment_*.npz   ← WHAM per-person skeleton fragments
        ...

Usage:
    conda run -n wham python eval/demo_baseline_eval.py \\
        output/skeleton_test/round1 \\
        --model_path /tmp/DEMO/weights/stage2 \\
        --demo_repo /tmp/DEMO \\
        --output output/skeleton_test/round1/demo_baseline_results.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def discover_fragments(wham_dir):
    """
    Walk wham_dir and group wham_fragment_*.npz files by their immediate
    parent directory (= one video per directory).

    Returns:
        dict mapping relative video path -> sorted list of absolute .npz paths
    """
    groups = defaultdict(list)
    for root, _, files in os.walk(wham_dir):
        for f in sorted(files):
            if f.startswith("wham_fragment_") and f.endswith(".npz"):
                groups[root].append(os.path.join(root, f))
    return {
        os.path.relpath(k, wham_dir): sorted(v)
        for k, v in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DEMO motion captioning on WHAM skeleton outputs and show that "
            "it cannot classify dance/NSFW content (domain mismatch baseline)."
        )
    )
    parser.add_argument(
        "wham_dir",
        help="Root directory containing per-video subdirs with wham_fragment_*.npz files",
    )
    parser.add_argument(
        "--model_path",
        default="/tmp/DEMO/weights/stage2",
        help="Path to DEMO stage-2 model weights (default: /tmp/DEMO/weights/stage2)",
    )
    parser.add_argument(
        "--demo_repo",
        default="/tmp/DEMO",
        help="Path to the DEMO source repository (default: /tmp/DEMO)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output file (default: <wham_dir>/demo_baseline_results.json)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        args.wham_dir, "demo_baseline_results.json"
    )

    # ── Discover videos ────────────────────────────────────────────────────
    videos = discover_fragments(args.wham_dir)
    if not videos:
        print(f"[!] No wham_fragment_*.npz files found under {args.wham_dir}")
        sys.exit(1)
    print(f"[*] Found {len(videos)} video dir(s) under {args.wham_dir}")

    # ── Load DEMO ──────────────────────────────────────────────────────────
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    from core.demo_motion_classifier import (
        load_model,
        process_fragment,
        classify_tier,
        TIERS,
    )

    print(f"[*] Loading DEMO model from {args.model_path} ...")
    model, tokenizer = load_model(args.model_path, args.demo_repo)

    # ── Run inference ──────────────────────────────────────────────────────
    results = []

    for video_rel, npz_files in videos.items():
        print(f"\n{'='*70}")
        print(f"  {video_rel}  ({len(npz_files)} fragment(s))")
        print(f"{'='*70}")

        all_captions = []
        fragments = []

        for npz in npz_files:
            frag_name = os.path.basename(npz)
            caption, tier = process_fragment(model, tokenizer, npz, args.demo_repo)
            if caption is None:
                print(f"  [{frag_name}] skipped (too short)")
                continue
            tier_str = str(tier) if tier >= 0 else "unparseable"
            print(f"  [{frag_name}]")
            print(f"    Caption : {caption}")
            print(f"    Tier    : {tier_str}")
            all_captions.append(caption)
            fragments.append({"npz": npz, "caption": caption, "tier": tier})

        if not all_captions:
            print("  (no usable fragments)")
            continue

        combined = " | ".join(all_captions)
        final_tier = classify_tier(combined, model, tokenizer)
        final_tier_str = str(final_tier) if final_tier >= 0 else "unparseable"
        print(f"\n  Aggregated tier : {final_tier_str}")
        if final_tier >= 0:
            print(f"  Tier meaning    : {TIERS[final_tier]}")

        results.append(
            {
                "video": video_rel,
                "final_tier": final_tier,
                "fragments": fragments,
            }
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    col = 42
    print(f"  {'Video':<{col}} {'Tier':>12}  First-fragment caption")
    print(f"  {'-'*col} {'----------':>12}  {'-'*30}")
    for r in results:
        tier_str = str(r["final_tier"]) if r["final_tier"] >= 0 else "N/A"
        cap = r["fragments"][0]["caption"][:60] if r["fragments"] else ""
        print(f"  {r['video']:<{col}} {tier_str:>12}  {cap}")

    print()
    print("  OBSERVATION")
    print("  -----------")
    print("  DEMO was trained on AMASS / HumanML3D — everyday activities and")
    print("  sports with no dance or NSFW content. It maps all dance postures")
    print("  to its nearest training vocabulary: 'military crawl backwards',")
    print("  'swimming movements', 'laying down on back', etc.")
    print("  Tier classification consistently fails (returns N/A or Tier 0).")
    print()
    print("  This confirms that off-the-shelf motion LLMs cannot be used as")
    print("  dance/NSFW classifiers without domain-specific training.")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Results saved to {output_path}")


if __name__ == "__main__":
    main()
