#!/usr/bin/env python3
"""
DEMO (Dense Motion Captioning) tier classifier for KineGuard.

Loads WHAM .npz files, extracts 22 SMPL body joints (T, 22, 3),
feeds them into DEMO (LLaMA-3.1-8B + motion MLP adapter), gets a
dense motion caption, then classifies into KineGuard tiers.

Usage:
    conda run -n wham python core/demo_motion_classifier.py \
        --model_path /tmp/DEMO/weights/stage2 \
        --demo_repo /tmp/DEMO \
        output/skeleton_test/round1/chairdance/7170751798842526982 \
        output/skeleton_test/round1/twerk/6927659767498394885

DEMO repo: https://github.com/41xu/DEMO
Model weights: https://huggingface.co/Xusy2333/DEMO
"""
import argparse
import os
import sys
import copy
import glob
import json

import numpy as np
import torch


# ── Tier descriptions ──────────────────────────────────────────────────────────
TIERS = {
    0: "Non-dance everyday movement: walking, standing, sitting, waving, "
       "stretching, or ordinary physical activity with no dance intent",
    1: "Artistic non-sensual dance or athletics: street dance, hip-hop, "
       "gymnastics, breakdancing, K-pop group choreography, or other vigorous "
       "movement without sexual undertones",
    2: "Suggestive or sensual dance movement: hip isolations, body waves, "
       "twerking, sensual floorwork, pelvic rolls, slow grinding, wine/whine, "
       "or any choreography emphasizing sexual body movement — including in "
       "tutorial or class settings",
    3: "Explicit hardcore pornography, sexual intercourse, or overtly sexual "
       "physical actions",
}

# ── Joint format ───────────────────────────────────────────────────────────────
# WHAM smpl_output.joints → (T, 45, 3), first 22 = standard SMPL body joints.
# DEMO was trained on HumanML3D new_joints which stores (T, 22, 3) at 20 FPS.
DEMO_FPS = 20
SMPL_JOINTS = 22


def load_model(model_path, demo_repo):
    """Load DEMO M2TLlamaForCausalLM from local path."""
    sys.path.insert(0, demo_repo)
    from model.m2t_llama import M2TLlamaForCausalLM, M2TConfig
    from transformers import AutoConfig, AutoTokenizer

    print(f"[*] Loading DEMO model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Set motion_dim and mlp_hidden if missing (they're None in the saved config)
    if not hasattr(config, "motion_dim") or config.motion_dim is None:
        config.motion_dim = 1056   # 16 frames × 22 joints × 3 = 1056
    if not hasattr(config, "mlp_hidden") or config.mlp_hidden is None:
        config.mlp_hidden = 1024
    # pretrain_mm points to a training-time path that no longer exists;
    # weights are already merged into the safetensors checkpoint.
    config.pretrain_mm = None

    model = M2TLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    model.eval()
    return model, tokenizer


# Mapping from WHAM's 31-joint Halpe-based regressor to standard SMPL 22 body joints
# (verified by comparing rest-pose euclidean distances).
# Values are WHAM joint indices; -1 = computed (see _wham31_to_smpl22).
_SMPL22_FROM_WHAM31 = [
    -1,  # 0: hips/root → mean(WHAM[11], WHAM[12])
    11,  # 1: l_hip
    12,  # 2: r_hip
    -2,  # 3: spine → interpolated
    21,  # 4: l_knee  (exact)
    18,  # 5: r_knee  (exact)
    -3,  # 6: spine1 → interpolated
    22,  # 7: l_ankle (exact)
    17,  # 8: r_ankle (exact)
    -4,  # 9: spine2 → interpolated
    -5,  # 10: l_toe → l_ankle − 0.05 m
    -6,  # 11: r_toe → r_ankle − 0.05 m
    29,  # 12: neck   (dist=0.012)
     5,  # 13: l_collar
     6,  # 14: r_collar
     0,  # 15: head   (best available)
    26,  # 16: l_shoulder (exact)
    25,  # 17: r_shoulder (exact)
    27,  # 18: l_elbow    (exact)
    24,  # 19: r_elbow    (exact)
    28,  # 20: l_wrist    (exact)
    23,  # 21: r_wrist    (exact)
]


def _wham31_to_smpl22(joints31: np.ndarray) -> np.ndarray:
    """
    Convert WHAM 31-joint output (T, 31, 3) to standard SMPL 22 body joints (T, 22, 3).

    WHAM uses a custom 31-joint Halpe-based regressor (J_regressor_wham.npy).
    Its first 22 joints are face/arm/hip joints in a non-standard order — NOT
    the standard SMPL body joints used by HumanML3D.  This function extracts
    the correct SMPL 22 joints via the mapping above.
    """
    T = joints31.shape[0]
    out = np.zeros((T, 22, 3), dtype=np.float32)
    for smpl_idx, wham_idx in enumerate(_SMPL22_FROM_WHAM31):
        if wham_idx >= 0:
            out[:, smpl_idx, :] = joints31[:, wham_idx, :]
    # Computed joints
    hips  = (joints31[:, 11, :] + joints31[:, 12, :]) / 2.0   # root / pelvis
    neck  = joints31[:, 29, :]                                  # neck
    out[:, 0, :] = hips
    out[:, 3, :] = hips + 0.213 * (neck - hips)   # spine
    out[:, 6, :] = hips + 0.477 * (neck - hips)   # spine1
    out[:, 9, :] = hips + 0.581 * (neck - hips)   # spine2
    out[:, 10, :] = out[:, 7, :].copy(); out[:, 10, 1] -= 0.05  # l_toe ≈ l_ankle
    out[:, 11, :] = out[:, 8, :].copy(); out[:, 11, 1] -= 0.05  # r_toe ≈ r_ankle
    return out


def _normalize_to_humanml3d(joints: np.ndarray) -> np.ndarray:
    """
    Normalize SMPL 22 joints (T, 22, 3) to HumanML3D new_joints format:
      - XZ centered at first-frame root (joint 0)
      - Y floor at 0
      - Facing direction rotated to +Z at t=0
    DEMO training data was NOT rotated by rotate_y_up_to_z_up; don't apply it.
    """
    joints = joints.copy().astype(np.float32)
    joints[:, :, 0] -= joints[0, 0, 0]
    joints[:, :, 2] -= joints[0, 0, 2]
    joints[:, :, 1] -= joints[:, :, 1].min()
    # Facing: derive from l_hip (joint 1) → r_hip (joint 2) vector at t=0
    hip_vec = joints[0, 2, [0, 2]] - joints[0, 1, [0, 2]]
    forward_xz = np.array([-hip_vec[1], hip_vec[0]])
    n = np.linalg.norm(forward_xz)
    if n > 1e-6:
        forward_xz /= n
        angle = np.arctan2(forward_xz[0], forward_xz[1])
        ca, sa = np.cos(-angle), np.sin(-angle)
        x, z = joints[:, :, 0].copy(), joints[:, :, 2].copy()
        joints[:, :, 0] = ca * x - sa * z
        joints[:, :, 2] = sa * x + ca * z
    return joints


def load_joints(npz_path):
    """Return (T, 22, 3) SMPL body joints from a WHAM .npz, resampled to DEMO_FPS."""
    data = np.load(npz_path)
    joints31 = data["joints"].astype(np.float32)  # (T, 31, 3) WHAM custom joints

    # Convert to standard SMPL 22 joints in HumanML3D order
    joints = _wham31_to_smpl22(joints31)

    # Resample from video FPS → 20 FPS
    src_fps = float(data["fps"]) if "fps" in data else 30.0
    if abs(src_fps - DEMO_FPS) > 0.5:
        T = joints.shape[0]
        target_T = max(1, int(round(T * DEMO_FPS / src_fps)))
        idx = np.round(np.linspace(0, T - 1, target_T)).astype(int)
        joints = joints[idx]

    # Normalize to HumanML3D coordinate system
    joints = _normalize_to_humanml3d(joints)

    return joints


def caption_motion(model, tokenizer, joints, demo_repo):
    """
    Run DEMO inference on (T, 22, 3) SMPL body joints and return a caption string.

    Notes:
    - joints must already be in HumanML3D format (call load_joints / _normalize_to_humanml3d first)
    - Do NOT apply rotate_y_up_to_z_up: DEMO training data was NOT rotated
    - Uses manual greedy decoding to bypass DEMO's broken super().generate() loop
    """
    from data.generate_json import time_convert
    from utils.conversation import conv_templates
    from utils.utils import DEFAULT_MOTION_TOKEN, MOTION_TOKEN_INDEX
    from utils.mm_utils import tokenizer_motion_token

    T = joints.shape[0]
    duration = time_convert(T / DEMO_FPS)
    device = next(model.parameters()).device

    # Do NOT rotate — DEMO training data was NOT rotated (rotate_y_up_to_z_up is
    # commented out in DEMO's datasets.py)
    motion = torch.tensor(joints, dtype=torch.float32).to(device)

    human_input = (
        DEFAULT_MOTION_TOKEN + "\n\n"
        f"Given a complex human motion sequence of duration {duration} which "
        "includes several actions, describe these actions in the motion with "
        "natural language according to the movement of human.\n"
        "The description of each action should be in the format 'mm:ss:ms - text'.\n"
        "Here is an example: 00:00:00 - moves in a curve to the right side, "
        "00:05:09 - doing a left foot squat\n"
    )

    conv = copy.deepcopy(conv_templates["llama_3"])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], human_input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_motion_token(
        prompt, tokenizer, MOTION_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    attn_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)

    # Manual greedy decoding — bypasses DEMO's broken super().generate() call
    with torch.no_grad():
        (_, pos_ids, new_attn, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attn_mask, None, None, [motion], dtype=torch.float32
        )
        generated = []
        past_kv = None
        cur_embeds = inputs_embeds
        for _ in range(400):
            fwd = model(
                input_ids=None,
                attention_mask=new_attn,
                position_ids=pos_ids,
                past_key_values=past_kv,
                inputs_embeds=cur_embeds,
                use_cache=True,
            )
            past_kv = fwd.past_key_values
            next_tok = fwd.logits[:, -1, :].argmax(dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            generated.append(next_tok.item())
            cur_embeds = model.model.embed_tokens(next_tok.unsqueeze(0))
            new_attn = torch.cat(
                [new_attn, torch.ones((1, 1), dtype=new_attn.dtype, device=device)], dim=1
            )
            pos_ids = None

    return tokenizer.decode(generated, skip_special_tokens=True)


def classify_tier(caption, model, tokenizer):
    """
    Ask the LLM to classify a motion caption into tiers 0-3.
    Bypasses DEMO's custom generate() using LlamaForCausalLM.generate directly.
    """
    from transformers import LlamaForCausalLM

    tier_list = "\n".join(f"  Tier {k}: {v}" for k, v in TIERS.items())
    prompt = (
        f"Given this description of a human motion:\n\n"
        f"\"{caption}\"\n\n"
        f"Classify it into exactly one of these tiers:\n{tier_list}\n\n"
        f"Reply with ONLY a single digit: 0, 1, 2, or 3."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = LlamaForCausalLM.generate(
            model, **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for ch in reply:
        if ch in "0123":
            return int(ch)
    return -1  # unable to parse


def process_fragment(model, tokenizer, npz_path, demo_repo):
    joints = load_joints(npz_path)
    if joints.shape[0] < 20:
        return None, None
    caption = caption_motion(model, tokenizer, joints, demo_repo)
    tier = classify_tier(caption, model, tokenizer)
    return caption, tier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_dirs", nargs="+",
                        help="Directories containing wham_fragment_*.npz files, "
                             "OR direct paths to .npz files")
    parser.add_argument("--model_path", default="/tmp/DEMO/weights/stage2")
    parser.add_argument("--demo_repo", default="/tmp/DEMO")
    parser.add_argument("--output", default=None,
                        help="Optional JSON output file")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.demo_repo)

    results = []
    for path in args.npz_dirs:
        if path.endswith(".npz"):
            npz_files = [path]
        elif os.path.isdir(path):
            npz_files = sorted(glob.glob(os.path.join(path, "wham_fragment_*.npz")))
        else:
            print(f"[!] Skipping unknown path: {path}")
            continue

        if not npz_files:
            print(f"[!] No .npz files found in {path}")
            continue

        # Aggregate captions across all fragments for this video
        all_captions = []
        frag_results = []
        for npz in npz_files:
            print(f"\n  [{os.path.basename(npz)}]")
            caption, tier = process_fragment(model, tokenizer, npz, args.demo_repo)
            if caption is None:
                print("    Skipped (too short)")
                continue
            print(f"    Caption: {caption[:200]}{'...' if len(caption) > 200 else ''}")
            print(f"    Tier:    {tier}")
            all_captions.append(caption)
            frag_results.append({"npz": npz, "caption": caption, "tier": tier})

        if not all_captions:
            continue

        # Final classification on concatenated captions
        combined = " | ".join(all_captions)
        final_tier = classify_tier(combined, model, tokenizer)

        label = f"TIER{final_tier}" if final_tier >= 0 else "UNKNOWN"
        print(f"\n  {'='*60}")
        print(f"  {label}  {path}")
        print(f"  {'='*60}")

        results.append({
            "path": path,
            "final_tier": final_tier,
            "fragments": frag_results,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Path':<45} {'Tier':>6}")
    print(f"{'='*70}")
    for r in results:
        label = f"TIER{r['final_tier']}" if r['final_tier'] >= 0 else "UNKNOWN"
        print(f"  {label}  {os.path.basename(r['path'])}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[*] Saved results to {args.output}")


if __name__ == "__main__":
    main()
