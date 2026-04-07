# KineGuard Research Brief — Motion-Language Model Landscape & Curation Strategy

**Date:** 2026-03-30
**Author:** Jaehoon (Sogang University, graduate researcher)
**Purpose:** Find the best approach to classify suggestive/NSFW dance motion from 3D skeleton data alone, for dataset curation during web crawling.

---

## 1. Project Context

### What is KineGuard?
KineGuard is a skeleton-based video content safety classifier targeting a 4-tier ordinal scale:
- **Tier 0:** Normal — everyday non-dance movement (walking, sitting, waving)
- **Tier 1:** Artistic — high-energy but wholesome dance/athletics (K-pop, breakdancing, gymnastics)
- **Tier 2:** Suggestive — sensual or sexually suggestive movement (twerking, sensual floorwork, pelvic rolls, hip isolations, body waves)
- **Tier 3:** Explicit — pornography or overtly sexual physical actions

The critical distinction: KineGuard classifies **motion patterns**, NOT visual appearance. The same hip isolation in a bikini vs a tracksuit must get the same tier. Skeleton representation is chosen specifically to strip away appearance bias (skin, clothing, camera angles, lighting, body type).

### Current Pipeline
1. **WHAM** (3D human mesh recovery) extracts 3D skeleton from video → outputs `wham_fragment_*.npz` files containing (T, 31, 3) joints in a custom Halpe-based ordering
2. These 31 WHAM joints are mapped to **SMPL 22-body-joint format** (the standard used by HumanML3D) via a verified mapping
3. LMA (Laban Movement Analysis) features are also extracted from the skeleton
4. A classifier (KineGuard — not yet trained) would take skeleton data and output a tier

### The Curation Problem
To train KineGuard, we need labeled skeleton data. During web crawling (primarily TikTok), we need to automatically classify videos into tiers so we can build a balanced training dataset. We cannot use human annotators for the initial bulk labeling at scale — we need an automated or semi-automated approach.

**Critical constraint:** The labels must reflect motion content, not appearance. An appearance-biased labeler would produce biased training data, which would make KineGuard itself appearance-biased — defeating its entire purpose.

---

## 2. What We Tried and Why It Failed

### 2.1 DEMO (Dense Motion Captioning)
- **What:** LLaMA-3.1-8B-Instruct + lightweight MLP motion adapter (vision_tower 2-layer MLP: Linear(1056,1024)→ReLU→Linear(1024,1024) + mm_projector)
- **Published:** Late 2024, arXiv
- **Training data:** AMASS / HumanML3D exclusively — everyday activities (walking, running, sitting, picking up objects) and some sports. Zero dance content.
- **Input format:** (T, 22, 3) float32 joints in HumanML3D "new_joints" format (Y-up, XZ centered at first-frame root, floor at Y=0, facing +Z). Windowed into patches: W=16, stride S=8 → patches of shape (N, 1056=16×66).
- **Result on our test videos:** Complete failure. Every dance video produces captions like "military crawl backwards", "swimming movements", "doing a handstand", "performing cartwheels". The model maps every dance posture to the nearest AMASS training example. Tier classification returns N/A (unparseable) for 7/8 test videos.
- **Root cause:** Domain gap. DEMO's vocabulary is entirely limited to its training data (AMASS/HumanML3D). It has never seen dance, let alone suggestive dance.
- **Technical fixes we applied (for correct inference, not the domain gap):**
  - `config.pretrain_mm = None` — training-time checkpoint path doesn't exist
  - `torch_dtype=torch.float32` — avoids Float/BFloat16 mismatch in vision tower
  - No `rotate_y_up_to_z_up` — DEMO's training code has this commented out, so inference must match
  - Manual greedy decoding — DEMO's custom `generate()` is broken; workaround uses `prepare_inputs_labels_for_multimodal` then manual token-by-token generation with KV cache
  - `LlamaForCausalLM.generate(model, ...)` bypass — DEMO's custom generate() fails for text-only prompts (classify_tier); calling the parent class method directly works
  - Correct WHAM31→SMPL22 joint mapping (see section below)

### 2.2 Qwen3-VL-Embedding-2B on Original Video
- **What:** Vision-language embedding model, zero-shot cosine similarity against tier text prompts
- **Result:** Correctly classifies chairdance as tier2 (score 0.579 vs next-highest 0.414)
- **Why we can't use it for curation:** It classifies based on VISUAL APPEARANCE — skin exposure, clothing style, camera angles, environment. A person doing a non-suggestive K-pop routine in revealing clothing would be classified as tier2. A person twerking in full winter gear would be classified as tier1. This is exactly the bias KineGuard is designed to eliminate. Using appearance-biased labels to train a motion classifier would encode that bias into the model.

### 2.3 Qwen3-VL-Embedding-2B on Rendered Skeleton Video
- **What:** Same model, but given a stick-figure skeleton video (rendered from WHAM output) instead of the original video
- **Result:** Nearly flat scores (T0=0.333, T1=0.393, T2=0.373, T3=0.346) — classified as tier1 (barely). The model is essentially guessing because a stick figure contains zero visual NSFW cues.
- **Implication:** Vision-language models cannot classify motion content from skeleton visualizations. They need appearance cues.

### 2.4 Summary of Failure Modes

| Method | Input | Result | Why it fails |
|--------|-------|--------|-------------|
| DEMO | skeleton .npz | N/A — "swimming", "military crawl" | No dance vocabulary in training data |
| Qwen3-VL | original video | tier2 ✓ (but appearance-biased) | Detects clothing/skin, not motion |
| Qwen3-VL | skeleton video | tier1 (guessing, flat scores) | No appearance cues in stick figure |

---

## 3. WHAM-to-SMPL22 Joint Mapping (Verified)

WHAM uses `J_regressor_wham.npy` producing 31 custom Halpe-based joints — NOT standard SMPL joints. The first 22 are face/shoulder/arm keypoints in non-standard order. The verified mapping from WHAM 31 to SMPL 22 body joints:

```
SMPL idx → WHAM idx   (joint name)
 0 → mean(11,12)       hips/root (midpoint of WHAM l_hip and r_hip)
 1 → 11                l_hip
 2 → 12                r_hip
 3 → interpolated      spine (hips + 0.213 × (neck − hips))
 4 → 21                l_knee (exact match)
 5 → 18                r_knee (exact match)
 6 → interpolated      spine1 (hips + 0.477 × (neck − hips))
 7 → 22                l_ankle (exact match)
 8 → 17                r_ankle (exact match)
 9 → interpolated      spine2 (hips + 0.581 × (neck − hips))
10 → l_ankle − 0.05m   l_toe (approximated)
11 → r_ankle − 0.05m   r_toe (approximated)
12 → 29                neck (dist=0.012m from SMPL neck)
13 → 5                 l_collar
14 → 6                 r_collar
15 → 0                 head (best available)
16 → 26                l_shoulder (exact)
17 → 25                r_shoulder (exact)
18 → 27                l_elbow (exact)
19 → 24                r_elbow (exact)
20 → 28                l_wrist (exact)
21 → 23                r_wrist (exact)
```

Spine interpolation percentages (0.213, 0.477, 0.581) derived from SMPL rest-pose proportions.

---

## 4. Models Discovered So Far (from initial research)

### 4.1 MotionScript (IROS 2025)
- **arXiv:** 2312.12634
- **Approach:** Rule-based (no ML training). Converts 3D skeleton to structured text using "posecodes" (static pose features) and "motioncodes" (temporal changes).
- **Output example:** Detailed per-joint descriptions like "left hip rotates outward", "torso leans forward"
- **Code:** github.com/pjyazdian/MotionScript (MIT license)
- **Relevance:** Could convert WHAM skeletons to text, then feed to LLM for tier classification. Purely motion-based, no appearance. Rule-based means deterministic and interpretable.
- **Shortcomings:** Hand-crafted rules may miss nuanced motion patterns. Limited expressive vocabulary. No semantic understanding of motion context (describes "hip rotation" but can't distinguish dance vs exercise vs sexual motion).

### 4.2 MotionLLM (IDEA Research, May 2024)
- **arXiv:** 2405.20340
- **Key innovation:** Joint video + motion training. Uses MoVid dataset which includes dance, performance, kungfu, music categories.
- **Input:** SMPL motion sequences + video
- **Weights:** Public (Google Drive, LoRA weights + projection layer)
- **Relevance:** Trained on dance data, goes beyond HumanML3D. Could potentially understand dance motion.
- **Concern:** Does it learn motion semantics or just visual semantics projected onto motion? Unclear if it can distinguish suggestive vs artistic dance from skeleton alone.

### 4.3 M3GPT (NeurIPS 2024)
- **Approach:** Unified text + music + motion/dance in single LLM. Supports music-to-dance, dance-to-music.
- **Dance data:** AIST++ (music-dance pairs)
- **Code:** github.com/luomingshuang/M3GPT
- **Relevance:** Dance-aware by design. But focused on generation, not classification.

### 4.4 MG-MotionLLM (CVPR 2025)
- **Key:** Multi-granularity — 28 distinct motion tasks including fine-grained captioning, temporal localization
- **Code:** github.com/CVI-SZU/MG-MotionLLM
- **Relevance:** Fine-grained captioning could produce better descriptions than DEMO

### 4.5 MotionLib / Being-M0 (ICML 2025)
- **Dataset:** 1.2M motion clips — 15x larger than prior datasets
- **Relevance:** Models trained on this have exposure to far more diverse motions

### 4.6 Superman (Feb 2026)
- **arXiv:** 2602.02401
- **Key:** Vision-Guided Motion Tokenizer — learns joint representations from both video and skeleton modalities
- **Relevance:** Most explicit video-to-skeleton knowledge distillation

### 4.7 Motion-X++ (Jan 2025)
- **arXiv:** 2501.05098
- **Dataset:** 19.5M whole-body pose annotations, 120.5K sequences, 80.8K RGB videos
- **Relevance:** Any model trained on this has much more diverse motion exposure

---

## 5. What We Need to Find (Research Questions)

### Primary Question
**Is there a model (published late 2024 – early 2026) that can take raw 3D skeleton input and produce text descriptions detailed enough to distinguish suggestive dance motion from artistic dance motion?**

Specifically:
- Can it describe hip isolations, pelvic thrusts, body waves, grinding as distinct from hip-hop isolations, ballet port de bras, gymnastics?
- Does it go beyond MotionScript's rule-based approach with learned representations?
- Does it have dance-specific training data?

### Secondary Questions
1. Are there any models that specifically CITE and BUILD UPON MotionScript, addressing its limitations?
2. Are there papers on fine-grained dance style classification from skeleton data (late 2024 – 2026)?
3. Are there papers on movement quality assessment from skeleton (Laban-related, effort/shape analysis)?
4. Are there any skeleton-based approaches to detecting sensual/suggestive motion qualities?

### Search Terms to Use
- Papers citing MotionScript (arXiv:2312.12634) on Google Scholar / Semantic Scholar
- "skeleton to text" OR "motion to text" 2025 2026
- "fine-grained motion captioning" 2025
- "motion description" learned 2025 2026
- "dance classification skeleton" 2024 2025 2026
- "dance style recognition" skeleton 2025
- "movement quality" skeleton 2025
- "Laban Movement Analysis" deep learning 2025
- "motion quality assessment" 2025
- "sensual motion" OR "suggestive motion" detection
- "per-joint motion description" 2025
- Check CVPR 2025, ECCV 2024, NeurIPS 2024, AAAI 2025, SIGGRAPH 2025, ICCV 2025 proceedings

---

## 6. Practical Curation Options (Ranked by Feasibility)

### Option A: MotionScript → LLM Classification
- Convert WHAM skeletons to text via MotionScript (rule-based, deterministic)
- Feed text descriptions to a strong LLM (e.g., GPT-4, Claude) with a prompt: "Given this motion description, classify as tier 0/1/2/3"
- **Pro:** No training, no appearance bias, purely motion-based, works today
- **Con:** MotionScript's rule-based vocabulary may not capture the nuance between suggestive and artistic motion
- **Status:** Not yet tested

### Option B: Better Motion-Language Model → Classification
- If a model exists that produces richer skeleton-to-text descriptions than MotionScript (learned, not rule-based), use that instead
- This is what the research question above is trying to answer

### Option C: Manual Annotation of Skeleton Videos
- Render stick-figure videos from WHAM output
- Human annotators label tiers watching ONLY the stick figures
- **Pro:** Ground truth, zero appearance bias
- **Con:** Slow, expensive, doesn't scale for web crawling
- **Potential hybrid:** Use Option A for bulk labeling, human review for disagreements

### Option D: Fine-tune an Existing Motion LLM
- Take MotionLLM or MG-MotionLLM, fine-tune on a small manually-annotated dance dataset
- Teach it to distinguish suggestive vs artistic motion
- **Pro:** Learned, potentially more nuanced than rules
- **Con:** Requires manual annotations first (chicken-and-egg), needs compute

### Option E: LMA Feature-Based Classifier
- We already extract LMA (Laban Movement Analysis) features from WHAM output
- Train a simple classifier (SVM, small MLP, or gradient boosting) on hand-crafted LMA features
- LMA features include: flow/effort, shape, space, weight — designed specifically to capture movement quality
- **Pro:** Fast, interpretable, motion-only, already computed in our pipeline
- **Con:** Needs labeled data (same chicken-and-egg), may not capture all relevant patterns

---

## 7. Existing Code & Artifacts

### Scripts in the KineGuard Repo

| File | Purpose |
|------|---------|
| `eval/kineguard_baseline.py` | Three-way comparison: DEMO vs Qwen3(skeleton) vs Qwen3(video). Accepts single video, folder, or CSV. Checkpoint-based (skips stages if output exists). |
| `eval/compare_demo_vs_qwen3.py` | Older two-way comparison (DEMO vs Qwen3 on original video only). Superseded by kineguard_baseline.py. |
| `eval/demo_baseline_eval.py` | DEMO-only evaluation on a directory of WHAM outputs. |
| `core/demo_motion_classifier.py` | DEMO inference module. Loads DEMO model, processes WHAM .npz files, generates captions, attempts tier classification. Has all fixes applied (pretrain_mm=None, float32, manual greedy decode, LlamaForCausalLM bypass). |
| `core/qwen3_vl_tier_classifier.py` | Qwen3-VL-Embedding-2B tier classifier. Zero-shot cosine similarity. Accepts video files. Includes Python 3.9 / torch 2.1.0 compatibility monkey-patch. |
| `core/wham_inference.py` | Full WHAM pipeline: video → 2D detection → 3D lifting → SMPL → .npz fragments + LMA features. |
| `core/render_skeleton.py` | Renders WHAM .npz data as stick-figure skeleton videos (standalone or overlay on original). |

### Test Results
- `output/kineguard_baseline/7170751798842526982/` — full pipeline output for chairdance video
- `output/skeleton_test/round1/` — 8 test videos (chairdance, twerk, fancam, dancehall, bellydance, heels, fitness, bachata) with WHAM fragments, Qwen3 results, DEMO results

### Environment
- Conda env: `wham` (Python 3.9.23, torch 2.1.0+cu118)
- DEMO needs transformers==4.44.0
- Qwen3 needs transformers==4.57.6 (script handles upgrade/restore automatically)
- DEMO weights: `/tmp/DEMO/weights/stage2` (LLaMA-3.1-8B + motion adapter)
- DEMO repo: `/tmp/DEMO`
- Qwen3-VL-Embedding repo: `/tmp/Qwen3-VL-Embedding`
- GPUs: 4× NVIDIA RTX A6000 (48GB each)

### Test Videos
Located in `kineguard_recon_tiktok/`:
- `cat3_dancehall/` — dancehall dance videos
- `cat4_bachata/` — bachata dance videos
- `cat6_bellydance/` — belly dance videos
- `cat7_chairdance/` — chair dance videos

Round 1 test set (8 videos with ground-truth tier labels):
- chairdance (GT: T2), twerk (GT: T2), fancam (GT: T1), dancehall (GT: T2)
- bellydance (GT: T1/T2), heels (GT: T1/T2), fitness (GT: T0/T1), bachata (GT: T1/T2)

---

## 8. Key Takeaway for Next Steps

The fundamental unsolved problem is: **no existing model can distinguish suggestive from artistic motion using only skeleton data.** This is not a minor engineering gap — it's a genuine research contribution if solved.

The most promising immediate path is testing whether MotionScript (or a successor) can produce text descriptions detailed enough for an LLM to make the suggestive/artistic distinction. If the text says "pelvis thrusts forward repeatedly with lateral hip isolation and forward torso lean" vs "sharp lateral hip pop with upright posture and raised arms", an LLM *might* be able to classify the first as tier2 and the second as tier1.

But this needs to be validated empirically on the test videos.
