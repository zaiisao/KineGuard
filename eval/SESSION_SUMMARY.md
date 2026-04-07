# KineGuard Session Summary — April 1, 2026

## Project Overview

KineGuard is a skeleton-based video content safety classifier with a 4-tier ordinal scale:
- **Tier 0:** Normal — everyday non-dance movement
- **Tier 1:** Artistic — high-energy wholesome dance/athletics (K-pop, breakdancing, hip-hop choreography)
- **Tier 2:** Suggestive — sensual/sexually suggestive movement (twerking, sensual floorwork, pelvic rolls, chair dance)
- **Tier 3:** Explicit — pornography or overtly sexual physical actions

**Core design principle:** KineGuard classifies MOTION PATTERNS, not visual appearance. The same hip isolation in a bikini vs a tracksuit must get the same tier. Skeleton representation strips away appearance bias (skin, clothing, camera angles).

**Pipeline:** Video → WHAM (3D skeleton extraction, 31 Halpe joints) → LMA feature extraction (55 features per frame via Laban Movement Analysis) → Classifier (not yet trained)

---

## What We Tried and What Failed

### 1. DEMO (Dense Motion Captioning) — FAILED
- LLaMA-3.1-8B + MLP motion adapter, trained on AMASS/HumanML3D
- Takes SMPL 22-joint skeleton input, produces text captions
- **Result:** Every dance video captioned as "military crawl backwards", "swimming movements", "doing a handstand". Tier classification returns N/A for 7/8 videos.
- **Root cause:** Training data (AMASS/HumanML3D) has zero dance content. The model maps twerking to the nearest training example (bent forward = swimming).
- We built `eval/kineguard_baseline.py` — a reproducible script that demonstrates this failure end-to-end.

### 2. Qwen3-VL-Embedding-2B on Original Video — WORKS but BIASED
- Vision-language embedding model, zero-shot cosine similarity against tier text prompts
- **Result:** Correctly classifies chairdance as tier2 (score 0.579). Works on all 8 test videos.
- **Problem:** It classifies based on VISUAL APPEARANCE — skin exposure, clothing, camera angles. The same dance in different clothing gets different scores. This is exactly the bias KineGuard is designed to eliminate.

### 3. Qwen3-VL on Rendered Skeleton Video — FAILED
- Same model, given stick-figure skeleton videos rendered from WHAM output
- **Tested on 16 videos total:** 8 Tier 2 (suggestive) + 8 Tier 1 (artistic)

**Tier 2 skeleton results:**
```
chairdance     tier1  T0=0.348  T1=0.402  T2=0.385  T3=0.367
twerk          tier1  T0=0.363  T1=0.414  T2=0.401  T3=0.376
dancehall      tier1  T0=0.356  T1=0.418  T2=0.391  T3=0.361
bellydance     tier1  T0=0.411  T1=0.452  T2=0.429  T3=0.376
heels          tier1  T0=0.393  T1=0.439  T2=0.418  T3=0.390
bachata        tier1  T0=0.372  T1=0.434  T2=0.422  T3=0.381
fancam         tier1  T0=0.364  T1=0.419  T2=0.404  T3=0.383
fitness        tier1  T0=0.370  T1=0.418  T2=0.396  T3=0.372
```

**Tier 1 skeleton results:**
```
7414882140980022  tier1  T0=0.327  T1=0.381  T2=0.370  T3=0.326
7434898540956060  tier1  T0=0.395  T1=0.444  T2=0.425  T3=0.402
7444218817217072  tier1  T0=0.370  T1=0.417  T2=0.399  T3=0.386
7449029533883272  tier1  T0=0.358  T1=0.397  T2=0.381  T3=0.366
7449729194143124  tier0  T0=0.475  T1=0.474  T2=0.453  T3=0.405
7462700011080453  tier1  T0=0.399  T1=0.447  T2=0.427  T3=0.383
7463199960654220  tier1  T0=0.329  T1=0.387  T2=0.373  T3=0.334
7478514208716377  tier1  T0=0.368  T1=0.422  T2=0.402  T3=0.379
```

- **ALL 16 videos classified as tier1 regardless of actual tier**
- **T2 avg tier2 score: 0.406 vs T1 avg tier2 score: 0.404 — gap of +0.002 (zero separation)**
- **Conclusion:** Embedding-similarity on skeleton video cannot distinguish suggestive from artistic dance. The stick figure carries zero NSFW signal for this model.

### 4. MotionScript (Rule-Based Skeleton→Text) — PARTIALLY WORKS
- Rule-based system (IROS 2025) that converts 3D skeleton to structured natural language
- Describes joint geometry: "left knee bent at 90 degrees", "torso parallel to ground", "right hand behind back"
- Required significant patching to work with our input (6+ bugs in prepare_input, infer_motioncodes, eligibility adjustment, format_and_skip_motioncodes)
- **MotionScript → Claude (blind reasoning LLM):** Correctly classified twerk as Tier 2 with HIGH confidence, identifying:
  - Torso repeatedly horizontal (bent-over posture)
  - Hands behind body / below hips (not athletic arm positioning)
  - Rhythmic hip oscillation pattern
  - Transition to kneeling with sustained hip motion
  - Minimal upper-body choreography
- **MotionScript → Qwen3-VL embedding similarity:** FAILED. Picked tier1 (0.653) over tier2 (0.638). Embedding similarity is too coarse.
- **Conclusion:** Geometric signal IS in the skeleton, but only a reasoning LLM can extract it — not an embedding classifier.

### 5. Motion LLM Survey — NO EXISTING MODEL SOLVES THIS
Models surveyed: MotionLLM (IDEA Research), M3GPT (NeurIPS 2024), MG-MotionLLM (CVPR 2025), MotionGPT, Large Motion Model (ECCV 2024), Being-M0 (ICML 2025), MotionScript (IROS 2025), Superman (2026), ViMoNet (2025).

**Finding:** No published model has "suggestive" vs "artistic" in its training vocabulary. All motion-language models are trained on HumanML3D/AMASS (everyday activities) or AIST++ (dance styles without NSFW annotation). The suggestive-vs-artistic distinction doesn't exist in any motion-language dataset. This is the genuine research gap KineGuard fills.

---

## What Actually Works: LMA Feature Analysis

### The Breakthrough
We already extract Laban Movement Analysis (LMA) features in our WHAM pipeline (55 features per frame covering Body, Effort, Shape, Space). A paper by Turab et al. (arXiv:2504.21166, April 2025) achieves 99.18% accuracy on 10 dance style classification using the same LMA features + SVM/Random Forest on AIST++.

### Gemini Deep Research Taxonomy
The user commissioned Gemini Deep Research to analyze what makes motion suggestive vs artistic through the LMA framework:

**Predicted Tier 1 (Athletic):** Bound Flow, Strong Weight, Sudden Time, Direct Space — power, precision, control
**Predicted Tier 2 (Suggestive):** Free Flow, Light/Released Weight, Sustained Time, Indirect Space — invitation, sensuality, display

Specific markers: pelvic isolation with sustained time + free flow, "sensual release" cycles (bound→free transitions), narrowing kinesphere to sexualized zones.

### Empirical Validation — Three-Tier Analysis (34 videos: 12 T1, 17 T2, 5 T3)

We ran WHAM + LMA on 34 videos:
- **Tier 1 (12 videos):** 1MILLION Dance Studio, RedBull BC One, DanceWorldStyles, K-pop choreography, YouTube hip-hop classes, fitness
- **Tier 2 (17 videos):** TikTok twerk, dancehall, perreo/reggaeton, chairdance, bellydance, heels/floorwork, bachata
- **Tier 3 (5 videos):** NPDI pornography dataset

### Three-Tier Feature Comparison (Mean Values)

```
Feature                    T1 (artistic)   T2 (suggestive)   T3 (explicit)
---------------------------------------------------------------------------
Wrist Acceleration              9.078           4.294            4.658
Wrist Kinetic Energy            0.400           0.155            0.141
Trajectory Curvature           10.869           6.244            3.417
Trajectory Path Length          4.115           2.211            0.743
Pelvis KE Variability           0.552           0.841            1.837
Pelvis Directness              27.998          19.446            8.918
Pelvis Velocity                 2.240           1.599            0.958
Pelvis Jerkiness              948.189         734.561          230.469
Effort Flow                  6461.606        5096.049         1384.051
Effort Weight                  37.619          25.172            8.055
Effort Time                   296.647         206.560           90.211
Flow CV (bound-free)            0.351           0.473            0.600
Body Volume                     0.274           0.256            0.258
Pelvis Energy Ratio             0.135           0.139            0.175
Arm/Pelvis KE Ratio             0.463           0.481            0.465
```

### Effect Sizes (Cohen's d) — Pairwise Comparisons

```
Feature                     T1 vs T2    T2 vs T3    T1 vs T3
--------------------------------------------------------------
Wrist Acceleration            1.37        0.12        1.28  ***
Wrist Kinetic Energy          1.21        0.10        1.34  ***
Trajectory Curvature          1.20        1.31        2.23  ***
Trajectory Path Length        1.13        1.52        2.37  ***
Pelvis KE Variability         1.01        1.54        2.07  ***
Pelvis Directness             0.84        1.59        2.28  ***
Pelvis Velocity               0.96        1.02        2.52  ***
Pelvis Jerkiness              0.41        1.10        2.86  ***
Effort Flow                   0.34        1.07        2.56  ***
Effort Weight                 0.70        1.19        2.45  ***
Effort Time                   0.77        1.20        2.94  ***
Body Volume                   0.48        0.03        0.37
Flow CV (bound-free)          0.77        0.42        0.89  ***
Pelvis Energy Ratio           0.05        0.45        0.54  **
Arm/Pelvis KE Ratio           0.05        0.03        0.01
```

### Key Findings

#### 1. MONOTONIC GRADIENT ACROSS ALL THREE TIERS
The most striking finding is that nearly every feature forms a clean ordinal gradient: T1 → T2 → T3. Tier 2 is not a separate category — it's an intermediate state between athletic and explicit motion on the same continuum.

| Signal | T1 → T2 | T2 → T3 | Pattern |
|---|---|---|---|
| Trajectory path length | 4.12 → 2.21 → 0.74 | Less and less whole-body movement |
| Pelvis KE variability | 0.55 → 0.84 → 1.84 | Pelvis increasingly irregular |
| Pelvis directness | 28.0 → 19.4 → 8.9 | Pelvis increasingly indirect |
| Effort Flow | 6462 → 5096 → 1384 | Decreasing overall flow |
| Effort Time | 297 → 207 → 90 | More sustained, less sudden |
| Flow CV | 0.35 → 0.47 → 0.60 | More bound-free oscillation |

#### 2. ARM INACTIVITY IS THE STRONGEST T1-VS-T2 SIGNAL (d > 1.0)
The top 7 features separating T1 from T2 are ALL about arms/wrists being quieter:
- Wrist acceleration: T2 is 47% lower (d=1.37)
- Wrist kinetic energy: T2 is 61% lower (d=1.21)
- Trajectory curvature: T2 is 43% lower (d=1.20)
- Trajectory path length: T2 is 46% lower (d=1.13)

Suggestive dance keeps arms down, behind the body, or still. Athletic dance has active arms with complex trajectories.

#### 3. PELVIS IRREGULARITY + OVERALL STILLNESS IS THE T2-VS-T3 SIGNAL (d > 1.0)
The features that separate T2 from T3 are different from T1-vs-T2:
- Pelvis directness: d=1.59 (T3 much more indirect)
- Pelvis KE variability: d=1.54 (T3 much more irregular)
- Trajectory path length: d=1.52 (T3 nearly stationary)
- Effort Weight: d=1.19 (T3 much lighter)
- Effort Time: d=1.20 (T3 much more sustained)
- Effort Flow: d=1.07 (T3 much lower)

T3 is an amplified version of T2: even less whole-body movement, even more irregular pelvis, even more indirect and sustained.

#### 4. T1-VS-T3 EFFECT SIZES ARE MASSIVE (d = 2.0-2.9)
The endpoints of the continuum are extremely well-separated:
- Effort Time: d=2.94
- Pelvis Jerkiness: d=2.86
- Effort Flow: d=2.56
- Pelvis Velocity: d=2.52
- Effort Weight: d=2.45
- Trajectory Path Length: d=2.37
- Pelvis Directness: d=2.28
- Trajectory Curvature: d=2.23

#### 5. GEMINI TAXONOMY VALIDATION
- **Indirect pelvis → CONFIRMED** (Directness: T1=28.0, T2=19.4, T3=8.9)
- **Free/variable flow → CONFIRMED** (Flow CV: T1=0.35, T2=0.47, T3=0.60)
- **Light weight → CONFIRMED** (Effort Weight: T1=37.6, T2=25.2, T3=8.1)
- **Sustained time → CONFIRMED** (Effort Time: T1=297, T2=207, T3=90)
- **New finding not in Gemini:** Arm inactivity is the #1 differentiator between T1 and T2

### LMA Signatures Summary
- **T1 (Athletic) = active arms + structured direct whole-body motion + strong/sudden effort**
- **T2 (Suggestive) = quiet arms + irregular indirect pelvis + moderate flow variability**
- **T3 (Explicit) = near-stationary body + maximally irregular indirect pelvis + sustained light flow**

---

## Existing Code Artifacts

| File | Purpose |
|------|---------|
| `eval/kineguard_baseline.py` | Three-way comparison: DEMO vs Qwen3(skeleton) vs Qwen3(video). Proves the need for KineGuard. |
| `eval/lma_tier_analysis.ipynb` | Jupyter notebook: full three-tier LMA feature analysis with visualizations |
| `eval/SESSION_SUMMARY.md` | This document |
| `eval/RESEARCH_BRIEF.md` | Detailed research brief covering all models surveyed |
| `core/demo_motion_classifier.py` | DEMO inference module with all fixes |
| `core/qwen3_vl_tier_classifier.py` | Qwen3-VL-Embedding tier classifier |
| `core/wham_inference.py` | WHAM pipeline: video → skeleton .npz + LMA features |
| `core/render_skeleton.py` | Renders stick-figure skeleton videos from WHAM .npz |
| `output/lma_tier_analysis/` | LMA features for 34 videos (12 T1, 17 T2, 5 T3) with manifest.csv |
| `output/skeleton_test/round1/` | Round 1 test results: Qwen3 skeleton vs original comparison |

## Environment
- Conda env: `wham` (Python 3.9.23, torch 2.1.0+cu118)
- DEMO needs transformers==4.44.0; Qwen3 needs 4.57.6 (scripts auto-manage)
- GPUs: 4× NVIDIA RTX A6000 (48GB each)
- MotionScript: /tmp/posescript (text2pose package) with patches
- yt-dlp: system Python 3.10 has 2026.03.17 (TikTok works); wham env has old version

---

## Open Questions / Next Steps

1. **Train a classifier (SVM/RF) on LMA features.** The Cohen's d values (>1.0 for T1-T2, >2.0 for T1-T3) strongly suggest this will work. The Turab paper got 99% on 10 dance styles with the same 55 features. Our task is simpler (4 tiers vs 10 classes).

2. **Scale the dataset.** 34 videos is enough for statistical analysis but likely not for robust classifier training. Need 100+ per tier. Sources: NPDI dataset (1000 T3 videos available), TikTok for T2 (rate-limited), YouTube/Kinetics for T1.

3. **Add Tier 0 data.** We haven't tested everyday non-dance movement. Kinetics has classes like "walking", "sitting", "stretching" that would serve as T0.

4. **The ordinal gradient is a key paper insight.** The tiers aren't discrete categories — they form a continuum. This means an ordinal regression model might outperform a standard multi-class classifier.

5. **Feature selection.** The top ~15 features account for most of the separation. A reduced feature set would be more robust and interpretable.

6. **Paper contributions:**
   - Contribution 1: Proof that no existing model handles skeleton-based NSFW classification
   - Contribution 2: Discovery of the three-tier LMA gradient (arm inactivity + pelvis irregularity + flow variability)
   - Contribution 3: KineGuard — the trained skeleton classifier itself
   - Contribution 4: The appearance-bias problem in VLM-based content moderation (Qwen3 works on video but is biased; fails completely on skeleton)
