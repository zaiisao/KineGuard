# KineGuard Full Experiment Report
**Date:** April 6, 2026
**Author:** Jaehoon Ahn, Sogang University

---

## 1. Project Goal

KineGuard aims to classify human motion into 4 tiers based purely on 3D skeleton data — no visual appearance:
- **Tier 0:** Normal everyday movement (walking, sitting, waving)
- **Tier 1:** Artistic dance/athletics (K-pop, breakdancing, hip-hop choreography)
- **Tier 2:** Suggestive/sensual dance (twerking, floorwork, pelvic rolls, chair dance)
- **Tier 3:** Explicit sexual content (pornography)

The key constraint: the classifier must be **appearance-independent**. The same hip isolation in a bikini vs a tracksuit must receive the same tier. Skeleton representation strips away clothing, skin, camera angle, and environment.

**Pipeline:** Video → WHAM (3D skeleton + floor estimation) → LMA feature extraction (55 features/frame) → Classifier (OMIL-based, to be trained)

---

## 2. Experiments: What We Tried and What Failed

### 2.1 DEMO (Dense Motion Captioning LLM) — COMPLETE FAILURE

**What:** LLaMA-3.1-8B-Instruct with a 2-layer MLP motion adapter. Trained on AMASS/HumanML3D. Takes (T, 22, 3) SMPL joint sequences, outputs text captions.

**Technical fixes required to even run it:**
- WHAM outputs 31 Halpe joints, not SMPL 22. Built a verified mapping (WHAM31 → SMPL22) using euclidean distance comparison against SMPL rest pose.
- `config.pretrain_mm = None` — training-time checkpoint path doesn't exist in inference
- `torch_dtype=torch.float32` — avoids Float/BFloat16 mismatch in vision tower MLP
- Removed `rotate_y_up_to_z_up` — training code had this commented out, inference must match
- Manual greedy decoding — DEMO's custom `generate()` is broken; used `prepare_inputs_labels_for_multimodal` + token-by-token KV-cache generation
- `LlamaForCausalLM.generate(model, ...)` bypass for text-only `classify_tier()` prompts

**Results on 8 test videos:**
```
chairdance  → "gets on the floor and does swimming movements"       Tier: N/A
twerk       → "lays down on the ground and swim"                    Tier: N/A
fancam      → "doing the military crawl backwards"                  Tier: N/A
dancehall   → "rolls backwards happily, military crawl backwards"   Tier: 0
bellydance  → "doing the military crawl backwards"                  Tier: N/A
heels       → "laying down on their back, swimming"                 Tier: N/A
fitness     → "doing the military crawl backwards"                  Tier: N/A
bachata     → "gets on floor, swimming movements"                   Tier: N/A
```

7/8 videos returned unparseable tiers. Every caption mentions "military crawl", "swimming", or "laying down" — the model maps all dance postures to the nearest AMASS training example.

**Root cause:** AMASS/HumanML3D contains zero dance content. The model literally has no vocabulary for dance movement.

---

### 2.2 Qwen3-VL-Embedding-2B on Original Video — WORKS BUT APPEARANCE-BIASED

**What:** Vision-language embedding model. Zero-shot cosine similarity between video embedding and tier text prompt embeddings.

**Results on 8 test videos (original video input):**
```
Video        GT      Best   T0     T1     T2     T3
chairdance   T2      tier2  0.388  0.414  0.579  0.256
twerk        T2      tier2  0.369  0.399  0.563  0.245
dancehall    T2      tier2  0.399  0.432  0.510  0.271
bellydance   T1/T2   tier2  0.383  0.419  0.525  0.252
heels        T1/T2   tier2  0.359  0.394  0.447  0.281
bachata      T1/T2   tier2  0.371  0.396  0.442  0.258
fancam       T1      tier2  0.362  0.397  0.399  0.267  ← should be T1
fitness      T0/T1   tier2  0.372  0.397  0.402  0.243  ← should be T0/T1
```

It correctly identifies most T2 content but misclassifies fancam (T1) and fitness (T0/T1) as T2. The model detects visual appearance — skin exposure, camera framing, performance context — not motion patterns. This is exactly the bias KineGuard must eliminate.

---

### 2.3 Qwen3-VL on Rendered Skeleton Video — COMPLETE FAILURE

**What:** Same model, but given stick-figure skeleton videos rendered from WHAM output instead of original video.

**Tested on 16 videos:** 8 Tier 2 + 8 Tier 1 (K-pop, street dance, breakdancing from 1MILLION, RedBull BC One, etc.)

**Tier 2 skeleton results:**
```
chairdance   tier1  T0=0.348  T1=0.402  T2=0.385  T3=0.367
twerk        tier1  T0=0.363  T1=0.414  T2=0.401  T3=0.376
dancehall    tier1  T0=0.356  T1=0.418  T2=0.391  T3=0.361
bellydance   tier1  T0=0.411  T1=0.452  T2=0.429  T3=0.376
heels        tier1  T0=0.393  T1=0.439  T2=0.418  T3=0.390
bachata      tier1  T0=0.372  T1=0.434  T2=0.422  T3=0.381
fancam       tier1  T0=0.364  T1=0.419  T2=0.404  T3=0.383
fitness      tier1  T0=0.370  T1=0.418  T2=0.396  T3=0.372
```

**Tier 1 skeleton results:**
```
7414882..    tier1  T0=0.327  T1=0.381  T2=0.370  T3=0.326
7434898..    tier1  T0=0.395  T1=0.444  T2=0.425  T3=0.402
7444218..    tier1  T0=0.370  T1=0.417  T2=0.399  T3=0.386
7449029..    tier1  T0=0.358  T1=0.397  T2=0.381  T3=0.366
7449729..    tier0  T0=0.475  T1=0.474  T2=0.453  T3=0.405
7462700..    tier1  T0=0.399  T1=0.447  T2=0.427  T3=0.383
7463199..    tier1  T0=0.329  T1=0.387  T2=0.373  T3=0.334
7478514..    tier1  T0=0.368  T1=0.422  T2=0.402  T3=0.379
```

**All 16 videos classified as tier1.** T2 avg tier2 score = 0.406, T1 avg tier2 score = 0.404. Gap = +0.002 (zero separation). The stick figure carries no visual NSFW signal for embedding models.

---

### 2.4 MotionScript (Rule-Based Skeleton→Text) — PARTIAL SUCCESS

**What:** Rule-based system (IROS 2025) that converts 3D joint coordinates to structured natural language using "posecodes" (joint angles, distances, relative positions) and "motioncodes" (temporal changes).

Required 6+ patches to work with our WHAM 22-joint input (prepare_input index mismatch, p_eligibility scope bug, m_eligibility data structure mismatch, import path fixes, device fixes, eligibility adjustment bugs). Ultimately bypassed aggregation and motioncode stages; used posecodes only.

**Sample output (twerk video, frame 0):**
> "left hand is apart from the right. left foot is further up than left hip with left knee near right knee. left upper arm is flat and left thigh is aligned horizontally. right hand is stretched backwards and left foot is stretched forwards. left knee is in l-shape with left elbow unbent while right knee is forming an l shape."

**MotionScript → Claude (blind reasoning LLM):**
A Claude agent given ONLY the MotionScript text (no video name, no tier info, no context) classified the twerk video as **Tier 2 with HIGH confidence**, identifying:
- Torso repeatedly horizontal (bent-over posture)
- Hands behind body / below hips (not athletic arm positioning)
- Rhythmic hip oscillation pattern
- Transition to kneeling with sustained hip motion
- Minimal upper-body choreography

**MotionScript → Qwen3-VL embedding similarity:** FAILED. Picked tier1 (0.653) over tier2 (0.638). Embedding similarity is too coarse for geometric text.

**Conclusion:** The geometric signal IS in the skeleton data. A reasoning LLM can extract it from text descriptions. But embedding-similarity classifiers cannot — the task requires reasoning, not pattern matching.

---

### 2.5 Motion LLM Literature Survey — NO EXISTING MODEL SOLVES THIS

Models surveyed:
- MotionLLM (IDEA Research, 2024) — video+motion co-training with MoVid dataset (includes dance)
- M3GPT (NeurIPS 2024) — unified text+music+motion, music-to-dance
- MG-MotionLLM (CVPR 2025) — multi-granularity, 28 motion tasks
- MotionGPT (NeurIPS 2023) — motion as "foreign language" for LLMs
- Large Motion Model (ECCV 2024) — diffusion transformer with ArtAttention
- Being-M0 / MotionLib (ICML 2025) — 1.2M motion clips dataset
- MotionScript (IROS 2025) — rule-based skeleton-to-text
- Superman (2026) — vision-guided motion tokenizer
- ViMoNet (2025) — video+motion alignment

**Finding:** No published model includes "suggestive" vs "artistic" in its training vocabulary. All motion-language models are trained on HumanML3D/AMASS (walking, sitting, picking up objects) or AIST++ (dance style labels without NSFW annotation). The suggestive-vs-artistic distinction does not exist in any motion-language dataset. This is the research gap KineGuard fills.

---

## 3. The Breakthrough: LMA Feature Analysis

### 3.1 Background

Our WHAM pipeline already extracts 55 Laban Movement Analysis (LMA) features per frame window covering:
- **Body:** joint distances, angles, initiation detection
- **Effort:** Weight (kinetic energy), Time (acceleration), Flow (jerk), Space (directness)
- **Shape:** body volume via ConvexHull
- **Space:** trajectory path, curvature, spatial dispersion

Turab et al. (arXiv:2504.21166, April 2025) achieves 99.18% accuracy on 10 dance style classification using the same LMA features + SVM/Random Forest on AIST++.

### 3.2 Gemini Deep Research Taxonomy

We commissioned Gemini Deep Research to analyze what makes motion suggestive vs artistic through the LMA framework. Key predictions:

| LMA Factor | Tier 1 (Athletic) | Tier 2 (Suggestive) |
|---|---|---|
| Flow | Bound (controlled, stoppable) | Free (released, continuous) |
| Weight | Strong (powerful, grounded) | Light (yielding, soft) |
| Time | Sudden (impactful, quick) | Sustained (lingering, slow) |
| Space | Direct (goal-oriented) | Indirect (diffuse, performative) |

Additional markers: pelvic isolation with sustained time + free flow, "sensual release" cycles (bound→free transitions), narrowing kinesphere to sexualized zones.

### 3.3 Pilot Study: 34 Videos (12 T1, 17 T2, 5 T3)

**Data:**
- T1: 1MILLION Dance Studio, RedBull BC One, DanceWorldStyles, K-pop choreography, YouTube hip-hop, fitness
- T2: TikTok twerk, dancehall, perreo/reggaeton, chairdance, bellydance, heels/floorwork, bachata
- T3: NPDI pornography dataset (5 videos)

**Three-Tier Feature Comparison (Mean Values):**
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
```

**Effect Sizes (Cohen's d) — Pairwise:**
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
Flow CV (bound-free)          0.77        0.42        0.89  ***
```

**Pilot Key Findings:**
1. Nearly every feature forms a **monotonic ordinal gradient** T1 → T2 → T3
2. **Arm inactivity** is the #1 T1-vs-T2 signal (d=1.37), not pelvis activity
3. T1-vs-T3 effect sizes are massive (d=2.0-2.9)
4. T2 sits cleanly between T1 and T3 on the continuum

---

### 3.4 Full-Scale Study: 841 Videos (220 T2, 621 T3)

After the pilot, we scaled up data collection and processing:
- **T3:** 1000 NPDI pornography videos processed through YOLO pre-filter + WHAM + LMA (3 GPUs, ~2 days). 621 produced usable LMA features, 379 failed (no visible full-body humans after filtering).
- **T2:** 916 videos crawled from TikTok channels (twerk, dancehall, perreo, heels, bellydance, bachata) and YouTube searches. Processed through same pipeline. 220 produced usable LMA features.
- **YOLO pre-filtering** reduced processing time ~4x by skipping frames without visible humans (porn videos average only 23% usable content; dance videos average ~80%).

**Full-Scale T2 vs T3 Results (220 T2 vs 621 T3):**

```
Feature                      T2 mean    T3 mean   T2/T3    d     Direction
---------------------------------------------------------------------------
Body Volume CV                0.169      0.112   1.50x   0.85   T2>T3  ***
Traj Path Length              2.089      1.075   1.94x   0.78   T2>T3  **
Effort Flow                4550.049   1901.827   2.39x   0.76   T2>T3  **
Effort Time                 200.476    107.926   1.86x   0.76   T2>T3  **
Head Velocity                 1.609      0.852   1.89x   0.75   T2>T3  **
Pelvis Jerkiness            723.408    314.842   2.30x   0.73   T2>T3  **
Head Jerkiness              621.635    221.725   2.80x   0.70   T2>T3  **
Ankle Velocity                2.009      1.272   1.58x   0.69   T2>T3  **
Pelvis Acceleration          31.507     17.901   1.76x   0.68   T2>T3  **
Effort Space                114.627     82.115   1.40x   0.63   T2>T3  **
Pelvis Velocity               1.655      1.102   1.50x   0.62   T2>T3  **
Traj Curvature                5.757      3.943   1.46x   0.60   T2>T3  **
Pelvis Directness            16.361     11.713   1.40x   0.56   T2>T3  **
Effort Weight                23.585     12.556   1.88x   0.56   T2>T3  **
Hand Distance                 0.423      0.407   1.04x   0.51   T2>T3  **
```

**Features that did NOT separate T2 from T3 (d < 0.3):**
```
Flow CV                       0.529      0.572   0.93x   0.11   ~same
Body Volume                   0.250      0.241   1.04x   0.17   ~same
Pelvis Rhythmicity            0.815      0.858   0.95x   0.20   ~same
Pelvis/Ext Ratio              1.337      1.332   1.00x   0.01   ~same
Arm/Pelvis KE Ratio           0.464      0.480   0.97x   0.02   ~same
```

---

### 3.5 Gemini Taxonomy Validation at Scale

| Gemini Prediction | Pilot (5 T3) | Full Scale (621 T3) | Status |
|---|---|---|---|
| T3 has lower Effort Flow (less overall flow) | Confirmed (d=1.07) | Confirmed (d=0.76) | **CONFIRMED** |
| T3 has lower Effort Weight (lighter) | Confirmed (d=1.19) | Confirmed (d=0.56) | **CONFIRMED** |
| T3 has lower Effort Time (more sustained) | Confirmed (d=1.20) | Confirmed (d=0.76) | **CONFIRMED** |
| T3 has lower Pelvis Directness (more indirect) | Confirmed (d=1.59) | Confirmed (d=0.56) | **CONFIRMED** |
| T3 has higher Flow CV (more bound-free oscillation) | Confirmed (d=0.42) | Very weak (d=0.11) | **WEAK at scale** |
| T3 has higher Pelvis KE Variability | Confirmed (d=1.54) | Weak (d=0.29) | **WEAKER at scale** |

### 3.6 Key Differences Between Pilot and Full-Scale

1. **Effect sizes decreased** from pilot (d=1.0-1.5) to full scale (d=0.5-0.85). This is expected — the pilot had hand-picked "clean" examples; the full dataset has noise from web scraping.

2. **New #1 separator at scale: Body Volume CV** (d=0.85). T2 dancers change body shape more (expanding/contracting during dance). T3 bodies maintain more consistent shape. This wasn't visible in the 5-video pilot.

3. **Pelvis jerkiness reversed direction.** Pilot: T3 had less jerk than T2 (d=1.10, T2>T3). Full scale: same direction confirmed (d=0.73, T2>T3). Suggestive dance has MORE percussive/jerky pelvis movement than explicit content. T3 motion is smoother/more sustained.

4. **Head velocity emerged** as a strong separator (d=0.75). T3 heads are nearly still; T2 dancers move their heads actively (part of the performance/expression).

5. **Arm/pelvis ratio and pelvis energy ratio showed NO separation** between T2 and T3 (d<0.02). Both tiers have similar energy distribution. The arm inactivity signal separates T1 from T2, not T2 from T3.

---

## 4. LMA Signatures (Final)

Based on both pilot and full-scale analysis:

**Tier 1 (Athletic):**
- Active arms with complex trajectories (Wrist Accel ~9.1, KE ~0.40)
- Structured, direct whole-body motion (Pelvis Directness ~28, Traj Path ~4.1)
- Strong, sudden effort (Weight ~37.6, Time ~296)
- High trajectory curvature (~10.9) — complex spatial paths

**Tier 2 (Suggestive):**
- Quiet arms (Wrist Accel ~4.3-6.0, KE ~0.15-0.26) — 47-61% lower than T1
- Irregular, indirect pelvis motion (Directness ~16-19, KE CV ~0.84-0.92)
- Moderate flow/weight/time — between T1 and T3
- High body volume variability (CV ~0.17) — body shape changes during dance
- More percussive/jerky pelvis and head than T3

**Tier 3 (Explicit):**
- Near-stationary body (Traj Path ~0.74-1.08) — 49-82% less than T2
- Very low effort across all factors (Flow ~1384-1902, Weight ~8-12.6, Time ~90-108)
- Most indirect pelvis (Directness ~8.9-11.7)
- Head nearly still (Velocity ~0.85 vs T2's 1.61)
- Low body volume variability (CV ~0.11) — consistent shape
- Smooth/sustained movement, less jerky than T2

---

## 5. Infrastructure & Data Processing

### 5.1 YOLO Pre-Filtering
We integrated YOLO11x-pose as a pre-filter before WHAM. For each video:
1. YOLO scans every frame for visible humans (≥10 keypoints, ≥0.5 avg confidence)
2. Continuous segments ≥3 seconds are identified
3. Only those segments are clipped (ffmpeg) and passed to WHAM

**Impact:** Porn videos average only 23% usable content (much is close-ups, scene transitions, etc.). Dance videos average ~80%. This reduced WHAM processing time ~4x and eliminated GPU memory issues from long videos.

### 5.2 Data Scale

| Tier | Source | Videos Downloaded | WHAM+LMA Processed | Failed |
|---|---|---|---|---|
| T1 | YouTube + TikTok | 41 | Pending (external curation) | — |
| T2 | TikTok channels + YouTube | 916 | 220 usable | 696 |
| T3 | NPDI academic dataset | 1000 | 621 usable | 379 |

### 5.3 Processing Infrastructure
- 4× NVIDIA RTX A6000 (48GB each)
- GPU 0: reserved for other users
- GPUs 1-3: batch processing via tmux (survives SSH disconnects)
- Thermal protection: pauses at 82°C, resumes at 75°C
- Failed videos get `_FAILED` marker files for skip-on-retry

---

## 6. Proposed Pipeline: OMIL Classifier

### Why OMIL (Ordinal Multiple Instance Learning)?

1. **MIL (Multiple Instance Learning):** Web-scraped labels are noisy. A "twerk tutorial" search produces a "bag" of videos — most are T2 but some are fitness/comedy/unrelated. MIL learns from bag-level labels without per-instance annotation.

2. **Ordinal:** The tiers form an ordered scale (T0 < T1 < T2 < T3), not independent classes. Our LMA analysis proved this empirically — features form a monotonic gradient. Ordinal regression constrains the model to respect this ordering.

### Architecture Options

**Option A: LMA features → MLP**
- 55 features × summary stats (mean, std, CV) = ~165 dimensions → MLP → ordinal thresholds
- Fast, interpretable, proven features

**Option B: Raw skeleton → ST-GCN**
- Joint positions (T, 31, 3) → Spatio-Temporal GCN → learned features → ordinal head
- Potentially more powerful, needs more data

**Option C: Hybrid**
- Both LMA features and learned skeleton features → combined ordinal head

---

## 7. Open Questions

1. **T2 vs T3 separation is harder than T1 vs T2.** Pilot showed d>1.0, full scale shows d=0.5-0.85. Is this enough for a classifier, or do we need better features?

2. **High T2 failure rate (696/916).** Many T2 videos failed WHAM processing. Is this because dance videos have fast motion / unusual poses that WHAM can't handle? Or is it the YOLO filter being too strict?

3. **T1 data is severely lacking** (41 videos). Someone else is curating T1 data, but we need 100+ for balanced training.

4. **Noisy T2 labels.** YouTube search queries produce mixed content. How does this affect the LMA statistics? Some "T2" videos might actually be T1 or T0.

5. **The ordinal gradient suggests ordinal regression > multi-class.** But should we start with binary (suggestive vs not) and expand later?

6. **Feature selection.** Not all 55 LMA features separate tiers. Should we use only the top 15 (d>0.5) for a more robust, interpretable model?

7. **Body Volume CV was the #1 full-scale separator** (d=0.85) but wasn't notable in the pilot. How stable is this feature across different video sources?
