# KineGuard Pipeline Proposal — OMIL-Based Skeleton Classifier

## Context

We have empirically validated that Laban Movement Analysis (LMA) features extracted from 3D skeleton data carry sufficient signal to distinguish between dance tiers. On 34 videos (12 Tier 1, 17 Tier 2, 5 Tier 3), we found:

- **Tier 1 → Tier 2 separation:** Cohen's d > 1.0 on arm inactivity features, d ≈ 0.8-1.0 on pelvis irregularity
- **Tier 2 → Tier 3 separation:** Cohen's d > 1.0 on pelvis directness, trajectory length, effort measures
- **Tier 1 → Tier 3 separation:** Cohen's d = 2.0-2.9 (massive, trivially separable)
- The tiers form a **monotonic ordinal gradient**, not discrete categories

No existing motion-language model can perform this classification. DEMO (motion LLM) fails entirely. Qwen3-VL fails on skeleton video. The signal exists in the LMA features but no off-the-shelf classifier can extract it.

## The Problem: Noisy Web-Scraped Labels

To train a skeleton-based classifier, we need labeled data at scale. The practical data source is web scraping (TikTok, YouTube), using search queries and video tags as proxy labels:

- Search "twerk tutorial" → most results are Tier 2, but some are fitness/comedy/unrelated
- Search "K-pop dance cover" → most results are Tier 1, but some contain suggestive elements
- NPDI/NSPD porn datasets → reliable Tier 3 labels (curated academic dataset)

**The core challenge:** Search queries provide **bag-level labels** (the query intent), not **instance-level labels** (whether each specific video matches the tier). A standard classifier trained on these noisy labels would learn the noise.

## Proposed Pipeline

### Phase 1: Data Collection & Feature Extraction

```
Web Scraper (TikTok/YouTube)
    │
    ├── Query: "twerk tutorial" ──────→ Bag labeled T2 (but noisy)
    ├── Query: "K-pop dance cover" ───→ Bag labeled T1 (but noisy)
    ├── Query: "hip hop choreography" → Bag labeled T1 (but noisy)
    ├── Query: "sensual floorwork" ───→ Bag labeled T2 (but noisy)
    └── ...
    │
    ▼
WHAM (3D Skeleton Extraction)
    │
    ├── 3D joints (T, 31, 3) per frame
    ├── SMPL vertices (T, 6890, 3) per frame
    ├── Floor estimation
    └── Per-person fragment segmentation
    │
    ▼
LMA Feature Extraction (55 features per window)
    │
    ├── Body: joint distances, angles, initiation
    ├── Effort: Weight (KE), Time (acceleration), Flow (jerk), Space (directness)
    ├── Shape: body volume (ConvexHull)
    └── Space: trajectory, curvature, spatial dispersion
    │
    ▼
Feature Vector per video: (N_windows, 55) → summarized to per-video statistics
```

Additionally, for Tier 3 data, we have curated academic datasets (NPDI: 1000 videos, NSPD: 2000 videos) with reliable labels.

### Phase 2: OMIL (Ordinal Multiple Instance Learning) Training

#### Why OMIL, Not Standard Classification?

1. **Multiple Instance Learning (MIL):** Each search query produces a "bag" of videos. The bag-level label (the query) is known, but the instance-level labels (each video) are not. MIL learns from bag-level supervision — it assumes at least some instances in a positive bag are truly positive, without requiring per-instance annotation.

2. **Ordinal:** The tiers are not independent classes — they form an ordered scale (T0 < T1 < T2 < T3). Our LMA analysis proved this empirically: the features form a monotonic gradient. An ordinal formulation respects this structure, constraining the model to learn that T2 is "between" T1 and T3, not an unrelated category.

3. **Combined (OMIL):** Web-scraped data is both noisy (MIL handles this) and ordinal (ordinal regression handles this). OMIL addresses both simultaneously.

#### OMIL Formulation

**Input per video:**
- LMA feature vector: 55 features × summary statistics (mean, std, CV) = ~165 dimensions
- Optionally: raw skeleton sequence (T, 31, 3) for a learned feature extractor

**Bag construction:**
- Each search query defines a bag: {video_1, video_2, ..., video_K}
- Bag label = the tier implied by the query (e.g., "twerk" → T2)
- Instance labels are unknown (some videos in a "twerk" bag may actually be T1 or T0)

**Ordinal constraint:**
- Instead of 4 independent classes, learn 3 ordered thresholds: θ₁, θ₂, θ₃
- P(tier ≥ k) = σ(f(x) - θ_k) for k ∈ {1, 2, 3}
- This enforces: if a video scores high enough for T3, it automatically scores for T2 and T1

**MIL aggregation:**
- For a bag B with label T2: at least one instance should have P(tier ≥ 2) > 0.5
- Negative bags (T0/T1 queries): all instances should have P(tier ≥ 2) < 0.5
- Standard MIL pooling (max, attention-based, or top-K) selects the "most positive" instances

**Loss function:**
- Ordinal cross-entropy on bag-level predictions
- MIL constraint: aggregate instance predictions to bag prediction via pooling
- Optional: entropy regularization to prevent all instances being classified the same

#### Model Architecture Options

**Option A: LMA features → MLP (simplest)**
```
LMA features (165-dim) → MLP(165, 128, 64, 1) → ordinal thresholds → tier
```
- Fast, interpretable, needs least data
- Uses hand-crafted features proven to separate tiers

**Option B: Skeleton sequence → ST-GCN → ordinal head**
```
Raw joints (T, 31, 3) → ST-GCN → learned features → ordinal thresholds → tier
```
- Learns its own features, potentially more powerful
- Needs more data, less interpretable
- Can capture temporal patterns LMA might miss

**Option C: Hybrid — LMA + learned skeleton features**
```
Raw joints (T, 31, 3) → ST-GCN → 128-dim
LMA features (165-dim) → MLP → 64-dim
Concat(128, 64) → ordinal head → tier
```
- Best of both: proven hand-crafted features + learned temporal patterns
- Most data-hungry but most powerful

### Phase 3: Inference (Deployment)

```
New video
    │
    ▼
WHAM → 3D skeleton + floor
    │
    ▼
LMA features (+ optional skeleton sequence)
    │
    ▼
Trained OMIL classifier → Tier prediction (0/1/2/3)
```

**Key properties of the deployed system:**
- **Appearance-independent:** Only sees skeleton + LMA, never raw pixels
- **Privacy-preserving:** No raw video stored or processed at inference time
- **Lightweight:** LMA extraction is CPU-only; classifier is a small MLP or GCN
- **Interpretable:** LMA features have semantic meaning (effort, shape, space)

## LMA Features That Drive Classification

Based on our empirical analysis (34 videos), the most discriminative features are:

### T1 vs T2 (Artistic vs Suggestive)

| Feature | Cohen's d | What it measures |
|---|---|---|
| L_WRIST_Accel_mean | 1.37 | Arm activity level |
| L_WRIST_KE_mean | 1.21 | Arm kinetic energy |
| Traj_Curvature_mean | 1.20 | Movement path complexity |
| Traj_Path_Length_mean | 1.13 | Total distance traveled |
| PELVIS_KE_cv | 1.01 | Pelvis motion irregularity |
| PELVIS_Directness_mean | 0.84 | Pelvis spatial focus (direct vs indirect) |

**Key insight:** T2 = quiet arms + irregular indirect pelvis. T1 = active arms + structured direct whole-body motion.

### T2 vs T3 (Suggestive vs Explicit)

| Feature | Cohen's d | What it measures |
|---|---|---|
| PELVIS_Directness_mean | 1.59 | Pelvis becomes maximally indirect |
| PELVIS_KE_cv | 1.54 | Pelvis becomes maximally irregular |
| Traj_Path_Length_mean | 1.52 | Body becomes nearly stationary |
| Effort_Weight_Global_mean | 1.19 | Movement becomes very light |
| Effort_Time_Global_mean | 1.20 | Movement becomes very sustained |

**Key insight:** T3 is an amplified T2 — even less whole-body movement, even more irregular pelvis, maximally sustained and light.

### Three-Tier Gradient

```
Feature                    T1 (athletic)   T2 (suggestive)   T3 (explicit)
---------------------------------------------------------------------------
Trajectory Path Length          4.12            2.21             0.74
Pelvis KE Variability           0.55            0.84             1.84
Pelvis Directness              28.0            19.4              8.9
Effort Time                   297             207               90
Effort Weight                  37.6            25.2              8.1
Flow CV                        0.35            0.47             0.60
```

## Data Availability

### Currently Processed
- **Tier 1:** 12 videos (1MILLION, RedBull BC One, YouTube hip-hop/breakdancing)
- **Tier 2:** 17 videos (TikTok twerk, dancehall, perreo, chairdance, bellydance, heels, bachata)
- **Tier 3:** 1000 videos processing now (NPDI porn dataset, 3 GPUs, ~5-6 days ETA)
  - Additional 2000 videos available (NSPD dataset, to be queued later)

### Data Scaling Needed
- T1 and T2 need significant expansion (100+ videos each minimum)
- Sources: TikTok (rate-limited), YouTube (works), Kinetics dataset (labeled dance classes)
- OMIL reduces the labeling burden — only bag-level (query-level) labels needed, not per-video

## Theoretical Grounding

### Laban Movement Analysis (LMA)
The LMA framework (Body, Effort, Shape, Space) provides the feature vocabulary. Our empirical results validate predictions from choreographic theory:

- **Effort Flow:** Suggestive motion has higher bound↔free oscillation (Flow CV: 0.35 → 0.47 → 0.60)
- **Effort Weight:** Suggestive/explicit motion uses lighter weight (37.6 → 25.2 → 8.1)
- **Effort Space:** Suggestive motion is more indirect (Pelvis Directness: 28.0 → 19.4 → 8.9)
- **Effort Time:** Suggestive/explicit motion is more sustained (297 → 207 → 90)
- **New finding:** Arm inactivity (d=1.37) is the strongest single signal, not predicted by prior theory

### Ordinal Multiple Instance Learning
OMIL combines:
- **Ordinal regression** (Frank & Hall 2001; Niu et al. 2016): respects the ordered structure of tiers
- **Multiple instance learning** (Dietterich et al. 1997; Ilse et al. 2018): handles noisy bag-level supervision
- This combination is novel for motion classification — prior MIL work focuses on image/video bags, not skeleton-derived feature bags

## Open Questions for Discussion

1. **Which model architecture?** Option A (LMA→MLP) is fastest to implement and test. But does it leave performance on the table compared to Option C (hybrid)?

2. **Bag construction strategy:** How many videos per query bag? Should we use fixed queries or also cluster by visual similarity?

3. **Ordinal vs multi-class:** The gradient is clear on our 34-video analysis, but will it hold at scale? Should we start with binary (suggestive vs not) and expand to ordinal later?

4. **Temporal modeling:** LMA features are computed per sliding window then summarized per video. Should the classifier see the temporal sequence of LMA windows, or is the per-video summary sufficient?

5. **Threshold calibration:** The ordinal model produces P(tier ≥ k). How do we set the decision thresholds for deployment? ROC-based? Application-specific (high precision for T3)?

6. **Tier 0 data:** We haven't collected or analyzed Tier 0 (everyday non-dance movement). Kinetics/AMASS could provide this. Is it needed for training, or is the T1/T2/T3 distinction sufficient?
