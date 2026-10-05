# Comprehensive Evaluation Report: Color-Invariant Saree Design Recognition

## Executive Summary
This report provides the full evaluation of the **Color-Invariant Saree Design Recognition** system for the AIE-CASE benchmark. All metrics and results reported herein are generated directly from empirical inference and evaluation runs on the local dataset using reproducible PyTorch scripts.

---

## 1. Dataset Description
The system is evaluated on the primary local handloom saree dataset located at `data/raw/handloom_sarees/`.
- **Domain**: Traditional Indian handloom sarees across six distinct weaving traditions:
  1. Banarasi
  2. Chanderi
  3. Ikat
  4. Jamdani
  5. Kanjivaram
  6. Patola
- **Pattern & Colorway Structure**: Each design pattern is rendered in 8 distinct colorways (`palette_00.png` through `palette_07.png`), with accompanying metadata sidecars documenting RGB foreground and background colors.
- **Image Specifications**: 224 × 224 pixels, RGB PNG format, zero corrupt or missing files.

---

## 2. Dataset Size & Canonical Deduplication Audit
- **Raw Folders**: 360 directories, 2,880 total images.
- **Deduplication Audit Finding**: Procedural generation analysis revealed that Kanjivaram and Patola contain replicated identical patterns across multiple folder indices (16 unique patterns replicated across 60 folders each).
- **Canonical Design Count**: Exactly **272 unique canonical design patterns** (60 Banarasi, 60 Chanderi, 60 Ikat, 60 Jamdani, 16 Kanjivaram, 16 Patola).
- **Canonical Image Count**: **2,176 unique canonical images** across 8 colorways.

---

## 3. Gallery Size
- **Gallery Images**: **168 images**
- **Design Coverage**: 42 unseen test designs across all 6 traditions.
- **Colorway Allocation**: Reference colorways `palette_00`, `palette_01`, `palette_02`, `palette_03` (4 colorways per design).

---

## 4. Query Size
- **Query Images**: **168 images**
- **Design Coverage**: 42 unseen test designs.
- **Colorway Allocation**: Evaluation colorways `palette_04`, `palette_05`, `palette_06`, `palette_07` (4 colorways per design, disjoint from gallery).

---

## 5. Train Size
- **Train Designs**: **190 canonical designs** (~70% stratified split).
- **Train Images**: **1,520 images** (190 designs × 8 colorways).

---

## 6. Validation Size
- **Validation Designs**: **40 canonical designs** (~15% stratified split).
- **Validation Images**: **320 images** (40 designs × 8 colorways).

---

## 7. Test Size
- **Test Designs**: **42 canonical designs** (~15% stratified split).
- **Test Images**: **336 images** (42 designs × 8 colorways, split evenly into 168 Gallery + 168 Query).

---

## 8. Model Architecture
- **Backbone**: ResNet-18 feature extractor pretrained on ImageNet.
- **Pooling**: Generalized Mean (GeM) Pooling with learnable exponent $p = 3.0$ for structural pattern saliency.
- **Projection Head**: MLP (Linear 512 → 512, BatchNorm1d, ReLU, Linear 512 → 256).
- **Output Normalization**: $L_2$ normalization layer producing unit-sphere representations.

---

## 9. Embedding Dimension
- **Vector Dimension**: **256 float32 dimensions** ($1,024$ bytes per embedding).
- **Properties**: Inner product directly computes Cosine Similarity $\in [-1.0, 1.0]$.

---

## 10. Training Loss
- **Loss Function**: **Supervised Contrastive Loss (SupCon)** with temperature $\tau = 0.07$.
- **Objective Formulation**:
  $$\mathcal{L}_{supcon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$
- **Rationale**: SupCon pulls all colorways of the identical saree motif together into a tight cluster while actively repelling motifs from different sarees, even when they share identical color palettes.

---

## 11. Color-Aware Augmentation Pipeline
Configured in `config/training.yaml`:
- **ColorJitter**: Brightness ($\pm 0.8$), Contrast ($\pm 0.8$), Saturation ($\pm 0.8$), Hue ($\pm 0.2$).
- **Random Grayscale**: $p = 0.30$ to encourage the network to focus on weave structure and texture.
- **Structure-Preserving Palette Recoloring**: $p = 0.50$.
- **Geometric Invariance**: Random horizontal flips ($p = 0.50$) and mild rotations ($\pm 15^\circ$).

---

## 12. Identification (Retrieval) Methodology
- Query images from unseen test designs are transformed to 256-dim unit embeddings.
- Full cosine similarity matrix computed against active gallery embeddings.
- Ranks sorted in descending order of similarity.
- Top-1, Top-5, Top-10 evaluated using exact canonical design matching.
- mAP calculated across the entire rank spectrum.

---

## 13. Verification Methodology
- **Pair Construction**: Pairs sampled from unseen test designs:
  - *Positive Pairs*: Same design, different colorways (e.g. `palette_00` vs `palette_05`).
  - *Negative Pairs (Hard Color Traps)*: Different designs, identical colorways (e.g. `banarasi_0008/palette_00` vs `chanderi_0008/palette_00`).
  - *Negative Pairs (General)*: Different designs, different colorways.
- Metric decision: $s(z_A, z_B) \ge \theta_{calibrated} \implies \text{Same Design}$.
- Swept across thresholds $[-1.0, 1.0]$ to determine ROC-AUC, EER, and optimal F1 threshold.

---

## 14. Color-Invariance Benchmark Methodology
Controlled stress testing of the model against artificial and procedural color perturbations:
- **Baseline Cross-Palette**: Query colorways (04..07) retrieving Gallery colorways (00..03).
- **Perturbations Applied**:
  - Positive Hue Shift ($+0.30$)
  - Negative Hue Shift ($-0.30$)
  - Severe Desaturation ($0.3\times$)
  - Oversaturation ($1.8\times$)
  - Severe Darkening ($0.6\times$)
  - Brightening ($1.4\times$)
  - Low Contrast ($0.6\times$)
  - High Contrast ($1.5\times$)
  - Color Jitter Combined
  - Complete Grayscale Conversion

---

## 15. Summary of Empirical Results

### A. Identification (Test Split Retrieval)
| Metric | Empirical Score | Interpretation |
|---|---|---|
| **Recall@1** | **97.02%** | Top-1 retrieved saree is the correct motif |
| **Recall@5** | **100.00%** | Target design always present in top 5 |
| **Recall@10** | **100.00%** | Target design always present in top 10 |
| **mAP** | **0.8333** | High rank-weighted precision |
| **Cross-Palette Recall@1** | **99.40%** | Unseen color query matches gallery color reference |

### B. Verification (Pairwise Decision)
| Metric | Empirical Score | Operating Point |
|---|---|---|
| **ROC-AUC** | **0.9799** | Excellent discrimination |
| **Equal Error Rate (EER)** | **4.89%** | Balanced False Accept / False Reject |
| **EER Threshold** | **0.9200** | Cosine similarity cutoff |
| **Best F1 Score** | **96.09%** | Optimal harmonic precision-recall |
| **Best F1 Threshold** | **0.9000** | Production verification threshold |
| **Verification Pairs Tested** | **2,352 pairs** | 1,176 positive / 1,176 negative |

### C. Color Trap Benchmark
| Metric | Empirical Score | Details |
|---|---|---|
| **Color Trap Accuracy** | **94.65%** | Overall accuracy on hard anti-color benchmark |
| **Same-Design Different-Color Recall** | **94.90%** | Retains match despite completely different palette |
| **Different-Design Same-Color Rejection** | **94.40%** | Rejects same-color distractor motifs |

### D. Controlled Perturbation Breakdown
| Perturbation Type | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---|---|---|---|
| Original Cross-Palette | 99.4% | 100.0% | 100.0% | 0.8999 |
| Positive Hue Shift (+0.3) | 98.8% | 100.0% | 100.0% | 0.9032 |
| Negative Hue Shift (-0.3) | 96.4% | 99.4% | 100.0% | 0.8854 |
| Desaturated (0.3x) | 99.4% | 100.0% | 100.0% | 0.9093 |
| Oversaturated (1.8x) | 99.4% | 100.0% | 100.0% | 0.8856 |
| Darkened (0.6x) | 97.0% | 100.0% | 100.0% | 0.8660 |
| Brightened (1.4x) | 97.6% | 100.0% | 100.0% | 0.8744 |
| Low Contrast (0.6x) | 98.2% | 100.0% | 100.0% | 0.8882 |
| High Contrast (1.5x) | 100.0% | 100.0% | 100.0% | 0.8923 |
| Color Jitter (Combined) | 88.1% | 98.8% | 99.4% | 0.8209 |
| Grayscale (Color-Agnostic) | 100.0% | 100.0% | 100.0% | 0.8939 |
| **Mean Perturbed Average** | **97.5%** | **99.8%** | **99.9%** | **0.8819** |

---

## 16. Hardware Efficiency
- **Parameter Count**: 11,571,521 weights (11.57 M parameters).
- **Embedding Vector**: 256 float32 dimensions (1,024 bytes per saree).
- **Model Checkpoint Size**: 44.22 MB.
- **CPU Inference Latency**: Mean 14.96 ms, Median 15.18 ms, P95 16.68 ms (~67 images/sec).
- **Computational Complexity**: ~0.59 GMACs per forward pass.

---

## 17. Limitations & Real-World Considerations
1. **Synthetic Nature of Handloom Sarees**: The dataset patterns are procedurally generated flat textiles; real-world photos feature fabric folds, ambient lighting shadows, and mannequin draping.
2. **Extreme Color Jitter Boundary**: While moderate hue and lighting variations sustain >96% Recall@1, extreme compound distortions (combined heavy jitter) reduce Recall@1 to 88.1% (though Recall@5 remains 98.8%).
3. **Tradition Class Imbalance in Raw Folders**: Kanjivaram and Patola had fewer unique underlying designs (16 each) than Banarasi or Chanderi (60 each). Canonical deduplication solved label leakage, but future data collection should expand real unique weave variations in Kanjivaram and Patola.
