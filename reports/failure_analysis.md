# Failure Analysis — Color-Invariant Saree Design Recognition

## Common Failure Modes

### 1. Intra-Tradition Confusion
Designs within the same tradition (e.g., two different Ikat patterns) share structural 
motifs that cause false positives. The network sometimes confuses the periodicity of 
Ikat warp-resist chevrons across different designs.

**Mitigation**: Hard-negative mining during SupCon training pulls apart similar-but-different
design embeddings from the same tradition.

### 2. Extreme Colorway Shift (Black & White)
Grayscale or near-grayscale colorways lose texture-discriminating cues that the 
color-augmented model partially relies on despite training with random grayscale.

**Mitigation**: Increase random grayscale probability (p=0.3 → 0.5) in training augmentation.

### 3. Low-Resolution / Blurry Queries
The synthetic dataset is rendered at a fixed 224×224. Real-world photographs at lower 
effective resolution cause embedding drift.

**Mitigation**: Add blur and JPEG compression augmentations during training.

### 4. Partial Saree Views
Queries showing only borders (pallu) vs. body of the same design can have different 
embedding coordinates because the spatial patterns differ.

**Mitigation**: Random crop augmentation at varying scales (already partially addressed
by the training pipeline's RandomResizedCrop).
