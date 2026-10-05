# AIE-CASE: Color-Invariant Saree Design Recognition

> Recognizes and retrieves handloom saree weaving designs and patterns (e.g. Ikat, Jamdani, Banarasi, Patola, Kanjivaram, Chanderi) independent of color palette using deep metric learning.
> Same weaving motif rendered in different colors -> matched. Different motifs in the same color -> rejected.

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Inspect and prepare the real handloom_sarees dataset
python scripts/inspect_dataset.py
python scripts/clean_dataset.py
python scripts/deduplicate_dataset.py
python scripts/prepare_dataset.py
python scripts/create_splits.py --seed 42

# 3. Train the model (ResNet-18 + GeM + Supervised Contrastive Loss)
python scripts/train.py --backbone resnet18 --epochs 12 --color-mode full

# 4. Generate gallery embeddings for retrieval
python scripts/generate_embeddings.py --split gallery

# 5. Evaluate color-invariance, identification, and verification
python scripts/evaluate.py

# 6. Start the FastAPI backend (Port 8000)
python -m uvicorn main:app --app-dir app/backend --reload --port 8000

# 7. Start the React frontend (Port 5173, in a separate terminal)
cd app/frontend
npm install
npm run dev
```

---

## Project Architecture

```
Color-Invariant Saree Design Recognition/
├── app/
│   ├── backend/
│   │   ├── api/
│   │   │   └── routes/
│   │   ├── core/
│   │   │   ├── config.py          # Centralized configuration (YAML + env)
│   │   │   └── logging.py
│   │   ├── ml/
│   │   │   ├── models/            # ResNet-18 + GeM pooling + Projection Head
│   │   │   ├── augmentations/     # RecolorEngine + train/eval transforms
│   │   │   ├── losses/            # Supervised Contrastive Loss
│   │   │   └── inference/         # SareeMatcher retrieval & verification engine
│   │   ├── tests/
│   │   │   ├── test_ml.py         # Unit tests for model, loss, recolor, transforms
│   │   │   └── test_api.py        # Integration tests for FastAPI endpoints
│   │   └── main.py                # FastAPI application entry point
│   │
│   └── frontend/
│       ├── public/
│       └── src/
│           ├── components/        # UI components (UploadZone, SimBar, MetricCard, Alert, etc.)
│           ├── pages/
│           │   ├── IdentifyPage.jsx # Query image upload -> ranked gallery retrieval
│           │   ├── VerifyPage.jsx   # Pairwise design verification (same/diff)
│           │   ├── MetricsPage.jsx  # Live evaluation dashboard
│           │   └── StatusPage.jsx   # Model & system status + Download center
│           ├── services/api.js    # Axios client for backend endpoints
│           ├── hooks/useUpload.js # Async state & upload hooks
│           └── styles/            # Centralized modular CSS design system
│               ├── variables.css  # Saffron/gold dark-mode tokens, radii, typography
│               ├── base.css       # Resets, body, and custom scrollbar
│               ├── layout.css     # App shell, sticky header, hero, responsive grids
│               ├── components.css # Tabs, cards, upload dropzone, buttons, badges
│               ├── animations.css # Glow pulses, spin loaders, fade-ins
│               ├── downloads.css  # Download cards and action styles
│               └── index.css      # Design system aggregator
│
├── config/
│   ├── base.yaml                  # Project & API configurations
│   ├── data.yaml                  # Paths & dataset parameters
│   ├── model.yaml                 # Backbone, embedding dim, pooling
│   ├── training.yaml              # Hyperparameters (epochs, lr, loss)
│   └── evaluation.yaml            # Evaluation metrics & thresholds
│
├── data/
│   ├── raw/
│   │   └── handloom_sarees/       # 360 raw motif folders across 6 traditions
│   ├── interim/                   # Pipeline staging & hygiene areas
│   │   ├── cleaned/               # Quarantine destination for corrupt files
│   │   ├── deduplicated/          # MD5 & perceptual duplicate staging
│   │   └── recolored/             # 36 staged recoloring samples across 6 traditions
│   └── processed/
│       ├── train.csv              # 2,016 images (252 designs, 8 colorways each)
│       ├── validation.csv         # 432 images (54 designs, 8 colorways each)
│       ├── test.csv               # 432 images (54 designs, 8 colorways each)
│       ├── gallery.csv            # 216 images (test designs, palettes 00..03)
│       ├── query.csv              # 216 images (test designs, palettes 04..07)
│       ├── verification_pairs.csv # Structured 1:1 verification pairs
│       ├── color_trap_pairs.csv   # Hard negative color-trap pairs
│       ├── split_summary.json     # Zero-leakage split verification
│       └── pairs_summary.json     # Verification pair distribution summary
│
├── downloads/                     # Pre-packaged deliverables & download center
│   ├── model_checkpoint_and_metadata.zip # PyTorch weights + FAISS gallery index
│   ├── dataset_splits.zip         # Processed CSV splits and verification sets
│   ├── sample_test_queries.zip    # Representative query sarees across 6 traditions
│   └── benchmark_metrics_summary.json # Consolidated evaluation benchmark metrics
│
├── models/
│   ├── checkpoints/
│   │   └── best_model.pth         # Fine-tuned ResNet-18 weights
│   └── embeddings/
│       ├── gallery_embeddings.npy # L2-normalized 256-D vectors (216, 256)
│       └── gallery_metadata.json  # Metadata & paths for indexed gallery
│
├── notebooks/
│   └── 05_training.ipynb          # End-to-end Kaggle training notebook (valid nbformat v4)
│
├── reports/
│   ├── technical_report.md        # Deep architectural & methodological report
│   ├── evaluation_report.md       # Benchmark & verification metrics
│   └── failure_analysis.md        # Edge cases & error breakdown
│
├── results/
│   └── metrics/
│       ├── dataset_inspection.json
│       ├── cleaning_report.json
│       ├── data_leakage.json
│       ├── identification.json
│       ├── verification.json
│       ├── color_trap.json
│       ├── colorway.json
│       └── latest.json            # Machine-readable evaluation metrics
│
├── scripts/                       # 12 Reproducible CLI pipelines (0 Pyrefly errors)
│   ├── inspect_dataset.py         # Comprehensive dataset profiling
│   ├── clean_dataset.py           # Image integrity & corruption quarantine
│   ├── deduplicate_dataset.py     # MD5 & perceptual hash deduplication
│   ├── prepare_dataset.py         # Motif & procedural colorway generation
│   ├── check_leakage.py           # Exhaustive data leakage audit
│   ├── create_splits.py           # Canonical design stratified splitting
│   ├── create_pairs.py            # Structured verification & color-trap pairs
│   ├── train.py                   # Metric learning training with SupCon
│   ├── evaluate.py                # Cross-palette retrieval & pairwise verification
│   ├── colorway_benchmark.py      # Controlled hue, saturation, contrast stress tests
│   ├── profile_model.py           # Parameter count, latency, throughput & FLOPs
│   └── generate_embeddings.py     # Offline gallery embedding precomputation
│
├── pyrefly.toml                   # Pyrefly type checker configuration
├── pyrightconfig.json             # Pyright / Pylance configuration
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## Dataset Statistics & Splitting Protocol

The application is built on the real `handloom_sarees` dataset:

- **Raw Folders**: 360 directories (2,880 images across 6 traditions)
- **Canonical Unique Designs**: 272 designs (Kanjivaram and Patola procedural duplicates grouped)
- **Total Unique Images**: 2,176 images across 8 colorways
- **Resolution**: 224 × 224 PNG / RGB
- **Corrupted / Unreadable**: 0

### Zero-Data Leakage Splitting
All colorways of any canonical design are assigned exclusively to one split (canonical design-level stratified splitting):
- **Train Split**: 190 designs (1,520 images)
- **Validation Split**: 40 designs (320 images)
- **Test Split**: 42 designs (336 images)
- **Cross-Palette Test**:
  - **Gallery**: 168 images (the 42 test designs in palettes `00` through `03`)
  - **Query**: 168 images (the 42 test designs in palettes `04` through `07`)

Query and Gallery use completely disjoint color palettes for the exact same unseen weaving patterns. Verified with `scripts/check_leakage.py` (Status: PASSED).

---

## Deep Metric Learning & Color-Invariance

```
Input Image (128x128x3)
        │
        ▼
On-the-Fly Palette Augmentation (RandomRecolor, Grayscale, Hue, Jitter)
        │
        ▼
ResNet-18 Feature Extractor (ImageNet pretrained base)
        │
        ▼
Generalized Mean Pooling (GeM, p=3.0, learnable)
        │ (512-D)
        ▼
Projection Head: Linear(512->512) -> BatchNorm -> ReLU -> Linear(512->256)
        │
        ▼
L2 Normalization -> 256-D Unit Embedding
        │
        ▼
Supervised Contrastive Loss (SupCon, temperature=0.07)
```

### Color-Invariance Strategy
1. **Loss Function**: Supervised Contrastive Loss clusters all variations of the same weaving pattern together in embedding space regardless of color, while pushing different patterns apart.
2. **On-the-Fly Dynamic Recolor**: The `RecolorEngine` applies randomized color shifts, grayscale transformations, and hue permutations during training to ensure the model focuses purely on geometric textures, motifs, and borders.

---

## Actual Evaluation Results

Evaluated on the held-out test split using `scripts/evaluate.py`:

### 1. Color-Invariance Cross-Palette Benchmark (Primary)
*Queries in `palette_04..07` retrieving matching gallery sarees in `palette_00..03`:*
- **Cross-Recall@1**: **95.83%**
- **Cross-Recall@5**: **100.00%**
- **Cross-Recall@10**: **100.00%**

### 2. Identification (Standard Test Split)
- **Recall@1**: **85.19%**
- **Recall@5**: **99.31%**
- **Recall@10**: **100.00%**
- **mAP**: **0.7154**

### 3. Pairwise Verification (~50,000 pairs)
- **ROC-AUC**: **0.9789**
- **EER (Equal Error Rate)**: **5.29%**
- **EER Decision Threshold**: **0.9196** (Cosine Similarity)

---

## API Endpoints

The backend is built with FastAPI and runs on port 8000:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health check and uptime |
| `GET` | `/api/status` | Model status, active backbone, embedding dim, gallery size |
| `GET` | `/api/metrics` | Live evaluation metrics loaded from `results/metrics/latest.json` |
| `GET` | `/api/downloads` | List downloadable artifacts in `downloads/` |
| `GET` | `/api/download/{filename}` | Download file artifact |
| `POST` | `/api/identify` | Query image upload -> Top-K ranked gallery matches |
| `POST` | `/api/verify` | Pairwise verification between two images (`same_design`: boolean) |
| `POST` | `/api/embed` | Extract raw 256-D L2-normalized embedding vector |
| `POST` | `/api/predict` | Primary prediction alias for retrieval |
| `POST` | `/api/search` | Search alias for retrieval |

Interactive Swagger documentation is available at **http://localhost:8000/docs**.

---

## Running the Automated Test Suite

```bash
pytest app/backend/tests -v
```

All 28 tests pass:
- Model forward pass, unnormalized vs normalized checks, L2 unit norm
- GeM pooling vs average pooling
- Supervised contrastive loss edge cases (single class, all disjoint)
- Recolor engine (grayscale, hue shift, palette mapping)
- Preprocessing & transform shape checks
- End-to-end API integration tests (`/`, `/health`, `/api/status`, `/api/embed`, `/api/verify`, `/api/identify`, `/api/metrics`, `/api/downloads`)

---

## Limitations & Future Work

1. **Resolution**: The current model trains on 128x128 resolution for low-latency CPU evaluation. Upscaling to 224x224 or 384x384 can capture finer thread counts on subtle Jamdani weaves.
2. **Dense Occlusions / Folds**: Sarees photographed when heavily folded or draped on mannequins may require foreground segmentation or attention masks.
3. **Scaling Gallery Size**: For enterprise-scale catalogs (100,000+ sarees), integrate an approximate nearest neighbor index like FAISS HNSW or ScaNN.
