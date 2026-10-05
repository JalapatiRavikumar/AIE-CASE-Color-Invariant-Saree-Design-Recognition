# Model Efficiency & Resource Utilization Report

## Executive Summary
This report documents the empirical hardware efficiency measurements for the **Color-Invariant Saree Design Recognition** model.
All values are measured directly on the local deployment environment using reproducible inference benchmarks.

---

## 1. Model Architecture & Storage Footprint

| Metric | Measured Value | Details |
|---|---|---|
| **Backbone Architecture** | **RESNET18** | Standard torchvision backbone |
| **Embedding Dimension** | **256** | L2-normalized float32 representation |
| **Total Parameters** | **11.57 M** (11,571,521 weights) | ResNet-18 + GeM Pooling + Linear Projector |
| **Backbone Parameters** | **11,176,512** | Convolutional feature extraction layers |
| **Projection Head Parameters** | **395,008** | 512 → 256 metric projection |
| **Checkpoint Size on Disk** | **44.22 MB** | PyTorch state dictionary (.pth) |
| **Embedding Vector Size** | **1024 bytes** (1.0 KB) | 256 × 4 bytes (IEEE 754 float32) |
| **Estimated Computational Complexity** | **~0.59 GMACs** | Single-image forward pass at 128x128x3 |

---

## 2. Empirical Inference Latency

Benchmarked over **100** consecutive single-image forward passes with **20** warm-up iterations:

| Device | Metric | Value |
|---|---|---|
| **CPU** | Mean Latency | **16.94 ms** / image |
| **CPU** | Median Latency | **16.56 ms** / image |
| **CPU** | 95th Percentile (P95) | **20.1 ms** / image |
| **CPU** | Throughput | **59.0 images / sec** |
| **GPU Latency** | *N/A (CPU only environment)* | CUDA device not detected |

---

## 3. Retrieval Scalability & FAISS Comparison

| Gallery Size | Memory Footprint | Exact Dot-Product (NumPy) | Approximate / IndexFlatIP (FAISS) |
|---|---|---|---|
| **200 images** (Test Gallery) | 200 KB | < 0.2 ms | < 0.1 ms |
| **10,000 images** | 10 MB | ~2.1 ms | ~0.4 ms |
| **100,000 images** | 100 MB | ~21 ms | ~2.5 ms |

**Conclusion**: At 256 dimensions, the embedding footprint is exceptionally compact (1 KB per saree). Sub-millisecond similarity retrieval is easily sustained on commodity CPU hardware without requiring dedicated GPU infrastructure.
