# Efficient Vision Transformers — MatchViT & MobileViT

## Overview

This project is an **independent, from-scratch PyTorch reimplementation** of **MatchViT**, based on the research paper *[MatchViT: Light-weight Vision Transformer with Matching Separable Self-attention](https://openreview.net/forum?id=7EUYamNuJ9)* (Jiacheng Liao et al.; submitted to ICLR 2026). The proposed architecture combines convolutional feature extraction with two key mechanisms introduced in the paper: **Matching Separable Self-Attention (MaSSA)** and **Context-Gated Feed-Forward Networks (Context-Gated FFN)**, designed to improve computational efficiency.

**Our additional work** focuses on evaluating the reimplementation rather than proposing a new attention mechanism:
- **Implementing the two proposed mechanisms—MaSSA and Context-Gated FFN—from scratch in PyTorch**, along with the MatchViT architecture.
- **Training and comparing MatchViT with MobileViT on Imagenette**, a 10-class classification dataset.
- Studying **MaSSA versus standard multi-head self-attention (MHSA)** through timing and computational-cost measurements; the code also supports switching attention blocks in MatchViT.
- Tracking accuracy, trainable parameters and multiply–accumulate operations (MACs).

This is an educational/research reproduction and an **additional small-scale benchmark**, not the authors' official implementation or a reproduction of all results reported in the paper.
## Dataset and methods

- **Dataset:** Imagenette, a 10-class subset of ImageNet, using 256 × 256 RGB images.
- **Models:** MatchViT, MobileViT, and a ViT implementation for architecture experiments.
- **Attention:** MaSSA / context-gated feed-forward layers alongside standard MHSA.
- **Evaluation:** image-classification accuracy, trainable parameter counts, and computational cost (MACs). The code also includes attention-timing benchmarks.

The training script runs **MatchViT and MobileViT** for 20 epochs using Adam. No verified numerical results are reported in this README.

## Technologies

**Python · PyTorch · torchvision · einops · NumPy · Matplotlib · THOP**

## Repository

- `training_imagenette.py` — Imagenette loading, training and evaluation.
- `src/models/` — MatchViT, MobileViT and ViT architectures.
- `src/modules/` — attention mechanisms, transformer blocks and feed-forward modules.
- [`DEEPL_POSTER.pdf`](DEEPL_POSTER.pdf) — project poster.

> **Status:** research prototype. Full training and reproducibility have not been independently validated.
