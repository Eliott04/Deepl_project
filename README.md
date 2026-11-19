# Efficient Vision Transformers — MatchViT & MobileViT

## Overview

Experimental computer vision project exploring **efficient attention mechanisms for image classification**. The repository implements **MatchViT**, combining convolutional layers with **MaSSA (Matching and Sparse Sampling Attention)**, and compares it with **MobileViT**. It also includes a standalone **Vision Transformer (ViT)** implementation with standard multi-head self-attention (MHSA).

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
