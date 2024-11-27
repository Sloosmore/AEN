# Adaptable Embeddings Network (AEN)

This repository contains the supplementary materials, code implementations, and datasets for the paper "Adaptable Embeddings Network (AEN)" by Stan Loosmore and Alexander J. Titus.

## Overview

AEN is a novel dual-encoder architecture using Kernel Density Estimation (KDE) for efficient text classification in resource-constrained environments. This repository provides all necessary resources to reproduce our results and implement the AEN architecture.

## Repository Structure

```
.
├── code/
│   ├── models/              # Model implementations
│   │   ├── aen.py          # Core AEN architecture
│   │   └── kde.py          # KDE implementation
│   ├── training/           # Training scripts
│   └── evaluation/         # Evaluation scripts
├── data/
│   ├── raw/                # Raw synthetic data
│   ├── processed/          # Processed datasets
│   └── generation/         # Data generation scripts
└── docs/
    └── appendix/           # Supplementary materials
```

## Data Generation Pipeline

The repository includes our complete data generation pipeline, consisting of:
- Statement generation scripts
- Condition generation methods
- Labeling procedures
- Data cleaning and preprocessing tools

See `data/generation/README.md` for detailed instructions on generating synthetic datasets.

## Model Implementation

The core AEN architecture implementation includes:
- Dual-encoder architecture with KDE
- Multiple KDE kernel options (Gaussian, Epanechnikov, Triangular)
- Bandwidth estimation methods (Scott's rule, Silverman's rule)
- Training configurations for various hyperparameters

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.5+
- sentence-transformers
- scipy
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the AEN model:

```bash
python train.py --config configs/default.yaml
```

Key hyperparameters can be modified in the config file:
- Learning rate
- Batch size
- KDE parameters
- Loss weights
- Model architecture

## Evaluation

Run evaluation scripts:

```bash
python evaluate.py --model_path checkpoints/best_model.pt --test_data data/processed/test.csv
```

## Results

The repository includes scripts to reproduce all results reported in the paper:
- Comparison with SLMs (LLaMA 3.2 3B and Phi 3.5 Mini)
- Performance metrics across different hyperparameters
- Computational efficiency analysis

## Citation

If you use this
