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
