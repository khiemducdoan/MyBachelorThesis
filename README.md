# 🧠 MyBachelorThesis: AI Pipeline for TBI Analysis

> An advanced machine learning pipeline for analyzing Traumatic Brain Injury (TBI) data using multi-modal approaches.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-compatible-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/khiemducdoan/MyBachelorThesis)

## 🎯 Introduction

This project implements a sophisticated AI pipeline for TBI analysis, leveraging state-of-the-art machine learning architectures including NAIM and TabNet.

### ✨ Key Features

- 📊 Flexible data processing pipeline for TBI datasets
- 🔄 Multi-modal learning support (clinical + text data)
- ⚙️ Hydra-based configurable model architectures
- 📈 Integrated logging with TensorBoard and W&B
- 📊 Comprehensive evaluation metrics and visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/khiemducdoan/MyBachelorThesis.git
cd MyBachelorThesis
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage Guide

### 1️⃣ Data Preparation

| Step | Description |
|------|-------------|
| Raw Data | Place TBI data in `dataset/raw/` |
| Text Data | Ensure CSV format with required columns |
| Preprocessing | Use notebooks in `dataset/` for cleaning |

### 2️⃣ Configuration

Configurations are managed through Hydra in the `configs/` directory:

```
configs/
├── main.yaml          # Main configuration
├── model/            # Model-specific configs
├── data/             # Dataset configs
└── default/          # Default parameters
```

**Modifying Settings:**
```bash
# Via command line
python train.py model=naim_text data.batch_size=32
```

### 3️⃣ Training

```bash
# Basic training
python train.py

# With specific config
python train.py experiment=tbi_naim

# Hyperparameter sweep
python train.py logging.sweep=true
```

### 4️⃣ Monitoring

| Tool | Location | Purpose |
|------|----------|---------|
| TensorBoard | `outputs/logs/` | Training progress |
| Checkpoints | `outputs/<date>/` | Model saves |
| W&B | Online dashboard | Experiment tracking |

### 5️⃣ Project Structure

```
MyBachelorThesis/
├── 📁 configs/        # Configuration files
├── 📁 dataset/       # Data and preprocessing
├── 📁 src/          # Source code
│   ├── models/     # Model implementations
│   └── utils/      # Utility functions
├── 📁 outputs/     # Training outputs
└── 📁 notebooks/   # Analysis notebooks
```

## 📫 Contact

For questions about the dataset or project, please email: vinakhiem120@gmail.com

---
<div align="center">
Made with ❤️ for Traumatic Brain Injury Research
</div>
