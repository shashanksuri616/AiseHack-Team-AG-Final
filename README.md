# AiseHack-Team-AG-Final

# Air Pollution Forecasting (ConvLSTM + U-Net)

This project implements a deep learning pipeline for **spatio-temporal air pollution forecasting** using a ConvLSTM-based architecture combined with a U-Net backbone. It is designed for competition-style datasets with emphasis on **performance, scalability, and reproducibility**.

---

## Overview

* Hybrid **ConvLSTM + U-Net** architecture
* Captures both:

  * Temporal dependencies (time-series pollution trends)
  * Spatial correlations (grid-based environmental data)
* Optimized for:

  * Large datasets
  * GPU training
  * Fast inference

---

## Project Structure

```
.
├── TeamAG.ipynb        # Main training & inference notebook
├── best.pt             # Trained model weights (best checkpoint)
├── LICENSE             
├── README.md          
```

---

## Pretrained Model

The repository includes a trained model:

* **File**: `best.pt`
* **Description**: Best-performing checkpoint from training
* **Usage**: Can be directly loaded for inference or fine-tuning

### Load Model Example

```python
import torch

model = torch.load("best.pt", map_location="cpu")
model.eval()
```

> Ensure model architecture matches before loading (if modified)

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas torch torchvision tqdm
```

> Recommended: CUDA-enabled GPU for training

---

## How to Run

### 1. Set Dataset Path

```python
COMP_ROOT = "path_to_dataset"
RAW_PATH = os.path.join(COMP_ROOT, "raw")
```

---

### 2. Run Notebook

#### Option A: Local (Jupyter)

```bash
jupyter notebook
```

* Open `TeamAG.ipynb`
* Run all cells

---

#### Option B: Kaggle (Recommended)

1. Go to Kaggle → Code → New Notebook
2. Upload `TeamAG.ipynb`
3. Add dataset (right panel)
4. Set path:

```python
COMP_ROOT = "/kaggle/input/your-dataset-name"
```

5. Enable GPU:

   * Settings → Accelerator → GPU
6. Click **Run All**

---

## Training Pipeline

The notebook performs:

* Dataset preprocessing
* Z-score normalization
* DataLoader creation
* Model training with scheduler
* Validation tracking

---

## Inference Pipeline

* Loads trained model (`best.pt`)
* Generates predictions on test data
* Applies **weighted top-k ensembling**

---

## Model Architecture

* **Backbone**: U-Net
* **Temporal Layer**: ConvLSTM
* **Objective**: Predict pollution levels across spatial grids over time

---

## Key Features

* Memory-efficient pipeline
* Modular design for experimentation
* Custom training loop
* Ensemble-based predictions
* Competition-ready workflow

---

## Reproducibility Tips

* Fix random seeds for consistency
* Use same normalization stats
* Keep dataset structure identical
* Match PyTorch version

---

## Notes

* Designed for **Kaggle-style competitions**
* Paths must be adapted for local environments
* Large model file (`best.pt`) is managed via Git LFS

---

## License

This project uses an ANRF license. See `LICENSE` for details.

---

---
