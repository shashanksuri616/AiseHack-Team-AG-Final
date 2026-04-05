# AiseHack-Team-AG-Final
# Air Pollution Forecasting (ConvLSTM + U-Net)

This project implements a deep learning pipeline for **spatio-temporal air pollution forecasting** using a ConvLSTM-based model. It is designed for competition-style datasets and focuses on efficient training, validation, and inference.

---

## Overview

* Uses **ConvLSTM + U-Net** to model spatial + temporal dependencies
* Handles large datasets with optimized loading and preprocessing
* Includes:

  * Data normalization (z-score stats)
  * Training & validation pipeline
  * Evaluation metrics
  * Ensemble-based inference

---

## Project Structure

```
.
├── TeamAG.ipynb        
├── LICENSE             
├── README.md           
```

---

## Requirements

Install dependencies before running:

```bash
pip install numpy pandas torch torchvision tqdm
```

> Recommended: Use GPU (CUDA-enabled PyTorch)

---

## How to Run

### 1. Set Dataset Path

Update dataset paths in the notebook:

```python
COMP_ROOT = "path_to_dataset"
RAW_PATH = os.path.join(COMP_ROOT, "raw")
```

---

### 2. Run Notebook
Option A: Jupyter Notebook (Local)

Open the notebook using Jupyter:

jupyter notebook
Open TeamAG.ipynb
Click Run → Run All Cells

Option B: Kaggle Notebook (Recommended)
Go to Kaggle and log in
Click Code → New Notebook
Upload your notebook:
Click File → Upload Notebook
Select TeamAG.ipynb
Attach the dataset:
On the right panel, click Add Data
Search and add the required dataset

Update dataset path in the notebook:

COMP_ROOT = "/kaggle/input/your-dataset-name"
Enable GPU (important):
Go to Settings (right panel)
Set Accelerator → GPU
Run the notebook:
Click Run All (top toolbar)
---

### 3. Training

The notebook will:

* Compute normalization statistics
* Prepare datasets and loaders
* Train the ConvLSTM + U-Net model

---

### 4. Inference

Final section performs:

* Test predictions
* Weighted top-k ensembling

---

##  Model Summary

* **Architecture**: ConvLSTM + U-Net
* Captures:

  * Temporal patterns (time series)
  * Spatial dependencies (grid-based pollution data)

---

## Key Features

* Efficient memory usage
* Custom training loop with scheduler
* Modular structure (easy to tweak)
* Ensemble predictions for improved performance

---

## Notes

* Designed for competition environments (e.g., Kaggle)
* Paths and parameters may need adjustment for local runs
* Ensure dataset structure matches expected format

---

## License

This project uses a ANRF license. See the `LICENSE` file for details.
