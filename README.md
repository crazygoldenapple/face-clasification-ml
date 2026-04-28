# Image Classification: Anime / Cartoon / Human

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

A 3-class image classifier that distinguishes **Anime**, **Cartoon**, and **Human** images using hand-crafted feature descriptors and a neural network implemented entirely from scratch with NumPy — no PyTorch, no Keras, no deep learning frameworks.

**Author:** Yosef Pasha

---

## Overview

The goal is to classify images into one of three categories using traditional machine learning techniques:

1. Extract complementary hand-crafted features (HOG, color histograms, downscaled pixels)
2. Feed concatenated features into a fully-connected ANN built from first principles
3. Optimize via systematic grid search with 5-fold cross-validation

All gradient computations, backpropagation, and mini-batch SGD are implemented manually in NumPy, making the learning mechanics fully transparent.

---

## Notebook Structure

| Part | Title | Description |
|------|-------|-------------|
| Setup | Imports & Configuration | All hyperparameters and constants defined in one place |
| Part 1 | Data Loading | Load, resize, normalize images; 80/20 stratified split with visualizations |
| Part 2 | Feature Engineering | Extract and visualize HOG, color histogram, and pixel features |
| Part 3 | ANN Sanity Check | Verify NumPy ANN correctness on dummy data before real training |
| Part 4 | Baseline Training | Concatenate all features, apply StandardScaler, train baseline ANN |
| Part 5 | Test-Set Evaluation | Predictions, confidence bars, confusion matrix, macro F1 score |
| Part 6 | Bonus Experiments | 5-fold CV grid search: feature combinations + hyperparameter tuning |

---

## Dataset

```
Data/
├── Anime/
├── Cartoon/
└── Human/
```

- Images resized to **64×64 pixels**, converted to RGB
- Pixel values normalized to **[0, 1]**
- **80% training / 20% test** split with stratification to preserve class proportions
- Random seed fixed at `42` for full reproducibility

The notebook auto-detects the data path and supports local execution as well as Google Colab (with and without Drive).

---

## Feature Extraction

Three complementary descriptors are extracted and concatenated:

| Feature | Description | Dimensions |
|---------|-------------|-----------|
| **HOG** | Histogram of Oriented Gradients — captures edge and shape information | ~1,568–2,352 |
| **Color Histogram** | Per-channel RGB histogram — encodes color distribution | 96–192 |
| **Pixel Features** | Downscaled 32×32 image flattened — retains spatial layout | 3,072 |

> StandardScaler is always fit on training data only (inside each CV fold) to prevent data leakage.

---

## Model Architecture

A fully-connected feedforward ANN implemented in pure NumPy:

| Component | Choice | Reason |
|-----------|--------|--------|
| Hidden activations | ReLU | Avoids vanishing gradients, trains faster |
| Output activation | Softmax | Converts logits to class probabilities |
| Loss | Categorical cross-entropy | Standard multi-class objective |
| Optimizer | Mini-batch SGD | Simple and effective |
| Weight init | He initialization | Optimal variance scaling for ReLU |
| Gradient | ŷ − y_true at output | Simplified by softmax + cross-entropy pairing |

---

## Experiments

### Baseline (Part 4)

| Hyperparameter | Value |
|----------------|-------|
| Hidden layers | `[256, 128]` |
| Learning rate | `0.01` |
| Batch size | `32` |
| Epochs | `50` |

### Grid Search (Part 6)

All 36 combinations of the following are evaluated with 5-fold stratified CV:

| Parameter | Values |
|-----------|--------|
| Hidden layers | `[128]`, `[256,128]`, `[512,256]`, `[256,128,64]` |
| Learning rate | `0.001`, `0.01`, `0.1` |
| Batch size | `16`, `32`, `64` |

9 feature combination configurations are also compared in Part 6b.

---

## Evaluation

**Primary metric:** Macro-average F1 score (equal weight per class, robust to class imbalance)

**Additional outputs:**
- Per-class precision, recall, and F1
- Confusion matrix heatmap (seaborn)
- Confidence bar charts for individual predictions
- Training loss curves (baseline and final)
- Top-10 feature and hyperparameter configurations with mean ± std bars

---

## Setup & Usage

### Install dependencies

```bash
pip install numpy pandas scikit-learn scikit-image Pillow matplotlib seaborn tqdm
```

### Run

1. Place your image data in the following structure (relative to the notebook):
   ```
   Data/
   ├── Anime/
   ├── Cartoon/
   └── Human/
   ```
2. Open `ml_final_project.ipynb` in Jupyter or Google Colab
3. Run all cells in order from top to bottom

> The notebook is self-contained — all configuration (image size, split ratio, hyperparameters) is defined in the **Setup** cell at the top.

---

## Results

The notebook reports:

- **Baseline F1** — all three features concatenated with default hyperparameters
- **Best feature combination** — identified via 5-fold CV over 9 configurations
- **Best hyperparameters** — top configuration from 36-combo grid search
- **Final F1** — model trained with optimal settings; improvement over baseline is computed and displayed

A side-by-side bar chart comparing baseline vs. final F1 is generated at the end of the notebook.
