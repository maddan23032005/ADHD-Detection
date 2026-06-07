# ADHD EEG Classification using Deep Learning

A deep learning project for classifying ADHD vs. Control subjects using multi-channel EEG signals. The model combines a shared CNN encoder with cross-channel attention mechanisms and a Transformer encoder to capture both per-channel spectral features and inter-channel relationships.

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `ADHD.ipynb` | Initial/exploratory notebook — data inspection, EEG loading, segmentation, and a basic CNN model using TensorFlow/Keras |
| `NovileADHD.ipynb` | Intermediate notebook — improved pipeline with PyTorch |
| `NovileADHD2.ipynb` | Further iteration with PyTorch |
| `NovileADHD(UPDATED).ipynb` | **Final notebook** — full production pipeline with K-Fold cross-validation, ensemble inference, ROC/PR curve analysis, and attention visualization |

---

## 🧠 Problem Statement

Attention Deficit Hyperactivity Disorder (ADHD) is a neurodevelopmental disorder that affects millions worldwide. This project trains a classifier to distinguish ADHD patients from healthy controls using **resting-state EEG recordings** stored in MATLAB `.mat` format.

---

## 📊 Dataset

- **Format:** MATLAB `.mat` files, one file per subject
- **Classes:**
  - `ADHD/` — 61 files (label = 1)
  - `Control/` — 60 files (label = 0)
- **EEG:** 19-channel recordings at 128 Hz sampling rate
- **Data source:** Expected at `/kaggle/input/adhd-dataset/data/` (Kaggle dataset)

---

## 🏗️ Model Architecture

### `CrossChannelAttentionModel`

```
Input: (batch, 19 channels, 64 mel-bins, time_frames)
         │
         ▼
SharedCNNEncoder (per-channel, weights shared)
  Conv2D(1→32) → BN → LeakyReLU → MaxPool
  Conv2D(32→64) → BN → LeakyReLU → MaxPool
  Conv2D(64→128) → BN → LeakyReLU → AdaptiveAvgPool
  Flatten → Linear(128→256) → LayerNorm → LeakyReLU
         │
         ▼
Learnable per-channel scalar weights (optional)
         │
         ▼
Learnable positional embeddings (per channel)
         │
         ▼
MultiheadAttention (4 heads, embed_dim=256)
         │
         ▼
TransformerEncoder (2 layers)
         │
         ▼
Mean pooling across channels
         │
         ▼
Classifier: LayerNorm → Linear(256→128) → LeakyReLU → Dropout → Linear(128→2)
         │
         ▼
Output: class logits (ADHD vs. Control)
```

---

## 🔄 Data Processing Pipeline

1. **Load `.mat` files** — Auto-detects the EEG array variable, normalizes orientation to `(channels, samples)`
2. **Segmentation** — Splits signals into overlapping windows:
   - Segment length: 512 samples (~4 seconds at 128 Hz)
   - Overlap: 50%
3. **Z-score normalization** — Per-channel, per-segment
4. **Log-Mel Spectrogram** — Converts each channel's segment to a 2D representation:
   - FFT window: 256, Hop length: 128, Mel bins: 64
   - Output shape: `(19, 64, time_frames)`
5. **Augmentation** (training only):
   - Gaussian noise injection
   - Random temporal shift (±5%)

---

## 🔬 Training Strategy

### Stratified K-Fold Cross-Validation

- **K = 5** folds, stratified by label
- Each fold trains an independent model
- Best model checkpoint saved per fold (based on validation AUC)
- Early stopping with patience = 8 epochs

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Epochs | 40 |
| Embed dim | 256 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Dropout | 0.3 |

### Learning Rate Schedule
- Cosine Annealing (`T_max = 40`)

### Mixed Precision Training
- `torch.cuda.amp` (AMP) for faster training on GPU

---

## 📈 Results

### K-Fold Validation Performance

| Fold | AUC | ACC | F1 |
|------|-----|-----|----|
| 1 | 0.6728 | 0.619 | 0.653 |
| 2 | 0.9164 | 0.821 | 0.834 |
| 3 | 0.8557 | 0.764 | 0.787 |
| 4 | 0.7589 | 0.681 | 0.745 |
| 5 | 0.9266 | 0.850 | 0.857 |
| **Mean ± Std** | **0.8261 ± 0.1086** | **0.747 ± 0.096** | **0.775 ± 0.081** |

### Ensemble Test Performance (aggregated across all 5 folds)

| AUC | ACC | F1 |
|-----|-----|----|
| **0.9605** | **0.863** | **0.877** |

---

## 🧪 Inference

### Ensemble prediction for a single file

```python
avg_prob, seg_probs = predict_file_with_ensemble(
    "/path/to/eeg/file.mat",
    ckpt_paths  # list of fold checkpoint paths
)
print("ADHD probability:", avg_prob)
# avg_prob >= 0.5 → ADHD, < 0.5 → Control
```

The ensemble:
1. Loads each fold's best checkpoint
2. Segments the EEG file
3. Computes softmax probabilities per segment
4. Averages across segments and folds

---

## 📉 Visualizations

The final notebook generates:
- **Attention heatmaps** — channel × channel attention weights
- **Per-fold ROC curves**
- **Aggregated ROC curve**
- **Precision-Recall curve**

---

## ⚙️ Configuration (`CFG`)

Key configuration flags in the final notebook:

```python
CFG.USE_CHANNEL_WEIGHTS = True   # Learnable per-channel importance scalars
CFG.INIT_PRIOR = True            # Initialize with domain-knowledge priors (e.g., Fp1, Fp2 frontal channels)
CFG.AGGREGATE_FOLDS = True       # Average predictions from all fold models at inference
CFG.DEBUG = False                # Set True to run on a small subset for quick testing
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision librosa scipy scikit-learn matplotlib pandas
```

### On Kaggle (recommended)

1. Upload the ADHD dataset to Kaggle with path: `adhd-dataset/data/ADHD` and `adhd-dataset/data/Control`
2. Open `NovileADHD(UPDATED).ipynb` in a Kaggle notebook with **GPU enabled**
3. Run all cells sequentially

### Locally

1. Update `CFG.DATA_PATH`, `CFG.ADHD_DIR`, `CFG.CONTROL_DIR` to local paths
2. Ensure GPU is available (`torch.cuda.is_available()`)
3. Run the notebook

---

## 📌 Notes

- The model was developed and tested with Python 3.11 and PyTorch on CUDA (NVIDIA Tesla T4)
- Fold 1 underperforms significantly — this is likely due to data imbalance or a particularly difficult validation split. The ensemble mitigates this issue
- The `CHANNEL_PRIOR` config allows injecting neuroscience knowledge (e.g., frontal channels like Fp1, Fp2 are known to be relevant in ADHD) — requires providing a `name_to_idx` mapping

---

## 👤 Author

**Maddan Murugan**
