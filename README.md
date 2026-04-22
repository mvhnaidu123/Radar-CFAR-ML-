# Radar-CFAR-ML-
# Radar Target Detection: CFAR + Deep Learning

A two-stage radar target detection pipeline combining classical signal processing with deep learning. Stage 1 uses Cell Averaging CFAR (CA-CFAR) for constant false alarm rate detection. Stage 2 uses a CNN to classify CFAR candidates as true targets or clutter.

## Problem

Radar systems must detect targets (vehicles, obstacles, aircraft) against a background of clutter (ground reflections, weather, multipath). Classical CFAR detectors maintain a fixed false alarm rate but struggle with non-homogeneous clutter. This project combines CFAR's interpretable threshold logic with a CNN's ability to learn local patch features for improved discrimination.

## Pipeline

```
Raw Range-Doppler Map
        ↓
  CA-CFAR Detector        ← Classical: constant false alarm rate
  (threshold = α × noise estimate)
        ↓
  Extract patch around    ← 16×16 window centered on CFAR cell
  strongest detection
        ↓
  CNN Classifier          ← Learned: true target vs clutter
  (target / clutter)
        ↓
  Final Detection Decision
```

## CA-CFAR Implementation

For each cell under test (CUT):
- Surround with guard cells (prevent target leakage into noise estimate)
- Average power in outer training cells → noise estimate
- Threshold multiplier: `α = N × (Pfa^(-1/N) - 1)` where N = training cell count
- Declare detection if CUT power > threshold

No library used — implemented from scratch in NumPy.

## CNN Architecture

```
Input (1, 16, 16)
→ Conv2d(32) + BN + ReLU + MaxPool   → (32, 8, 8)
→ Conv2d(64) + BN + ReLU + MaxPool   → (64, 4, 4)
→ Conv2d(128) + BN + ReLU + AvgPool  → (128, 2, 2)
→ Flatten → FC(64) → Dropout → FC(2)
```

Shallow by design — CFAR handles detection, CNN only discriminates patch texture.

## Synthetic Data

No public labeled radar dataset at appropriate resolution, so scenes are simulated:
- **Clutter:** Rayleigh-distributed (exponential power) — standard model for ground/sea clutter
- **Target:** Gaussian PSF at random range/Doppler bin with configurable SNR (5–25 dB)
- **Validation:** Held-out scenes with same statistics, different random seed

## Results

| Metric | Value |
|---|---|
| Detection accuracy | ~91% |
| AUC-ROC | ~0.96 |
| False alarm rate | ≤ 10% (CFAR-controlled) |

## Usage

```bash
pip install -r requirements.txt
python train.py
```

No dataset download required — scenes are generated synthetically at runtime.

## Key Design Decisions

**Why CFAR first, CNN second?**  
CFAR provides a mathematically guaranteed false alarm rate regardless of noise level. The CNN then only needs to learn a binary discrimination problem on pre-selected cells, making training far more tractable than end-to-end detection from raw maps.

**Why synthetic data?**  
Labeled radar target datasets are scarce and often export-controlled. Synthetic data allows full control over SNR, target density, and clutter statistics while enabling reproducible experiments without data distribution agreements.

**Why 2D CNN over 1D?**  
Unlike the RF modulation classifier (temporal IQ stream), the CFAR patch is genuinely spatial — target PSF shape, clutter texture, and range-Doppler extent are 2D features best captured by 2D convolutions.

## References

- Richards, M. A. (2014). *Fundamentals of Radar Signal Processing*. McGraw-Hill.
- Rohling, H. (1983). Radar CFAR Thresholding in Clutter and Multiple Target Situations. *IEEE Transactions on Aerospace and Electronic Systems*.
