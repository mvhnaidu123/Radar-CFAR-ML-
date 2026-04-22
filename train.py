"""
Radar Target Detection: CFAR + Deep Learning
=============================================
Two-stage pipeline:
  Stage 1 — CA-CFAR (Cell Averaging CFAR): classical detector, constant false alarm rate
  Stage 2 — CNN classifier: distinguishes true targets from clutter using range-Doppler features

This directly mirrors real radar processing pipelines used in automotive (ADAS),
surveillance, and tracking systems.

Author: Your Name
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import os
import json

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "num_train_scenes":   4000,
    "num_val_scenes":     1000,
    "range_bins":         64,
    "doppler_bins":       64,
    "cfar_guard_cells":   2,
    "cfar_training_cells": 8,
    "cfar_pfa":           1e-3,    # Probability of false alarm target
    "snr_range_db":       (5, 25), # Target SNR range during training
    "batch_size":         128,
    "epochs":             25,
    "learning_rate":      1e-3,
    "seed":               42,
    "save_path":          "checkpoints/cfar_cnn_best.pth",
    "results_path":       "results_radar/"
}

# ─── Signal Simulation ────────────────────────────────────────────────────────

class RadarSceneGenerator:
    """
    Generates synthetic range-Doppler maps with targets and clutter.

    Range-Doppler map: 2D matrix where:
      - rows    = range bins (distance)
      - columns = Doppler bins (relative velocity via FFT)
      - values  = received power (linear scale)

    Clutter model: exponentially distributed (Rayleigh fading background)
    Target model:  Gaussian-shaped response at random range/Doppler bin + SNR
    """

    def __init__(self, range_bins=64, doppler_bins=64, seed=42):
        self.range_bins   = range_bins
        self.doppler_bins = doppler_bins
        np.random.seed(seed)

    def generate_clutter(self):
        """Rayleigh-distributed clutter (exponential power)."""
        return np.random.exponential(1.0, (self.range_bins, self.doppler_bins)).astype(np.float32)

    def add_target(self, scene, snr_db, range_bin=None, doppler_bin=None):
        """
        Add a point target with Gaussian spread at given range/Doppler.
        Returns scene with target and (range_bin, doppler_bin) of target center.
        """
        if range_bin is None:
            # Avoid edges (CFAR needs guard+training cell margin)
            margin = CONFIG['cfar_guard_cells'] + CONFIG['cfar_training_cells'] + 1
            range_bin   = np.random.randint(margin, self.range_bins - margin)
            doppler_bin = np.random.randint(margin, self.doppler_bins - margin)

        snr_linear = 10 ** (snr_db / 10)
        clutter_power = 1.0  # normalized
        target_amplitude = np.sqrt(snr_linear * clutter_power)

        # Gaussian target spread (PSF of real radar)
        rr, dd = np.mgrid[0:self.range_bins, 0:self.doppler_bins]
        target_response = target_amplitude * np.exp(
            -0.5 * (((rr - range_bin) / 1.5) ** 2 + ((dd - doppler_bin) / 1.5) ** 2)
        ).astype(np.float32)

        return scene + target_response, (range_bin, doppler_bin)

    def generate_scene(self, has_target=True, snr_db=10.0):
        """
        Returns:
          map_2d:     (range_bins, doppler_bins) float32 range-Doppler map
          label:      1 if target present, 0 if clutter only
          target_loc: (range_bin, doppler_bin) or None
        """
        scene = self.generate_clutter()
        if has_target:
            scene, loc = self.add_target(scene, snr_db)
            return scene, 1, loc
        return scene, 0, None


# ─── CFAR Detector ────────────────────────────────────────────────────────────

def ca_cfar_2d(rdmap, guard_cells=2, training_cells=8, pfa=1e-3):
    """
    2D Cell Averaging CFAR (CA-CFAR) detector.

    For each cell under test (CUT):
      1. Exclude the CUT and guard cells
      2. Average power in surrounding training cells → noise estimate
      3. Threshold = noise_estimate × alpha  (alpha derived from pfa)
      4. Declare detection if CUT > threshold

    Alpha derivation: For N training cells, CA-CFAR threshold multiplier is:
      alpha = N * (pfa^(-1/N) - 1)

    Returns binary detection map: 1 = detection, 0 = no detection.
    """
    R, D = rdmap.shape
    detection_map = np.zeros_like(rdmap, dtype=np.int32)

    N = (2 * (guard_cells + training_cells) + 1) ** 2 - (2 * guard_cells + 1) ** 2
    alpha = N * (pfa ** (-1.0 / N) - 1)

    margin = guard_cells + training_cells

    for r in range(margin, R - margin):
        for d in range(margin, D - margin):
            # Extract window around CUT
            window = rdmap[r - margin: r + margin + 1, d - margin: d + margin + 1].copy()

            # Zero out CUT and guard region
            g = guard_cells
            m = margin
            window[m - g: m + g + 1, m - g: m + g + 1] = 0

            # Estimate noise power from training cells
            noise_power = window.sum() / N

            # Threshold
            threshold = alpha * noise_power

            if rdmap[r, d] > threshold:
                detection_map[r, d] = 1

    return detection_map


def extract_cfar_patch(rdmap, detection_map, patch_size=16):
    """
    Find the strongest CFAR detection and extract a patch around it.
    If no detection, extract patch around peak power cell.
    Returns (patch_size, patch_size) normalized patch.
    """
    half = patch_size // 2
    R, D = rdmap.shape

    detections = np.argwhere(detection_map > 0)
    if len(detections) > 0:
        # Pick detection with highest power
        powers = [rdmap[r, d] for r, d in detections]
        best   = detections[np.argmax(powers)]
    else:
        best = np.array(np.unravel_index(rdmap.argmax(), rdmap.shape))

    r0, d0 = best
    r0 = np.clip(r0, half, R - half - 1)
    d0 = np.clip(d0, half, D - half - 1)

    patch = rdmap[r0 - half: r0 + half, d0 - half: d0 + half]

    # Normalize patch to [0, 1]
    p_min, p_max = patch.min(), patch.max()
    if p_max > p_min:
        patch = (patch - p_min) / (p_max - p_min)

    return patch.astype(np.float32)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class RadarDataset(Dataset):
    """
    Pre-generated radar scenes with CFAR preprocessing applied.
    Each sample: (1, patch_size, patch_size) normalized patch around strongest CFAR cell.
    Label: 1 (target) or 0 (clutter).
    """

    def __init__(self, num_scenes, generator, patch_size=16, snr_range=(5, 25)):
        self.patches = []
        self.labels  = []

        print(f"Generating {num_scenes} radar scenes...")
        for i in range(num_scenes):
            has_target = (i % 2 == 0)
            snr_db     = np.random.uniform(*snr_range) if has_target else 0.0
            rdmap, label, _ = generator.generate_scene(has_target=has_target, snr_db=snr_db)

            det_map = ca_cfar_2d(rdmap,
                                  guard_cells=CONFIG['cfar_guard_cells'],
                                  training_cells=CONFIG['cfar_training_cells'],
                                  pfa=CONFIG['cfar_pfa'])

            patch = extract_cfar_patch(rdmap, det_map, patch_size=patch_size)
            self.patches.append(patch[np.newaxis, :, :])  # (1, H, W)
            self.labels.append(label)

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{num_scenes}")

        self.patches = torch.tensor(np.array(self.patches), dtype=torch.float32)
        self.labels  = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


# ─── CNN Classifier ───────────────────────────────────────────────────────────

class RadarTargetCNN(nn.Module):
    """
    Lightweight CNN to classify CFAR patch as true target vs clutter.

    Input:  (batch, 1, 16, 16) — single-channel normalized range-Doppler patch
    Output: (batch, 2) — [clutter_score, target_score]

    Architecture is deliberately shallow — the CFAR stage already handles the
    detection problem. The CNN only needs to learn local patch texture differences
    between target PSF shape and clutter statistics.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (32, 8, 8)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (64, 4, 4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),                       # → (128, 2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * len(y)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels, all_probs


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_pipeline(generator, model, device, save_path):
    """
    Visualize one full pipeline: raw RD map → CFAR detections → CNN decision.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = ['Target Scene (RD Map)', 'CFAR Detections', 'CNN Input Patch',
              'Clutter Scene (RD Map)', 'CFAR Detections', 'CNN Input Patch']

    for row, has_target in enumerate([True, False]):
        snr_db = 12.0 if has_target else 0.0
        rdmap, label, target_loc = generator.generate_scene(has_target=has_target, snr_db=snr_db)
        det_map = ca_cfar_2d(rdmap)
        patch   = extract_cfar_patch(rdmap, det_map)

        # Predict
        x = torch.tensor(patch[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x)
            prob_target = torch.softmax(logits, dim=1)[0, 1].item()

        # Plot
        axes[row, 0].imshow(10 * np.log10(rdmap + 1e-6), cmap='hot', aspect='auto')
        axes[row, 0].set_title(f"{'Target' if has_target else 'Clutter'} (SNR={snr_db}dB)")
        if has_target and target_loc:
            axes[row, 0].plot(target_loc[1], target_loc[0], 'c+', markersize=12, markeredgewidth=2)

        axes[row, 1].imshow(det_map, cmap='Reds', aspect='auto')
        axes[row, 1].set_title(f"CFAR Detections ({det_map.sum()} cells)")

        axes[row, 2].imshow(patch, cmap='hot')
        axes[row, 2].set_title(f"CNN Patch | P(target)={prob_target:.3f}")

    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel('Doppler Bin')
        ax.set_ylabel('Range Bin')

    plt.suptitle('Radar Detection Pipeline: CFAR + CNN', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pipeline visualization saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(CONFIG['results_path'], exist_ok=True)

    generator = RadarSceneGenerator(CONFIG['range_bins'], CONFIG['doppler_bins'], CONFIG['seed'])

    train_ds = RadarDataset(CONFIG['num_train_scenes'], generator, snr_range=CONFIG['snr_range_db'])
    val_ds   = RadarDataset(CONFIG['num_val_scenes'],   generator, snr_range=CONFIG['snr_range_db'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], shuffle=False)

    model     = RadarTargetCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels, val_probs = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        auc = roc_auc_score(val_labels, val_probs)
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"Train: {train_loss:.4f}/{train_acc:.3f} | "
              f"Val: {val_loss:.4f}/{val_acc:.3f} | AUC: {auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"  ✓ Best model saved")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    print("\nClassification Report:")
    _, _, final_preds, final_labels, final_probs = eval_epoch(model, val_loader, criterion, device)
    print(classification_report(final_labels, final_preds, target_names=['Clutter', 'Target']))

    visualize_pipeline(generator, model, device,
                       os.path.join(CONFIG['results_path'], 'pipeline_visualization.png'))

    with open(os.path.join(CONFIG['results_path'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
