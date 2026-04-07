"""
train_pose_lifter.py
====================
Trains a small MLP to lift normalized 2D hand landmarks → MANO pose parameters.

Input:  (42,)  normalized 2D landmarks  (21 joints × xy)
Output: (48,)  MANO pose params         (global_orient(3) + hand_joints(45))

Architecture:
  42 → 256 → 256 → 128 → 48   (ReLU + BatchNorm, dropout 0.2)

At inference:
  MediaPipe 21 landmarks → normalize → this model → pose params
  → LBS on K-MANO average → Korean hand mesh
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import bone feature extractor from unified model
sys.path.insert(0, os.path.dirname(__file__))
from train_unified_model import extract_bone_features

# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
LM_NPY    = '../data/landmarks_2d.npy'
POSE_NPY  = '../data/pose_params.npy'
MODEL_OUT = '../models/pose_lifter.pth'
CHART_OUT = '../outputs/pose_lifter_training.png'

# ──────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────
BATCH_SIZE  = 256
LR          = 1e-3
EPOCHS      = 500
VAL_SPLIT   = 0.1
DROPOUT     = 0.2
RANDOM_SEED = 42
AUGMENT_NOISE = 0.02   # random landmark noise for data augmentation

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────
class LandmarkPoseDataset(Dataset):
    def __init__(self, landmarks, poses, augment=False):
        self.X = torch.tensor(landmarks, dtype=torch.float32)
        self.Y = torch.tensor(poses,     dtype=torch.float32)
        self.augment = augment

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.augment:
            # Small random noise on landmarks — simulates MediaPipe jitter
            x = x + torch.randn_like(x) * AUGMENT_NOISE
            x = torch.clamp(x, -5.0, 5.0)
        return x, y


# ──────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────
class PoseLifter(nn.Module):
    """Wider + deeper with residual connections + bone length features (matches PoseEncoder)."""
    def __init__(self, in_dim=52, out_dim=48, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res1 = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512),
        )
        self.res2 = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512),
        )
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        import torch.nn.functional as F
        h = self.input_proj(x)
        h = F.relu(h + self.res1(h))
        h = F.relu(h + self.res2(h))
        return self.output_head(h)


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # ── Load data ───────────────────────────
    print("\nLoading data...")
    X = np.load(LM_NPY)
    Y = np.load(POSE_NPY)
    print(f"  Landmarks : {X.shape}")
    print(f"  Poses     : {Y.shape}")

    N        = len(X)
    n_val    = int(N * VAL_SPLIT)
    n_train  = N - n_val
    idx      = np.random.RandomState(RANDOM_SEED).permutation(N)
    train_i, val_i = idx[:n_train], idx[n_train:]

    train_ds = LandmarkPoseDataset(X[train_i], Y[train_i], augment=True)
    val_ds   = LandmarkPoseDataset(X[val_i],   Y[val_i],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"  Train: {n_train}  Val: {n_val}")

    # ── Model ───────────────────────────────
    model     = PoseLifter().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # ── Training loop ───────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"\nTraining for {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            bone_feat = extract_bone_features(xb)  # (B, 10)
            xb_full = torch.cat([xb, bone_feat], dim=-1)  # (B, 52)
            optimizer.zero_grad()
            pred = model(xb_full)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                bone_feat = extract_bone_features(xb)
                xb_full = torch.cat([xb, bone_feat], dim=-1)
                val_loss += criterion(model(xb_full), yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUT)

        if (epoch + 1) % 25 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
                  f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
                  f"LR: {lr_now:.2e}  {'← best' if val_loss == best_val_loss else ''}")

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Model saved → {MODEL_OUT}")

    # ── Plot ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss', color='#FF6B6B', linewidth=2)
    ax.plot(val_losses,   label='Val Loss',   color='#4ECDC4', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (pose params, rad²)')
    ax.set_title('Pose Lifter — 2D Landmarks → MANO Pose Parameters', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(CHART_OUT, dpi=120)
    print(f"Chart saved → {CHART_OUT}")
