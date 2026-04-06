"""
prepare_training_data.py
========================
Converts FreiHAND data into (input, target) numpy arrays for training.

Input  → 21 MediaPipe-style 2D landmarks, normalized (42 values)
Target → 48 MANO pose parameters (global_orient(3) + hand_joints(45))

Normalization:
  - Center: subtract wrist (landmark 0)
  - Scale:  divide by wrist→middle_tip distance  (scale-invariant)

Output:
  data/landmarks_2d.npy   shape (N, 42)   float32
  data/pose_params.npy    shape (N, 48)   float32
"""

import json
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
XYZ_JSON   = '../data/training_xyz.json'
K_JSON     = '../data/training_K.json'
MANO_JSON  = '../data/training_mano.json'
OUT_LM     = '../data/landmarks_2d.npy'
OUT_POSE   = '../data/pose_params.npy'

# FreiHAND joint 12 = middle fingertip (same order as MANO)
MIDDLE_TIP_IDX = 12

# ──────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────
print("Loading FreiHAND data...")
with open(XYZ_JSON)  as f: all_xyz   = json.load(f)
with open(K_JSON)    as f: all_K     = json.load(f)
with open(MANO_JSON) as f: all_mano  = json.load(f)

N = len(all_xyz)
print(f"  Samples: {N}")

# ──────────────────────────────────────────
# BUILD ARRAYS
# ──────────────────────────────────────────
landmarks_2d = np.zeros((N, 42), dtype=np.float32)
pose_params  = np.zeros((N, 48), dtype=np.float32)

skipped = 0
for i in tqdm(range(N), desc="Processing"):
    joints_3d = np.array(all_xyz[i],     dtype=np.float64)   # (21, 3) meters
    K         = np.array(all_K[i],       dtype=np.float64)   # (3, 3)
    pose      = np.array(all_mano[i][0], dtype=np.float32)   # (61,)

    # ── Project 3D joints → 2D pixels ──────
    # Perspective projection: u = fx*X/Z + cx
    Z  = joints_3d[:, 2]
    if np.any(Z <= 0):
        skipped += 1
        continue

    u = (K[0, 0] * joints_3d[:, 0] / Z) + K[0, 2]   # (21,)
    v = (K[1, 1] * joints_3d[:, 1] / Z) + K[1, 2]   # (21,)
    lm = np.stack([u, v], axis=1)                      # (21, 2) pixels

    # ── Normalize ───────────────────────────
    # 1. Center at wrist (landmark 0)
    wrist_2d = lm[0].copy()
    lm -= wrist_2d

    # 2. Scale by wrist→middle_tip distance (makes it size-invariant)
    scale = np.linalg.norm(lm[MIDDLE_TIP_IDX])
    if scale < 1e-6:
        skipped += 1
        continue
    lm /= scale

    landmarks_2d[i] = np.clip(lm.flatten(), -5.0, 5.0)   # (42,) clip outliers
    pose_params[i]  = pose[:48]             # pose(48) only, skip shape+trans

# ──────────────────────────────────────────
# TRIM SKIPPED ROWS
# ──────────────────────────────────────────
if skipped > 0:
    valid = np.any(landmarks_2d != 0, axis=1)
    landmarks_2d = landmarks_2d[valid]
    pose_params  = pose_params[valid]
    print(f"  Skipped {skipped} invalid samples")

# ──────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────
np.save(OUT_LM,   landmarks_2d)
np.save(OUT_POSE, pose_params)

print(f"\nSaved:")
print(f"  {OUT_LM}   {landmarks_2d.shape}")
print(f"  {OUT_POSE}  {pose_params.shape}")
print(f"\nInput  range: [{landmarks_2d.min():.3f}, {landmarks_2d.max():.3f}]")
print(f"Target range: [{pose_params.min():.3f}, {pose_params.max():.3f}] rad")
