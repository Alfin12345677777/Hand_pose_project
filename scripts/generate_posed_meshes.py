"""
generate_posed_meshes.py  — FreiHAND Edition
=============================================
Uses the FreiHAND dataset (32,560 real hand poses, already included) to
generate posed hand meshes. No synthetic LBS needed — FreiHAND vertices
are already correctly articulated MANO meshes from real captures.

Pipeline per sample:
  1. Load 3D vertices from training_verts.json  (778 verts, camera space, meters)
  2. Root-center at wrist joint (training_xyz.json[i][0])
  3. Convert to mm  (×1000)
  4. Save as .obj with MANO faces

Output: data/posed_meshes/*.obj  (778 verts, MANO topology)
"""

import os
import json
import pickle
import numpy as np
import trimesh
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
VERTS_JSON  = '../data/training_verts.json'
XYZ_JSON    = '../data/training_xyz.json'
MANO_PKL    = '../models/MANO_RIGHT.pkl'
OUTPUT_DIR  = '../data/posed_meshes'

N_SAMPLES   = 20000
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────
print("[1] Loading FreiHAND data...")
with open(VERTS_JSON) as f:
    all_verts = json.load(f)
with open(XYZ_JSON) as f:
    all_xyz = json.load(f)

print(f"    Total samples : {len(all_verts)}")
print(f"    Vertices/mesh : {len(all_verts[0])} (MANO topology)")

# ──────────────────────────────────────────
# 2. LOAD MANO FACES
# ──────────────────────────────────────────
print("\n[2] Loading MANO faces...")
with open(MANO_PKL, 'rb') as f:
    mano = pickle.load(f, encoding='latin1')
faces = np.array(mano['f'], dtype=np.int32)   # (1538, 3)
print(f"    Faces : {faces.shape}")

# ──────────────────────────────────────────
# 3. SAMPLE INDICES
# ──────────────────────────────────────────
indices = np.random.choice(len(all_verts), size=N_SAMPLES, replace=False)
print(f"\n[3] Sampling {N_SAMPLES} poses from {len(all_verts)} available\n")

# ──────────────────────────────────────────
# 4. GENERATE MESHES
# ──────────────────────────────────────────
for out_idx, src_idx in enumerate(tqdm(indices, desc="Generating", unit="mesh")):

    # Load vertices (camera space, meters)
    v = np.array(all_verts[src_idx], dtype=np.float32)          # (778, 3)

    # Root-center at wrist joint, convert to mm
    wrist = np.array(all_xyz[src_idx][0], dtype=np.float32)     # (3,)
    v_mm  = (v - wrist) * 1000.0                                 # (778, 3) mm

    # Save
    mesh = trimesh.Trimesh(vertices=v_mm, faces=faces, process=False)
    mesh.export(os.path.join(OUTPUT_DIR, f'posed_{out_idx:05d}.obj'))

print(f"\nDone! {N_SAMPLES} meshes saved to: {OUTPUT_DIR}")
print(f"Disk usage: ~{N_SAMPLES * 0.12:.0f} MB")
print(f"\nTo generate 10,000: change N_SAMPLES = 10000  (~{10000*0.12/1000:.1f} GB)")
