import os
import json
import pickle
import numpy as np
import trimesh
from tqdm import tqdm

# ==========================================
# PATHS
# ==========================================
MANO_PKL     = '../models/MANO_RIGHT.pkl'
MESH_DIR     = '../data/posed_meshes'
OUTPUT_JSON  = '../data/training_labels.json'

# Standard MANO Fingertip Vertices
FINGERTIP_VERTICES = [745, 333, 444, 555, 672] 

print("--- INITIALIZING FAST EXTRACTION (778-VERTEX MODE) ---")

# ==========================================
# 1. LOAD REGRESSOR
# ==========================================
with open(MANO_PKL, 'rb') as f:
    mano = pickle.load(f, encoding='latin1')
J_reg_mano = np.array(mano['J_regressor'].todense(), dtype=np.float64)

# ==========================================
# 2. BATCH PROCESS MESHES
# ==========================================
obj_files = sorted([f for f in os.listdir(MESH_DIR) if f.endswith('.obj')])
print(f"Found {len(obj_files)} meshes to process.")

training_data = {}

for obj_file in tqdm(obj_files, desc="Extracting Joints"):
    filepath = os.path.join(MESH_DIR, obj_file)
    
    # Fast load vertices (Skip face processing to save time)
    mesh = trimesh.load(filepath, process=False)
    v_posed = np.array(mesh.vertices)
    
    # Multiply the 778 vertices directly by the regressor
    joints_16 = J_reg_mano @ v_posed
    fingertips = v_posed[FINGERTIP_VERTICES]
    joints_21 = np.vstack([joints_16, fingertips])
    
    # Save to dictionary
    training_data[obj_file] = np.round(joints_21, 4).tolist()

# ==========================================
# 3. EXPORT TO JSON
# ==========================================
print("\nSaving to JSON...")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(training_data, f)

print(f"Success! Extracted {len(obj_files)} joint sets to {OUTPUT_JSON}")