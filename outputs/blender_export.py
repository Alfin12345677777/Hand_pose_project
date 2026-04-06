"""
Blender Export Script - Save Weights
Run this AFTER weight painting is done.
"""

import bpy
import numpy as np
import os

output_path = "/Users/alfinwilliam/Documents/hand_pose_project/data/mano_transferred_weights.npy"
bone_names = ['wrist', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3']

mesh_obj = bpy.data.objects.get("KoreanHand")
if mesh_obj is None:
    raise RuntimeError("Can't find 'KoreanHand' mesh!")

n_verts = len(mesh_obj.data.vertices)
n_joints = len(bone_names)
weights = np.zeros((n_verts, n_joints), dtype=np.float32)

# Map vertex group names to joint indices
vg_map = {}
for i, bone_name in enumerate(bone_names):
    vg = mesh_obj.vertex_groups.get(bone_name)
    if vg is not None:
        vg_map[vg.index] = i
    else:
        print(f"WARNING: No vertex group for bone '{bone_name}'")

# Extract weights
for v in mesh_obj.data.vertices:
    for g in v.groups:
        if g.group in vg_map:
            joint_idx = vg_map[g.group]
            weights[v.index, joint_idx] = g.weight

# Normalize
row_sums = weights.sum(axis=1, keepdims=True)
weights /= np.maximum(row_sums, 1e-8)

# Stats
hard = (weights.max(axis=1) > 0.95).sum()
print(f"\nExported: {weights.shape}")
print(f"Hard (>0.95): {hard}/{n_verts} ({100*hard/n_verts:.1f}%)")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, weights)
print(f"Saved: {output_path}")

backup = output_path.replace('.npy', '_blender.npy')
np.save(backup, weights)
print(f"Backup: {backup}")
