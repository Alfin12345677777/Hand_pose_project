"""
export_for_blender.py
=====================
Generates two Blender Python scripts:
  blender_setup.py  — imports mesh, creates armature, applies auto weights
  blender_export.py — exports painted weights as .npy for training

Usage:
  python scripts/export_for_blender.py
  Then open Blender → Scripting → Open → blender_setup.py → Run Script
"""

import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
MESHES_DIR   = os.path.join(PROJECT_ROOT, 'meshes')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

SKELETON_JSON = os.path.join(MODELS_DIR, 'korean_hand_skeleton.json')
KOREAN_OBJ    = os.path.join(MESHES_DIR, 'AVERAGE_KOREAN_HAND_CENTERED.obj')

os.makedirs(OUTPUTS_DIR, exist_ok=True)

with open(SKELETON_JSON) as f:
    skel = json.load(f)

JOINT_NAMES_16 = [
    'wrist',
    'index1',  'index2',  'index3',
    'middle1', 'middle2', 'middle3',
    'pinky1',  'pinky2',  'pinky3',
    'ring1',   'ring2',   'ring3',
    'thumb1',  'thumb2',  'thumb3',
]

PARENTS = {
    'wrist': None,
    'index1': 'wrist',   'index2': 'index1',   'index3': 'index2',
    'middle1': 'wrist',  'middle2': 'middle1',  'middle3': 'middle2',
    'pinky1': 'wrist',   'pinky2': 'pinky1',   'pinky3': 'pinky2',
    'ring1': 'wrist',    'ring2': 'ring1',      'ring3': 'ring2',
    'thumb1': 'wrist',   'thumb2': 'thumb1',    'thumb3': 'thumb2',
}

# Build joint positions in meters (Blender units)
joints_lines = []
for name in JOINT_NAMES_16:
    pos = skel[name]['position_mm']
    joints_lines.append(
        f"    '{name}': [{pos[0]/1000:.6f}, {pos[1]/1000:.6f}, {pos[2]/1000:.6f}],"
    )
joints_block = "{\n" + "\n".join(joints_lines) + "\n}"

parents_lines = []
for name in JOINT_NAMES_16:
    p = PARENTS[name]
    if p is None:
        parents_lines.append(f"    '{name}': None,")
    else:
        parents_lines.append(f"    '{name}': '{p}',")
parents_block = "{\n" + "\n".join(parents_lines) + "\n}"

bone_order_str = repr(JOINT_NAMES_16)
weights_output = os.path.join(DATA_DIR, 'mano_transferred_weights.npy')

# ── Blender Setup Script ─────────────────────────────────────────────────
setup_script = '''"""
Blender Setup Script - Korean Hand Rigging
Run this in Blender's Scripting tab.
"""

import bpy
import mathutils

# Clean scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ── 1. Import mesh ──
mesh_path = "''' + KOREAN_OBJ + '''"
bpy.ops.wm.obj_import(filepath=mesh_path)

mesh_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        mesh_obj = obj
        break
if mesh_obj is None:
    raise RuntimeError("No mesh imported!")
mesh_obj.name = "KoreanHand"
print(f"Imported: {mesh_obj.name}, {len(mesh_obj.data.vertices)} verts")

# ── 2. Create armature ──
bpy.ops.object.armature_add(enter_editmode=True)
armature_obj = bpy.context.object
armature_obj.name = "HandArmature"
armature = armature_obj.data
armature.name = "HandArmature"

# Remove default bone
for bone in armature.edit_bones:
    armature.edit_bones.remove(bone)

joint_positions = ''' + joints_block + '''

parent_map = ''' + parents_block + '''

bone_order = ''' + bone_order_str + '''

# Create bones
bones = {}
for bone_name in bone_order:
    parent_name = parent_map[bone_name]
    head = mathutils.Vector(joint_positions[bone_name])

    bone = armature.edit_bones.new(bone_name)
    bone.head = head

    if parent_name is not None and parent_name in bones:
        parent_bone = bones[parent_name]
        bone.parent = parent_bone
        parent_bone.tail = head

    # Default tail (adjusted below for tips)
    bone.tail = head + mathutils.Vector((0, 0.005, 0))
    bones[bone_name] = bone

# Fix tip bone tails: extend along parent direction
for tip_name in ['index3', 'middle3', 'pinky3', 'ring3', 'thumb3']:
    bone = bones[tip_name]
    parent_name = parent_map[tip_name]
    if parent_name:
        parent = bones[parent_name]
        direction = (bone.head - parent.head).normalized()
        bone.tail = bone.head + direction * 0.015

bpy.ops.object.mode_set(mode='OBJECT')

# ── 3. Parent mesh to armature with automatic weights ──
bpy.ops.object.select_all(action='DESELECT')
mesh_obj.select_set(True)
armature_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj

bpy.ops.object.parent_set(type='ARMATURE_AUTO')

print()
print("=" * 60)
print("  DONE! Mesh parented with automatic weights.")
print("=" * 60)
print()
print("Next steps:")
print("  1. Select KoreanHand mesh")
print("  2. Ctrl+Tab → Weight Paint mode")
print("  3. Click vertex groups in Properties → Object Data")
print("  4. Test: select HandArmature → Pose Mode → rotate bones")
print("  5. Fix weights → run blender_export.py to save")
'''

setup_path = os.path.join(OUTPUTS_DIR, 'blender_setup.py')
with open(setup_path, 'w') as f:
    f.write(setup_script)
print(f'Saved: {setup_path}')


# ── Blender Export Script ─────────────────────────────────────────────────
export_script = '''"""
Blender Export Script - Save Weights
Run this AFTER weight painting is done.
"""

import bpy
import numpy as np
import os

output_path = "''' + weights_output + '''"
bone_names = ''' + bone_order_str + '''

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
print(f"\\nExported: {weights.shape}")
print(f"Hard (>0.95): {hard}/{n_verts} ({100*hard/n_verts:.1f}%)")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, weights)
print(f"Saved: {output_path}")

backup = output_path.replace('.npy', '_blender.npy')
np.save(backup, weights)
print(f"Backup: {backup}")
'''

export_path = os.path.join(OUTPUTS_DIR, 'blender_export.py')
with open(export_path, 'w') as f:
    f.write(export_script)
print(f'Saved: {export_path}')


print(f'''
========================================
  Blender Rigging Workflow
========================================

1. Open Blender

2. Scripting tab → Open → Run:
   {setup_path}
   (imports mesh + armature + auto weights)

3. Test the auto weights:
   - Select "HandArmature" → Pose Mode
   - Select a finger bone → R to rotate
   - Does it bend correctly?
   - Alt+R to reset

4. Fix problems in Weight Paint mode:
   - Select "KoreanHand" → Ctrl+Tab → Weight Paint
   - Select vertex groups (one per bone)
   - Red = full influence, Blue = none
   - Focus on knuckles and finger webbing

5. Export weights:
   Scripting tab → Open → Run:
   {export_path}
   (saves to data/mano_transferred_weights.npy)

6. Retrain:
   python scripts/train_unified_model.py
''')
