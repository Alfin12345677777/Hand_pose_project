"""
Blender Setup Script - Korean Hand Rigging
Run this in Blender's Scripting tab.
"""

import bpy
import mathutils

# Clean scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ── 1. Import mesh ──
mesh_path = "/Users/alfinwilliam/Documents/hand_pose_project/meshes/AVERAGE_KOREAN_HAND_CENTERED.obj"
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

joint_positions = {
    'wrist': [-0.001654, -0.086545, -0.019536],
    'index1': [0.012736, -0.018175, -0.008484],
    'index2': [0.021731, 0.024556, -0.001576],
    'index3': [0.027847, 0.053614, 0.003121],
    'middle1': [-0.001706, -0.010550, -0.007549],
    'middle2': [-0.001738, 0.036947, -0.000056],
    'middle3': [-0.001760, 0.069244, 0.005039],
    'pinky1': [-0.020450, -0.018852, -0.008272],
    'pinky2': [-0.032196, 0.023456, -0.001231],
    'pinky3': [-0.040184, 0.052226, 0.003556],
    'ring1': [-0.015691, -0.017592, -0.008454],
    'ring2': [-0.024464, 0.025504, -0.001528],
    'ring3': [-0.030430, 0.054809, 0.003182],
    'thumb1': [0.017487, -0.019681, -0.008886],
    'thumb2': [0.029450, 0.022110, -0.002230],
    'thumb3': [0.037585, 0.050527, 0.002296],
}

parent_map = {
    'wrist': None,
    'index1': 'wrist',
    'index2': 'index1',
    'index3': 'index2',
    'middle1': 'wrist',
    'middle2': 'middle1',
    'middle3': 'middle2',
    'pinky1': 'wrist',
    'pinky2': 'pinky1',
    'pinky3': 'pinky2',
    'ring1': 'wrist',
    'ring2': 'ring1',
    'ring3': 'ring2',
    'thumb1': 'wrist',
    'thumb2': 'thumb1',
    'thumb3': 'thumb2',
}

bone_order = ['wrist', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3']

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
