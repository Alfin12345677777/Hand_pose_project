"""
eda_ego_data.py
===============
EDA for the Ego-Vision hand dataset (ego_data.zip).

Answers:
  1. How many samples, subjects, poses?
  2. Mesh topology — vertex/face counts consistent?
  3. Coordinate system — camera-space vs world-space?
  4. Keypoint format — same as grasping / FreiHAND?
  5. Scale — how does ego mesh relate to Korean template?
  6. Registration feasibility — can we map 7180 ego verts → 11279 Korean?

Output: outputs/eda_ego_data.png

Usage:
  my_venv/bin/python scripts/eda_ego_data.py
"""

import os, sys, json, zipfile, ast
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ZIP_PATH     = '/media/user/My Passport/Sample/ego_data.zip'
KOREAN_TMPL  = '/home/user/Documents/Handpose_project/meshes/AVERAGE_KOREAN_HAND_CENTERED.obj'
OUTPUT_PNG   = '/home/user/Documents/Handpose_project/outputs/eda_ego_data.png'

os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Inventory the zip
# ---------------------------------------------------------------------------
print('=== 1. Inventorying zip ===')
z = zipfile.ZipFile(ZIP_PATH)
all_files = z.namelist()

obj_r  = sorted([f for f in all_files if f.endswith('_mesh_R.obj')])
obj_l  = sorted([f for f in all_files if f.endswith('_mesh_L.obj')])
jsons  = sorted([f for f in all_files if f.endswith('.json')])
jpgs   = sorted([f for f in all_files if f.endswith('.jpg')])
pngs   = sorted([f for f in all_files if f.endswith('.png')])

print(f'  Total files  : {len(all_files)}')
print(f'  JSON labels  : {len(jsons)}')
print(f'  mesh_R.obj   : {len(obj_r)}')
print(f'  mesh_L.obj   : {len(obj_l)}')
print(f'  JPG images   : {len(jpgs)}')
print(f'  PNG depth    : {len(pngs)}')

# Extract subject/session IDs from filenames
session_ids = set()
for f in obj_r:
    parts = f.split('/')
    session_ids.add(parts[-2])  # e.g. 265_25_110431
print(f'  Unique sessions: {len(session_ids)}  → {sorted(session_ids)[:5]} ...')

# Pose class from path
pose_classes = set(f.split('/')[2] for f in obj_r if '/' in f)
print(f'  Pose classes : {sorted(pose_classes)}')

# ---------------------------------------------------------------------------
# 2. Mesh topology check (all frames)
# ---------------------------------------------------------------------------
print('\n=== 2. Mesh topology ===')
vert_counts_r, face_counts_r = [], []
vert_counts_l, face_counts_l = [], []

for fn in obj_r:
    data = z.read(fn).decode('utf-8', errors='ignore')
    vc = sum(1 for l in data.splitlines() if l.startswith('v '))
    fc = sum(1 for l in data.splitlines() if l.startswith('f '))
    vert_counts_r.append(vc); face_counts_r.append(fc)

for fn in obj_l:
    data = z.read(fn).decode('utf-8', errors='ignore')
    vc = sum(1 for l in data.splitlines() if l.startswith('v '))
    fc = sum(1 for l in data.splitlines() if l.startswith('f '))
    vert_counts_l.append(vc); face_counts_l.append(fc)

print(f'  Right hand verts: {set(vert_counts_r)}  faces: {set(face_counts_r)}')
print(f'  Left  hand verts: {set(vert_counts_l)}  faces: {set(face_counts_l)}')
topology_consistent = len(set(vert_counts_r)) == 1 and len(set(vert_counts_l)) == 1
print(f'  Topology consistent: {topology_consistent}')

N_EGO   = vert_counts_r[0]
N_KOR   = len(trimesh.load(KOREAN_TMPL, process=False).vertices)
print(f'  Ego verts: {N_EGO}  |  Korean template verts: {N_KOR}')
print(f'  → Different topology: need one-time nearest-neighbour registration')

# ---------------------------------------------------------------------------
# 3. Parse all JSON keypoints
# ---------------------------------------------------------------------------
print('\n=== 3. Parsing keypoints ===')
all_kp_r, all_kp_l = [], []
category_names = set()
actor_ages, actor_heights, actor_sexes = [], [], []

for jf in jsons:
    try:
        d = json.loads(z.read(jf).decode('utf-8'))
        actor_ages.append(d['actor']['age'])
        actor_heights.append(d['actor']['height'])
        actor_sexes.append(d['actor']['sex'])
        category_names.add(d['annotations'][0]['category_name'])

        kp_ann = next((a for a in d['annotations'] if a['type'] == 'keypoint_b'), None)
        if kp_ann:
            kp = ast.literal_eval(kp_ann['data'])
            all_kp_r.append(np.array(kp[0], dtype=np.float32))
            all_kp_l.append(np.array(kp[1], dtype=np.float32))
    except Exception as e:
        pass

all_kp_r = np.stack(all_kp_r)   # (N, 21, 3)
all_kp_l = np.stack(all_kp_l)   # (N, 21, 3)
print(f'  Loaded keypoints: {len(all_kp_r)} frames (right), {len(all_kp_l)} (left)')
print(f'  Category names: {sorted(category_names)}')
print(f'  Actor ages: {sorted(set(actor_ages))}')
print(f'  Actor heights: {sorted(set(actor_heights))}')
print(f'  Actor sexes: {sorted(set(actor_sexes))}')

# Keypoint z stats (wrist z=0, rest are relative depths in mm)
z_vals = all_kp_r[:, 1:, 2]   # exclude wrist (always 0)
print(f'\n  Right hand Z depth stats (mm, wrist-relative):')
print(f'    min={z_vals.min():.1f}  max={z_vals.max():.1f}  '
      f'mean={z_vals.mean():.1f}  std={z_vals.std():.1f}')

# Hand size in pixels (wrist→middle3 = joint 15 in this format)
hand_size_px = np.linalg.norm(
    all_kp_r[:, 15, :2] - all_kp_r[:, 0, :2], axis=1
)
print(f'\n  Right hand size in pixels (wrist→middle3):')
print(f'    min={hand_size_px.min():.0f}  max={hand_size_px.max():.0f}  '
      f'mean={hand_size_px.mean():.0f}  std={hand_size_px.std():.0f}')

# ---------------------------------------------------------------------------
# 4. Load one ego mesh and compare to Korean template
# ---------------------------------------------------------------------------
print('\n=== 4. Mesh scale & coordinate system ===')
ego_r = trimesh.load('/tmp/110431_100_mesh_R.obj', process=False)
kor   = trimesh.load(KOREAN_TMPL, process=False)
ev = np.array(ego_r.vertices)
kv = np.array(kor.vertices)

# Ego is in camera-space mm; centre it at wrist
# Wrist pixel from JSON: [1384.43, 941.22, 0.0] — use mesh centroid as proxy
ev_c = ev - ev.mean(axis=0)
kv_c = kv - kv.mean(axis=0)

ego_extent = ev_c.max(axis=0) - ev_c.min(axis=0)
kor_extent = kv_c.max(axis=0) - kv_c.min(axis=0)
print(f'  Ego mesh extent (centred): {ego_extent}')
print(f'  Korean template extent:    {kor_extent}')
scale_ratio = ego_extent / kor_extent
print(f'  Scale ratio ego/korean:    {scale_ratio}')
print(f'  → Both ~mm scale, ego is ~{scale_ratio.mean():.1f}x larger in extent '
      f'(camera-space coords vs wrist-relative)')

# ---------------------------------------------------------------------------
# 5. Registration test: nearest-neighbour ego→Korean
# ---------------------------------------------------------------------------
print('\n=== 5. Registration feasibility ===')
from scipy.spatial import cKDTree

# Rough scale-normalise both to unit cube for NN test
ev_n = ev_c / np.linalg.norm(ev_c, axis=1, keepdims=True).max()
kv_n = kv_c / np.linalg.norm(kv_c, axis=1, keepdims=True).max()

tree = cKDTree(ev_n)
dists, idxs = tree.query(kv_n)   # for each Korean vert, nearest ego vert
print(f'  NN dists (Korean→Ego, normalised units):')
print(f'    mean={dists.mean():.4f}  median={np.median(dists):.4f}  '
      f'max={dists.max():.4f}  95pct={np.percentile(dists,95):.4f}')
print(f'  → {"Good match — NN registration feasible" if np.median(dists) < 0.05 else "Rough match — ICP needed after NN init"}')

# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------
print('\n=== 6. Plotting ===')
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#1a1a1a')

def dark_ax(ax, title):
    ax.set_facecolor('#1a1a1a')
    ax.set_title(title, color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

# -- A: Vertex count distribution
ax1 = fig.add_subplot(3, 4, 1)
dark_ax(ax1, 'Mesh Vert Counts\n(all frames)')
ax1.bar(['Ego R', 'Ego L', 'Korean'],
        [vert_counts_r[0], vert_counts_l[0], N_KOR],
        color=['#4499cc', '#ee8844', '#55cc55'])
ax1.set_ylabel('Vertex count', color='white')
for bar, val in zip(ax1.patches, [vert_counts_r[0], vert_counts_l[0], N_KOR]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(val), ha='center', color='white', fontsize=8)

# -- B: Z depth distribution (right hand, all joints except wrist)
ax2 = fig.add_subplot(3, 4, 2)
dark_ax(ax2, 'Z Depth Distribution\n(right hand, mm, wrist-relative)')
ax2.hist(z_vals.flatten(), bins=40, color='#4499cc', edgecolor='none', alpha=0.85)
ax2.set_xlabel('Z depth (mm)', color='white')
ax2.set_ylabel('Count', color='white')
ax2.axvline(z_vals.mean(), color='#ffaa44', linewidth=1.5, label=f'mean={z_vals.mean():.0f}mm')
ax2.legend(fontsize=7, labelcolor='white', facecolor='#333333')

# -- C: Hand size in pixels
ax3 = fig.add_subplot(3, 4, 3)
dark_ax(ax3, 'Hand Size in Pixels\n(wrist→middle3, right hand)')
ax3.hist(hand_size_px, bins=20, color='#55cc55', edgecolor='none', alpha=0.85)
ax3.set_xlabel('Pixels', color='white')
ax3.axvline(hand_size_px.mean(), color='#ffaa44', linewidth=1.5,
            label=f'mean={hand_size_px.mean():.0f}px')
ax3.legend(fontsize=7, labelcolor='white', facecolor='#333333')

# -- D: Wrist position spread (pixel XY) across all frames
ax4 = fig.add_subplot(3, 4, 4)
dark_ax(ax4, 'Wrist Position Spread\n(pixel XY, right hand)')
ax4.scatter(all_kp_r[:, 0, 0], all_kp_r[:, 0, 1],
            s=8, alpha=0.6, color='#ee5555')
ax4.scatter(all_kp_l[:, 0, 0], all_kp_l[:, 0, 1],
            s=8, alpha=0.6, color='#4499cc', label='left')
ax4.set_xlabel('X pixel', color='white'); ax4.set_ylabel('Y pixel', color='white')
ax4.invert_yaxis()
ax4.legend(['right', 'left'], fontsize=7, labelcolor='white', facecolor='#333333')

# -- E: Mean 2D hand pose (normalised, right hand)
ax5 = fig.add_subplot(3, 4, 5)
dark_ax(ax5, 'Mean Normalised 2D Pose\n(right hand)')
kp_norm = all_kp_r[:, :, :2] - all_kp_r[:, :1, :2]
scale_n = np.linalg.norm(kp_norm[:, 15, :], axis=1, keepdims=True)[:, :, None]
kp_norm = kp_norm / (scale_n + 1e-6)
mean_kp = kp_norm.mean(axis=0)
std_kp  = kp_norm.std(axis=0)
bones = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]
for a, b in bones:
    ax5.plot([mean_kp[a,0], mean_kp[b,0]], [mean_kp[a,1], mean_kp[b,1]],
             'w-', linewidth=1, alpha=0.6)
ax5.scatter(mean_kp[:,0], mean_kp[:,1], s=20, c='#ffaa44', zorder=5)
ax5.invert_yaxis()
ax5.set_aspect('equal')

# -- F: Per-joint Z depth across frames (violin-style box)
ax6 = fig.add_subplot(3, 4, 6)
dark_ax(ax6, 'Per-Joint Z Depth\n(right hand, mm)')
joint_z = [all_kp_r[:, j, 2] for j in range(21)]
bp = ax6.boxplot(joint_z, patch_artist=True, medianprops={'color':'white'})
for patch in bp['boxes']:
    patch.set_facecolor('#4499cc'); patch.set_alpha(0.7)
ax6.set_xlabel('Joint index', color='white'); ax6.set_ylabel('Z mm', color='white')

# -- G: 3D view of one ego mesh
ax7 = fig.add_subplot(3, 4, 7, projection='3d')
ax7.set_facecolor('#1a1a1a')
ax7.set_title('Ego Mesh (frame 100, R)\nCamera-space coords', color='white', fontsize=9)
# Subsample faces for speed
step = max(1, len(ego_r.faces) // 1000)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
faces_sub = np.array(ego_r.faces)[::step]
tri = np.array(ego_r.vertices)[faces_sub]
poly = Poly3DCollection(tri, alpha=0.6, linewidth=0)
poly.set_facecolor('#4499cc'); poly.set_edgecolor('none')
ax7.add_collection3d(poly)
ax7.set_xlim(ev[:,0].min(), ev[:,0].max())
ax7.set_ylim(ev[:,1].min(), ev[:,1].max())
ax7.set_zlim(ev[:,2].min(), ev[:,2].max())
ax7.tick_params(colors='white', labelsize=6)
ax7.set_xlabel('X', color='white', fontsize=7)
ax7.set_ylabel('Y', color='white', fontsize=7)
ax7.set_zlabel('Z', color='white', fontsize=7)

# -- H: 3D view of Korean template (for comparison)
ax8 = fig.add_subplot(3, 4, 8, projection='3d')
ax8.set_facecolor('#1a1a1a')
ax8.set_title('Korean Template\nWrist-centred coords', color='white', fontsize=9)
step_k = max(1, len(kor.faces) // 1000)
faces_k = np.array(kor.faces)[::step_k]
tri_k = np.array(kor.vertices)[faces_k]
poly_k = Poly3DCollection(tri_k, alpha=0.6, linewidth=0)
poly_k.set_facecolor('#55cc55'); poly_k.set_edgecolor('none')
ax8.add_collection3d(poly_k)
ax8.set_xlim(kv[:,0].min(), kv[:,0].max())
ax8.set_ylim(kv[:,1].min(), kv[:,1].max())
ax8.set_zlim(kv[:,2].min(), kv[:,2].max())
ax8.tick_params(colors='white', labelsize=6)
ax8.set_xlabel('X', color='white', fontsize=7)
ax8.set_ylabel('Y', color='white', fontsize=7)
ax8.set_zlabel('Z', color='white', fontsize=7)

# -- I: NN registration distance histogram
ax9 = fig.add_subplot(3, 4, 9)
dark_ax(ax9, 'NN Distance: Korean→Ego\n(scale-normalised)')
ax9.hist(dists, bins=40, color='#9467bd', edgecolor='none', alpha=0.85)
ax9.axvline(np.median(dists), color='#ffaa44', linewidth=1.5,
            label=f'median={np.median(dists):.4f}')
ax9.set_xlabel('NN distance (normalised)', color='white')
ax9.set_ylabel('Korean vertices', color='white')
ax9.legend(fontsize=7, labelcolor='white', facecolor='#333333')

# -- J: Summary stats table
ax10 = fig.add_subplot(3, 4, 10)
ax10.axis('off')
ax10.set_facecolor('#1a1a1a')
rows = [
    ['Frames (R meshes)', str(len(obj_r))],
    ['Ego mesh verts', str(N_EGO)],
    ['Ego mesh faces', str(face_counts_r[0])],
    ['Korean template verts', str(N_KOR)],
    ['Topology consistent', str(topology_consistent)],
    ['Z depth range (mm)', f'{z_vals.min():.0f} to {z_vals.max():.0f}'],
    ['Hand size (px, mean)', f'{hand_size_px.mean():.0f}'],
    ['NN median dist', f'{np.median(dists):.4f}'],
    ['Unique sessions', str(len(session_ids))],
    ['Pose class', sorted(pose_classes)[0] if pose_classes else 'N/A'],
    ['Action', sorted(category_names)[0] if category_names else 'N/A'],
    ['Actor sex/age', f'{sorted(set(actor_sexes))[0]}, {sorted(set(actor_ages))[0]}'],
]
for i, (k, v) in enumerate(rows):
    y = 0.97 - i * 0.08
    ax10.text(0.02, y, k+':', color='#aaaaaa', fontsize=8, transform=ax10.transAxes, va='top')
    ax10.text(0.55, y, v, color='white', fontsize=8, transform=ax10.transAxes, va='top')

# -- K: Pose diversity — wrist-relative normalised 2D scatter per finger
ax11 = fig.add_subplot(3, 4, 11)
dark_ax(ax11, 'Pose Diversity\n(fingertip scatter, right hand)')
colors = ['#4499cc','#55cc55','#ee5555','#ffaa44','#9467bd']
tips = [4, 8, 12, 16, 20]  # thumb,index,middle,ring,pinky tips in 21-joint order
for ci, (tip, col) in enumerate(zip(tips, colors)):
    ax11.scatter(kp_norm[:, tip, 0], kp_norm[:, tip, 1],
                 s=10, alpha=0.5, color=col,
                 label=['thumb','index','middle','ring','pinky'][ci])
ax11.invert_yaxis()
ax11.set_aspect('equal')
ax11.legend(fontsize=6, labelcolor='white', facecolor='#333333', ncol=2)

# -- L: File size distribution
ax12 = fig.add_subplot(3, 4, 12)
dark_ax(ax12, 'OBJ File Sizes\n(right hand, KB)')
sizes = [z.getinfo(f).file_size / 1024 for f in obj_r]
ax12.hist(sizes, bins=20, color='#d62728', edgecolor='none', alpha=0.85)
ax12.axvline(np.mean(sizes), color='#ffaa44', linewidth=1.5,
             label=f'mean={np.mean(sizes):.0f}KB')
ax12.set_xlabel('File size (KB)', color='white')
ax12.legend(fontsize=7, labelcolor='white', facecolor='#333333')

fig.suptitle('EDA — Ego-Vision Hand Dataset  (ego_data.zip)',
             color='white', fontsize=14, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUTPUT_PNG, dpi=130, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f'\nSaved → {OUTPUT_PNG}')
z.close()
