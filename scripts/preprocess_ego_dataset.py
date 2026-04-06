"""
preprocess_ego_dataset.py
=========================
One-time preprocessing: registers ego-vision meshes (ego_data.zip) to Korean
template topology and extracts normalised 2D landmarks.

For each frame:
  1. Load right-hand mesh_R.obj (7180 verts, camera-space mm)
  2. Per-frame alignment: centre → flip Y → uniform scale → fixed ICP rotation
     (ICP rotation accounts for the axis convention difference between ego
      camera-space and Korean template space; computed once from a reference frame)
  3. NN-map 7180 ego verts → 11279 Korean verts (per-frame cKDTree query)
  4. Extract 2D landmarks from the JSON annotation (MediaPipe 21-joint format)
     and normalise to FreiHAND format (wrist-relative, scale by wrist→middleDIP)

Output (saved to data/):
  ego_landmarks.npy    (N, 42)         float32 — normalised 2D landmarks
  ego_posed_verts.npy  (N, 11279, 3)   float32 — Korean-topology posed verts mm

Usage:
  my_venv/bin/python scripts/preprocess_ego_dataset.py
"""

import os, sys, json, zipfile, ast, io
import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = '/home/user/Documents/Handpose_project'
ZIP_PATH     = '/media/user/My Passport/Sample/ego_data.zip'
KOREAN_TMPL  = os.path.join(PROJECT_ROOT, 'meshes', 'AVERAGE_KOREAN_HAND_CENTERED.obj')
OUT_LM       = os.path.join(PROJECT_ROOT, 'data', 'ego_landmarks.npy')
OUT_VERTS    = os.path.join(PROJECT_ROOT, 'data', 'ego_posed_verts.npy')

os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)


def load_obj_from_bytes(raw_bytes):
    """Parse OBJ from raw bytes, return (V,3) float32 vertex array."""
    text = raw_bytes.decode('utf-8', errors='ignore')
    verts = []
    for line in text.splitlines():
        if line.startswith('v '):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def align_ego_to_korean(ev, s, R_icp):
    """
    Per-frame alignment of ego mesh vertices to Korean template space.
    ev    : (V, 3) float64 raw ego verts (camera-space mm)
    s     : float  uniform scale factor (computed from reference frame)
    R_icp : (3, 3) fixed rotation (computed from reference frame ICP)
    Returns (V, 3) float32 aligned verts.
    """
    ev_c    = ev - ev.mean(0)                        # centre at centroid
    ev_flip = ev_c * np.array([1.0, -1.0, 1.0])     # flip Y (image↔world)
    ev_norm = ev_flip * s                            # scale to Korean size
    ev_rot  = (R_icp @ ev_norm.T).T                 # fixed rotation correction
    return ev_rot.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Load Korean template
# ---------------------------------------------------------------------------
print('[Pre] Loading Korean template ...')
kor = trimesh.load(KOREAN_TMPL, process=False)
kv  = np.array(kor.vertices, dtype=np.float32)   # (11279, 3) mm
print(f'[Pre] Korean template: {len(kv)} verts')

# ---------------------------------------------------------------------------
# 2. Inventory zip
# ---------------------------------------------------------------------------
print('[Pre] Scanning zip ...')
zf = zipfile.ZipFile(ZIP_PATH)
all_names  = zf.namelist()
obj_r_list = sorted([f for f in all_names if f.endswith('_mesh_R.obj')])
json_list  = sorted([f for f in all_names if f.endswith('_rgb.json')])

# frame_id → json_path lookup
json_lookup = {}
for jp in json_list:
    fid = jp.split('/')[-1].replace('_rgb.json', '')
    json_lookup[fid] = jp

print(f'[Pre] {len(obj_r_list)} right-hand meshes, {len(json_list)} JSONs')

# ---------------------------------------------------------------------------
# 3. Compute fixed registration from reference frame
#    Reference = first frame; computes scale s and ICP rotation R_icp
# ---------------------------------------------------------------------------
print('[Pre] Computing registration from reference frame ...')
ref_raw  = zf.read(obj_r_list[0])
ref_v    = load_obj_from_bytes(ref_raw).astype(np.float64)

# Initial alignment
ref_c    = ref_v - ref_v.mean(0)
ref_flip = ref_c * np.array([1.0, -1.0, 1.0])
s        = float(kv[:, 1].ptp() / ref_flip[:, 1].ptp())
ref_norm = ref_flip * s

# ICP for residual rotation (rotation only — translation is handled per-frame by centring)
try:
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(ref_norm)
    tgt.points = o3d.utility.Vector3dVector(kv.astype(np.float64))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=30.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    R_icp = result.transformation[:3, :3].astype(np.float64)
    print(f'[Pre] ICP fitness={result.fitness:.4f}  RMSE={result.inlier_rmse:.2f} mm')
except ImportError:
    print('[Pre] open3d not found — using trimesh ICP fallback')
    matrix, _ = trimesh.registration.icp(ref_norm, kv, max_iterations=100)
    R_icp = matrix[:3, :3].astype(np.float64)

print(f'[Pre] Scale factor s = {s:.4f}')
print(f'[Pre] ICP rotation:\n{np.round(R_icp, 4)}')

# Verify on reference frame
ref_final = align_ego_to_korean(ref_v, s, R_icp)
tree_ref  = cKDTree(ref_final)
d_ref, _  = tree_ref.query(kv)
print(f'[Pre] Reference frame NN: median={np.median(d_ref):.2f}mm  '
      f'95pct={np.percentile(d_ref, 95):.2f}mm  (hand height ~{kv[:,1].ptp():.0f}mm)')

# ---------------------------------------------------------------------------
# 4. Process all frames
# ---------------------------------------------------------------------------
print(f'[Pre] Processing {len(obj_r_list)} frames ...')
landmarks_list   = []
posed_verts_list = []
skipped = 0

for i, obj_path in enumerate(obj_r_list):
    try:
        frame_id = obj_path.split('/')[-1].replace('_mesh_R.obj', '')

        if frame_id not in json_lookup:
            skipped += 1
            continue

        # ── Load and register ego mesh ──────────────────────────────────
        raw_obj = zf.read(obj_path)
        ev      = load_obj_from_bytes(raw_obj).astype(np.float64)
        ev_reg  = align_ego_to_korean(ev, s, R_icp)   # (7180, 3) mm

        # NN map: for each of the 11279 Korean verts, find nearest ego vert
        tree      = cKDTree(ev_reg)
        _, nn_idx = tree.query(kv)                     # (11279,)
        posed_korean = ev_reg[nn_idx].astype(np.float32)  # (11279, 3) mm

        # ── Extract and normalise 2D landmarks ─────────────────────────
        raw_json = zf.read(json_lookup[frame_id]).decode('utf-8')
        d        = json.loads(raw_json)
        kp_ann   = next(a for a in d['annotations'] if a['type'] == 'keypoint_b')
        kp       = ast.literal_eval(kp_ann['data'])
        kp_r     = np.array(kp[0], dtype=np.float32)   # (21, 3): [px_x, px_y, z_mm]

        kp2d  = kp_r[:, :2].copy()          # (21, 2) pixel coords
        kp2d -= kp2d[0:1]                   # wrist at origin
        scale_lm = np.linalg.norm(kp2d[11]) + 1e-6  # wrist→middle DIP (joint 11)
        kp2d /= scale_lm                    # normalised, same as FreiHAND format

        landmarks_list.append(kp2d.reshape(-1).astype(np.float32))   # (42,)
        posed_verts_list.append(posed_korean)                         # (11279, 3)

        if (i + 1) % 20 == 0 or i == len(obj_r_list) - 1:
            print(f'  [{i+1:3d}/{len(obj_r_list)}] processed')

    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f'  [skip] {obj_path.split("/")[-1]}: {e}')

zf.close()

N = len(landmarks_list)
print(f'\n[Pre] Done: {N} frames ({skipped} skipped)')

if N == 0:
    print('[Pre] ERROR: no frames processed. Check zip path.')
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
landmarks_arr   = np.stack(landmarks_list)    # (N, 42)
posed_verts_arr = np.stack(posed_verts_list)  # (N, 11279, 3)

np.save(OUT_LM,    landmarks_arr)
np.save(OUT_VERTS, posed_verts_arr)
print(f'\n[Pre] Saved:')
print(f'  {OUT_LM}     shape={landmarks_arr.shape}')
print(f'  {OUT_VERTS}  shape={posed_verts_arr.shape}  '
      f'({posed_verts_arr.nbytes / 1e6:.1f} MB)')

# ---------------------------------------------------------------------------
# 6. Sanity check
# ---------------------------------------------------------------------------
print(f'\n[Pre] Landmark stats:')
print(f'  mean={landmarks_arr.mean():.4f}  std={landmarks_arr.std():.4f}')

print(f'\n[Pre] Posed verts stats (mm):')
print(f'  mean={posed_verts_arr.mean():.2f}  std={posed_verts_arr.std():.2f}')
y_ranges = posed_verts_arr[:, :, 1].max(axis=1) - posed_verts_arr[:, :, 1].min(axis=1)
print(f'  Y extent (finger direction): mean={y_ranges.mean():.1f}mm  '
      f'(Korean template: {kv[:,1].ptp():.1f}mm)')
