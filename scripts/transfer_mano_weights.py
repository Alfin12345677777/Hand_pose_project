"""
transfer_mano_weights.py
========================
Final approach: Barycentric MANO weight transfer + original Korean joints.

Key insight: joints and weights are SEPARATE concerns.
  - Joints: use the original korean_hand_skeleton.json (placed on the actual mesh)
  - Weights: transfer from MANO via barycentric surface projection (gives smooth
    weight fields that respect MANO's hand-painted anatomy)

The previous attempts failed because we either:
  1. Used wrong joints (Procrustes-aligned → compressed bone lengths)
  2. Used wrong weights (KNN too coarse, geodesic too aggressive)
  3. Changed both at once (impossible to debug)

This version:
  - Aligns MANO to Korean using per-FINGER Procrustes (not global!)
    so each finger maps correctly despite different proportions
  - Barycentric interpolation on MANO surface for smooth weights
  - Original skeleton joints for LBS (they're in the right place on the mesh)
"""

import os
import sys
import json
import pickle
import types
import numpy as np
import trimesh
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Chumpy stub
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'chumpy'))
try:
    import chumpy
except Exception:
    class _FakeObj:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, name): return _FakeObj()
    for sub in ['chumpy', 'chumpy.ch', 'chumpy.reordering', 'chumpy.ch_ops']:
        mod = types.ModuleType(sub)
        for attr in ['Select', 'MatVecMult', 'Ch', 'depends_on', 'array']:
            setattr(mod, attr, _FakeObj)
        sys.modules[sub] = mod

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = _PROJECT_ROOT
MESHES_DIR   = os.path.join(PROJECT_ROOT, 'meshes')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')

KOREAN_TEMPLATE  = os.path.join(MESHES_DIR, 'AVERAGE_KOREAN_HAND_CENTERED.obj')
MANO_PKL         = os.path.join(MODELS_DIR, 'MANO_RIGHT.pkl')
SKELETON_JSON    = os.path.join(MODELS_DIR, 'korean_hand_skeleton.json')
OUTPUT_WEIGHTS   = os.path.join(DATA_DIR, 'mano_transferred_weights.npy')

JOINT_NAMES_16 = [
    'wrist',
    'index1',  'index2',  'index3',
    'middle1', 'middle2', 'middle3',
    'pinky1',  'pinky2',  'pinky3',
    'ring1',   'ring2',   'ring3',
    'thumb1',  'thumb2',  'thumb3',
]

# MANO kintree: finger → joint indices
FINGER_JOINTS = {
    'index':  [1, 2, 3],
    'middle': [4, 5, 6],
    'pinky':  [7, 8, 9],
    'ring':   [10, 11, 12],
    'thumb':  [13, 14, 15],
}


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def procrustes_align(source, target):
    """Umeyama similarity: target ≈ s * R @ source + t"""
    n = source.shape[0]
    mu_s, mu_t = source.mean(0), target.mean(0)
    sc, tc = source - mu_s, target - mu_t
    var_s = np.mean(np.sum(sc ** 2, axis=1))
    K = (tc.T @ sc) / n
    U, sigma, Vt = np.linalg.svd(K)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, float(np.sign(d))])
    R = U @ D @ Vt
    s = float(np.sum(sigma * np.diag(D)) / var_s)
    t = mu_t - s * (R @ mu_s)
    return s, R, t


def align_mano_per_finger(mano_verts, mano_weights, J_mano, J_korean, parents):
    """
    Align MANO to Korean space using per-finger transforms.

    1. Global Procrustes on all 16 joints for the palm/wrist
    2. Per-finger affine for each of the 5 fingers
    3. Each MANO vertex is warped by a weighted blend of the global
       and its dominant finger's transform

    This handles different finger proportions between MANO and Korean.
    """
    # Global alignment
    s_g, R_g, t_g = procrustes_align(J_mano, J_korean)
    global_xform = lambda v: s_g * (v @ R_g.T) + t_g

    # Per-finger alignment: use wrist + 3 finger joints as correspondences
    finger_xforms = {}
    for fname, jids in FINGER_JOINTS.items():
        src_pts = np.array([J_mano[0]] + [J_mano[j] for j in jids])  # 4 points
        tgt_pts = np.array([J_korean[0]] + [J_korean[j] for j in jids])
        s_f, R_f, t_f = procrustes_align(src_pts, tgt_pts)
        finger_xforms[fname] = (s_f, R_f, t_f)

    # For each MANO vertex, determine which finger it belongs to
    # (based on which finger joints have the most weight)
    n = len(mano_verts)
    aligned = np.zeros_like(mano_verts)

    for i in range(n):
        w = mano_weights[i]

        # Find dominant finger
        best_finger = None
        best_w = 0
        for fname, jids in FINGER_JOINTS.items():
            fw = sum(w[j] for j in jids)
            if fw > best_w:
                best_w = fw
                best_finger = fname

        wrist_w = w[0]

        if best_finger is not None and best_w > 0.1:
            s_f, R_f, t_f = finger_xforms[best_finger]
            finger_pos = s_f * (mano_verts[i] @ R_f.T) + t_f
            global_pos = global_xform(mano_verts[i])

            # Blend: more finger weight → more finger transform
            alpha = min(best_w / (best_w + wrist_w + 1e-8), 1.0)
            aligned[i] = alpha * finger_pos + (1 - alpha) * global_pos
        else:
            aligned[i] = global_xform(mano_verts[i])

    return aligned


# ---------------------------------------------------------------------------
# Barycentric surface projection
# ---------------------------------------------------------------------------

def closest_point_on_triangle(p, v0, v1, v2):
    """Closest point on triangle, returns (point, barycentric_coords)."""
    ab, ac, ap = v1 - v0, v2 - v0, p - v0
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return v0, np.array([1., 0., 0.])

    bp = p - v1
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return v1, np.array([0., 1., 0.])

    cp = p - v2
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return v2, np.array([0., 0., 1.])

    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1/(d1-d3)
        return v0 + v*ab, np.array([1-v, v, 0.])

    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2/(d2-d6)
        return v0 + w*ac, np.array([1-w, 0., w])

    va = d3*d6 - d5*d4
    if va <= 0 and (d4-d3) >= 0 and (d5-d6) >= 0:
        w = (d4-d3)/((d4-d3)+(d5-d6))
        return v1 + w*(v2-v1), np.array([0., 1-w, w])

    denom = 1.0/(va+vb+vc)
    v, w = vb*denom, vc*denom
    return v0 + v*ab + w*ac, np.array([1-v-w, v, w])


def project_and_interpolate(query_pts, mesh_verts, mesh_faces, mesh_weights):
    """
    Project query points onto mesh surface, barycentric-interpolate weights.
    """
    N = len(query_pts)
    tri_verts = mesh_verts[mesh_faces]
    centroids = tri_verts.mean(axis=1)
    tree = cKDTree(centroids)
    K = min(32, len(mesh_faces))
    _, candidates = tree.query(query_pts, k=K)

    result_weights = np.zeros((N, mesh_weights.shape[1]), dtype=np.float64)
    result_dists = np.zeros(N)

    for i in range(N):
        best_d, best_b, best_f = 1e10, None, 0
        for fi in candidates[i]:
            f = mesh_faces[fi]
            cp, bary = closest_point_on_triangle(
                query_pts[i], mesh_verts[f[0]], mesh_verts[f[1]], mesh_verts[f[2]])
            d = np.linalg.norm(query_pts[i] - cp)
            if d < best_d:
                best_d, best_b, best_f = d, bary, fi

        f = mesh_faces[best_f]
        result_weights[i] = (best_b[0] * mesh_weights[f[0]] +
                              best_b[1] * mesh_weights[f[1]] +
                              best_b[2] * mesh_weights[f[2]])
        result_dists[i] = best_d

    # Normalize
    result_weights /= result_weights.sum(axis=1, keepdims=True).clip(min=1e-8)
    return result_weights, result_dists


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_weights(weights, faces, n_iters=3, lam=0.15):
    n_verts = len(weights)
    adj = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])
    adj = [list(a) for a in adj]
    max_nn = max(len(a) for a in adj)
    nn_idx = np.zeros((n_verts, max_nn), dtype=np.int32)
    nn_mask = np.zeros((n_verts, max_nn), dtype=bool)
    for v in range(n_verts):
        nn = adj[v]
        nn_idx[v, :len(nn)] = nn
        nn_mask[v, :len(nn)] = True
    W = weights.copy()
    for _ in range(n_iters):
        g = W[nn_idx]; g[~nn_mask] = 0.0
        c = nn_mask.sum(axis=1, keepdims=True).clip(min=1)
        W = (1-lam)*W + lam*(g.sum(axis=1)/c)
        W /= W.sum(axis=1, keepdims=True).clip(min=1e-8)
    return W


# ---------------------------------------------------------------------------
# LBS
# ---------------------------------------------------------------------------

def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-8: return np.eye(3)
    k = r/theta
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

def lbs_pose(v_rest, pose, J, W, parents):
    Rs = np.stack([rodrigues(pose[j*3:(j+1)*3]) for j in range(16)])
    G = [None]*16
    for j in range(16):
        p = parents[j]
        T = np.eye(4); T[:3,:3]=Rs[j]; T[:3,3]=J[j] if p<0 else J[j]-J[p]
        G[j] = T if p<0 else G[p]@T
    G = np.stack(G); Gf = G.copy()
    for j in range(16):
        Gf[j,:3,3] = G[j,:3,:3]@(-J[j]) + G[j,:3,3]
    Tb = np.einsum('vj,jab->vab', W, Gf)
    vh = np.c_[v_rest, np.ones(len(v_rest))]
    return np.einsum('vab,vb->va', Tb, vh)[:,:3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────
    print('[1] Loading ...')
    korean_mesh = trimesh.load(KOREAN_TEMPLATE, process=False)
    korean_verts = np.array(korean_mesh.vertices, dtype=np.float64)
    korean_faces = np.array(korean_mesh.faces, dtype=np.int32)

    with open(MANO_PKL, 'rb') as f:
        mano = pickle.load(f, encoding='latin1')
    mano_rest = np.array(mano['v_template'], dtype=np.float64)
    mano_weights = np.array(mano['weights'], dtype=np.float64)
    mano_faces = np.array(mano['f'], dtype=np.int32)
    J_reg = np.array(mano['J_regressor'].todense(), dtype=np.float64)
    J_mano = J_reg @ mano_rest
    kintree = np.array(mano['kintree_table'], dtype=np.int64)
    parents = list(kintree[0].copy()); parents[0] = -1

    with open(SKELETON_JSON) as f:
        skel = json.load(f)
    J_korean = np.array([skel[n]['position_mm'] for n in JOINT_NAMES_16],
                        dtype=np.float64)

    print(f'    Korean: {korean_verts.shape}, MANO: {mano_rest.shape}')

    # ── 2. Per-finger alignment ───────────────────────────────────────────
    print('\n[2] Per-finger MANO alignment ...')
    mano_aligned = align_mano_per_finger(
        mano_rest, mano_weights, J_mano, J_korean, parents)

    # Check alignment quality
    tree = cKDTree(mano_aligned)
    dists_check, _ = tree.query(korean_verts)
    print(f'    Surface distance: mean={dists_check.mean():.1f}mm, '
          f'median={np.median(dists_check):.1f}mm')

    # Save aligned mesh
    aligned_obj = os.path.join(MESHES_DIR, 'MANO_ALIGNED_TO_KOREA.obj')
    trimesh.Trimesh(vertices=mano_aligned, faces=mano_faces,
                    process=False).export(aligned_obj)

    # ── 3. Barycentric weight transfer ────────────────────────────────────
    print('\n[3] Barycentric surface projection ...')
    transferred, proj_dists = project_and_interpolate(
        korean_verts, mano_aligned, mano_faces, mano_weights)

    print(f'    Projection dist: mean={proj_dists.mean():.1f}mm, '
          f'median={np.median(proj_dists):.1f}mm, max={proj_dists.max():.1f}mm')
    hard = (transferred.max(axis=1) > 0.95).sum()
    print(f'    Hard (>0.95): {hard}/{len(transferred)} ({100*hard/len(transferred):.1f}%)')

    # ── 4. Smoothing ──────────────────────────────────────────────────────
    print('\n[4] Smoothing ...')
    W = smooth_weights(transferred, korean_faces, n_iters=3, lam=0.15)

    hard_s = (W.max(axis=1) > 0.95).sum()
    print(f'    Hard (>0.95): {hard_s}/{len(W)} ({100*hard_s/len(W):.1f}%)')

    dom = W.argmax(axis=1)
    print(f'\n    Joint distribution:')
    for j, name in enumerate(JOINT_NAMES_16):
        cnt = int((dom == j).sum())
        if cnt > 0:
            print(f'      J{j:02d} {name:<12}: {cnt:5d} ({100*cnt/len(dom):.1f}%)')

    # ── 5. Save ───────────────────────────────────────────────────────────
    print(f'\n[5] Saving → {OUTPUT_WEIGHTS}')
    np.save(OUTPUT_WEIGHTS, W.astype(np.float32))

    # ── 6. Diagnostics ────────────────────────────────────────────────────
    print('\n[6] Diagnostic meshes (using ORIGINAL Korean joints for LBS) ...')
    poses = {
        'rest': np.zeros(48),
        'bend_index': _make_bend_index(),
        'fist': _make_fist(),
        'spread': _make_spread(),
    }
    for name, pose in poses.items():
        posed = lbs_pose(korean_verts, pose, J_korean, W, parents)
        mesh = trimesh.Trimesh(vertices=posed, faces=korean_faces, process=False)
        out = os.path.join(OUTPUTS_DIR, f'transfer_diag_{name}.obj')
        mesh.export(out)
        print(f'    {out}')

    # MANO reference with per-finger aligned joints
    s_g, R_g, t_g = procrustes_align(J_mano, J_korean)
    J_mano_g = s_g * (J_mano @ R_g.T) + t_g
    mano_global = s_g * (mano_rest @ R_g.T) + t_g
    for name, pose in poses.items():
        if name == 'rest': continue
        posed = lbs_pose(mano_global, pose, J_mano_g, mano_weights, parents)
        mesh = trimesh.Trimesh(vertices=posed, faces=mano_faces, process=False)
        out = os.path.join(OUTPUTS_DIR, f'transfer_ref_mano_{name}.obj')
        mesh.export(out)
        print(f'    {out} (MANO ref)')

    print('\nDone!')


def _make_bend_index():
    p = np.zeros(48)
    p[1*3]=1.5; p[2*3]=1.2; p[3*3]=0.8
    return p

def _make_fist():
    p = np.zeros(48)
    for j in range(1,16):
        if j in [13,14,15]:
            p[j*3+2]=1.0; p[j*3]=0.3
        else:
            p[j*3]=1.3
    return p

def _make_spread():
    p = np.zeros(48)
    p[1*3+2]=0.3; p[4*3+2]=0.1; p[7*3+2]=-0.3
    p[10*3+2]=-0.1; p[13*3+2]=0.5
    return p


if __name__ == '__main__':
    main()
