"""
fit_mano_to_grasping.py
=======================
Offline MANO pose fitting for the grasping dataset.

For each sample, optimizes 48 MANO pose params so that the Korean template's
LBS joints (orthographically projected to XY) match the GT 2D keypoints.

Output:
  data/grasping_landmarks.npy   — (N, 42) landmarks in FreiHAND format
  data/grasping_pose_params.npy — (N, 48) fitted MANO pose params
  data/grasping_fit_errors.npy  — (N,)   final reprojection error per sample

Usage:
  my_venv/bin/python scripts/fit_mano_to_grasping.py
  my_venv/bin/python scripts/fit_mano_to_grasping.py --n-iters 200 --lr 0.02
"""

import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = '/home/user/Documents/Handpose_project'
GRASPING_DIR   = '/media/user/My Passport/Sample'
MANO_PKL       = os.path.join(PROJECT_ROOT, 'models', 'MANO_RIGHT.pkl')
KOREAN_SKEL    = os.path.join(PROJECT_ROOT, 'models', 'korean_hand_skeleton.json')
KOREAN_TMPL    = os.path.join(PROJECT_ROOT, 'meshes', 'AVERAGE_KOREAN_HAND_CENTERED.obj')
OUT_LANDMARKS  = os.path.join(PROJECT_ROOT, 'data', 'grasping_landmarks.npy')
OUT_POSES      = os.path.join(PROJECT_ROOT, 'data', 'grasping_pose_params.npy')
OUT_ERRORS     = os.path.join(PROJECT_ROOT, 'data', 'grasping_fit_errors.npy')

SCALE = 100.0   # mm → dm

# MANO 16-joint kintree order → MediaPipe 21-joint index
# (same mapping as train_unified_model.py)
MANO_TO_MP = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 2, 3, 4]

JOINT_NAMES_16 = [
    'wrist',
    'index1', 'index2', 'index3',
    'middle1','middle2','middle3',
    'pinky1', 'pinky2', 'pinky3',
    'ring1',  'ring2',  'ring3',
    'thumb1', 'thumb2', 'thumb3',
]

# ---------------------------------------------------------------------------
# Rodrigues rotation (batch)
# ---------------------------------------------------------------------------
def rodrigues_batch(rvecs):
    theta = rvecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = rvecs / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([ z,  -kz,  ky], dim=-1),
        torch.stack([ kz,  z,  -kx], dim=-1),
        torch.stack([-ky,  kx,  z ], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=rvecs.device, dtype=rvecs.dtype).view(1, 1, 3, 3)
    s = theta.unsqueeze(-1).sin()
    c = theta.unsqueeze(-1).cos()
    return I + s * K + (1 - c) * (K @ K)


# ---------------------------------------------------------------------------
# LBS forward (same as train script)
# ---------------------------------------------------------------------------
def lbs_forward(v_rest, pose_params, J, weights, parents):
    """
    v_rest      : (B, V, 3) dm
    pose_params : (B, 48)
    J           : (16, 3)  dm
    weights     : (V, 16)
    parents     : list[int]
    Returns joints_posed (B, 16, 3) dm
    """
    B = v_rest.shape[0]
    device = v_rest.device
    rvecs = pose_params.view(B, 16, 3)
    Rs = rodrigues_batch(rvecs)

    G_list = []
    for j in range(16):
        p = parents[j]
        T = torch.zeros(B, 4, 4, device=device, dtype=v_rest.dtype)
        T[:, :3, :3] = Rs[:, j]
        T[:, :3,  3] = J[j] if p < 0 else (J[j] - J[p])
        T[:, 3,   3] = 1.0
        G_list.append(T if p < 0 else G_list[p] @ T)

    G = torch.stack(G_list, dim=1)  # (B, 16, 4, 4)
    J_h = torch.cat([J, torch.ones(16, 1, device=device, dtype=v_rest.dtype)], dim=-1)
    joints_posed = torch.einsum('bjkl,jl->bjk', G, J_h)[:, :, :3]  # (B, 16, 3) dm
    return joints_posed


# ---------------------------------------------------------------------------
# Load LBS data
# ---------------------------------------------------------------------------
def load_lbs_data():
    import trimesh

    with open(MANO_PKL, 'rb') as f:
        mano = pickle.load(f, encoding='latin1')
    kintree = mano['kintree_table']
    parents = [int(kintree[0, j]) for j in range(16)]
    parents[0] = -1

    with open(KOREAN_SKEL) as f:
        skel = json.load(f)
    J_np = np.array([skel[n]['position_mm'] for n in JOINT_NAMES_16], dtype=np.float32)

    mesh = trimesh.load(KOREAN_TMPL, process=False)
    verts = np.array(mesh.vertices, dtype=np.float32)  # (11279, 3) mm

    return J_np, verts, parents


def compute_bone_weights(verts, J, parents, sigma=15.0):
    def seg_dist(v, a, b):
        ab = b - a
        t = np.clip(((v - a) @ ab) / (np.dot(ab, ab) + 1e-8), 0, 1)
        return np.linalg.norm(v - (a + t[:, None] * ab), axis=1)

    W = np.zeros((len(verts), 16), dtype=np.float64)
    for j, p in enumerate(parents):
        if p < 0:
            d = np.linalg.norm(verts - J[j], axis=1)
        else:
            d = seg_dist(verts, J[p], J[j])
        W[:, j] = np.exp(-d ** 2 / (2 * sigma ** 2))
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-8)
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Load all grasping samples
# ---------------------------------------------------------------------------
def load_grasping_samples(root_dir):
    json_files = sorted(glob.glob(
        os.path.join(root_dir, '02.라벨링데이터', '**', '*.json'), recursive=True
    ))
    print(f'[Grasping] Found {len(json_files)} JSON files.')

    landmarks_list = []
    skipped = 0
    for path in json_files:
        try:
            with open(path) as f:
                d = json.load(f)
            kp2d_flat = d['gesture']['hand_gesture_data']['hand_keypoints']['2D']
            kp2d = np.array(kp2d_flat, dtype=np.float32).reshape(21, 2)

            # FreiHAND format: wrist-relative, normalized by wrist→middle DIP (joint 11)
            kp2d -= kp2d[0:1, :]
            scale = np.linalg.norm(kp2d[11]) + 1e-6
            kp2d /= scale

            landmarks_list.append(kp2d.reshape(-1))  # (42,)
        except Exception:
            skipped += 1

    print(f'[Grasping] Loaded {len(landmarks_list)} samples ({skipped} skipped).')
    return np.stack(landmarks_list).astype(np.float32)  # (N, 42)


# ---------------------------------------------------------------------------
# Batch MANO fitting
# ---------------------------------------------------------------------------
def fit_batch(landmarks_batch, v_rest_dm, J_dm, weights_t, parents,
              device, n_iters=150, lr=0.01):
    """
    landmarks_batch : (B, 42) wrist-relative, scale-normalised
    Returns (pose_params (B, 48), errors (B,))
    """
    B = landmarks_batch.shape[0]
    gt_kp2d = torch.from_numpy(landmarks_batch).to(device)           # (B, 42)
    gt_2d   = gt_kp2d.view(B, 21, 2)[:, MANO_TO_MP, :]              # (B, 16, 2)

    pose = torch.zeros(B, 48, device=device, requires_grad=True)
    opt  = torch.optim.Adam([pose], lr=lr)

    v_batch = v_rest_dm.unsqueeze(0).expand(B, -1, -1)               # (B, V, 3) dm

    for it in range(n_iters):
        joints_dm = lbs_forward(v_batch, pose, J_dm, weights_t, parents)  # (B,16,3) dm
        joints_mm = joints_dm * SCALE                                       # (B,16,3) mm

        # Orthographic XY projection
        pred_2d = joints_mm[:, :, :2]                                      # (B,16,2)

        # Wrist-relative + normalise by wrist→middle DIP (MANO joint 6)
        pred_rel = pred_2d - pred_2d[:, :1, :]
        p_sc     = pred_rel[:, 6:7, :].norm(dim=-1, keepdim=True).clamp(min=1.0)
        pred_n   = pred_rel / p_sc                                         # (B,16,2)

        loss = F.mse_loss(pred_n, gt_2d)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Final errors per sample
    with torch.no_grad():
        joints_dm = lbs_forward(v_batch, pose, J_dm, weights_t, parents)
        joints_mm = joints_dm * SCALE
        pred_2d   = joints_mm[:, :, :2]
        pred_rel  = pred_2d - pred_2d[:, :1, :]
        p_sc      = pred_rel[:, 6:7, :].norm(dim=-1, keepdim=True).clamp(min=1.0)
        pred_n    = pred_rel / p_sc
        per_sample_err = ((pred_n - gt_2d) ** 2).mean(dim=[1, 2])   # (B,)

    return pose.detach().cpu().numpy(), per_sample_err.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Fit] Using device: {device}')

    # Load LBS data
    print('[Fit] Loading LBS data ...')
    J_np, verts_mm, parents = load_lbs_data()
    weights_np = compute_bone_weights(verts_mm, J_np, parents, sigma=15.0)

    J_dm       = torch.from_numpy(J_np / SCALE).float().to(device)         # (16,3)
    v_rest_dm  = torch.from_numpy(verts_mm / SCALE).float().to(device)     # (V,3)
    weights_t  = torch.from_numpy(weights_np).float().to(device)           # (V,16)

    # Load grasping landmarks
    landmarks = load_grasping_samples(GRASPING_DIR)   # (N, 42)
    N = len(landmarks)
    print(f'[Fit] Fitting {N} samples | iters={args.n_iters} lr={args.lr} batch={args.batch_size}')

    all_poses  = np.zeros((N, 48), dtype=np.float32)
    all_errors = np.zeros(N,       dtype=np.float32)

    n_batches = (N + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        s = bi * args.batch_size
        e = min(s + args.batch_size, N)
        batch = landmarks[s:e]

        poses, errs = fit_batch(
            batch, v_rest_dm, J_dm, weights_t, parents,
            device, n_iters=args.n_iters, lr=args.lr
        )
        all_poses[s:e]  = poses
        all_errors[s:e] = errs

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            pct = (bi + 1) / n_batches * 100
            print(f'  [{bi+1:4d}/{n_batches}] {pct:5.1f}%  '
                  f'mean_err={all_errors[:e].mean():.4f}  '
                  f'median_err={np.median(all_errors[:e]):.4f}')

    # Filter quality: keep samples with error below threshold
    threshold = np.percentile(all_errors, 90)   # top 90% quality
    good = all_errors < threshold
    print(f'\n[Fit] Error stats: mean={all_errors.mean():.4f}  '
          f'median={np.median(all_errors):.4f}  '
          f'90th-pct={threshold:.4f}')
    print(f'[Fit] Good samples: {good.sum()}/{N} ({good.mean()*100:.1f}%)')

    np.save(OUT_LANDMARKS, landmarks)
    np.save(OUT_POSES,     all_poses)
    np.save(OUT_ERRORS,    all_errors)
    print(f'[Fit] Saved:\n  {OUT_LANDMARKS}\n  {OUT_POSES}\n  {OUT_ERRORS}')

    # Quick sanity: how different are fitted poses from zeros?
    print(f'\n[Fit] Pose param stats:')
    print(f'  mean abs value : {np.abs(all_poses).mean():.4f}')
    print(f'  max abs value  : {np.abs(all_poses).max():.4f}')
    print(f'  std            : {all_poses.std():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-iters',    type=int,   default=150,   help='optimizer iterations per sample')
    parser.add_argument('--lr',         type=float, default=0.01,  help='Adam learning rate')
    parser.add_argument('--batch-size', type=int,   default=256,   help='samples per GPU batch')
    args = parser.parse_args()
    main(args)
