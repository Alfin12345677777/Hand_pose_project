"""
train_unified_model.py
======================
Semi-supervised training script for the Unified Shape+Pose hand model.

Architecture:
  ShapeEncoder  : flat Korean mesh (11279×3) → z_shape (64-dim VAE)
  ShapeDecoder  : z_shape (64) → vertex offsets from Korean template (11279×3)
  PoseEncoder   : 2D landmarks (42) → MANO pose params (48)
  DiffLBS       : (B,V,3) rest verts + (B,48) pose → (B,V,3) posed verts
  UnifiedHandModel : wraps all four + Korean template as buffer

Outputs:
  models/unified_hand_model.pth
  outputs/unified_training.png
"""

import os
import sys
import json
import argparse
import warnings
import pickle
import time
from itertools import cycle

warnings.filterwarnings('ignore')

# Allow importing laplacian_loss from the same scripts/ directory
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from laplacian_loss import LaplacianLoss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = '/home/user/Documents/Handpose_project'
DATA_DIR      = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR    = os.path.join(PROJECT_ROOT, 'models')
MESHES_DIR    = os.path.join(PROJECT_ROOT, 'meshes')
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, 'outputs')
SCRIPTS_DIR   = os.path.join(PROJECT_ROOT, 'scripts')

KOREAN_SCAN_DIR   = os.path.join(DATA_DIR, '손 데이터')
GRASPING_DIR      = '/media/user/My Passport/Sample'
LANDMARKS_NPY     = os.path.join(DATA_DIR, 'landmarks_2d.npy')
POSE_PARAMS_NPY   = os.path.join(DATA_DIR, 'pose_params.npy')
GRASP_LANDMARKS_NPY = os.path.join(DATA_DIR, 'grasping_landmarks.npy')
GRASP_POSES_NPY     = os.path.join(DATA_DIR, 'grasping_pose_params.npy')
GRASP_ERRORS_NPY    = os.path.join(DATA_DIR, 'grasping_fit_errors.npy')
EGO_LANDMARKS_NPY   = os.path.join(DATA_DIR, 'ego_landmarks.npy')
EGO_POSED_VERTS_NPY = os.path.join(DATA_DIR, 'ego_posed_verts.npy')
MANO_WEIGHTS_NPY    = os.path.join(DATA_DIR, 'mano_transferred_weights.npy')
KOREAN_TEMPLATE   = os.path.join(MESHES_DIR, 'AVERAGE_KOREAN_HAND_CENTERED.obj')
MANO_PKL          = os.path.join(MODELS_DIR, 'MANO_RIGHT.pkl')
MANO_ALIGNED      = os.path.join(MESHES_DIR, 'MANO_ALIGNED_TO_KOREA.obj')
POSE_LIFTER_PTH   = os.path.join(MODELS_DIR, 'pose_lifter.pth')
SHAPE_CACHE       = os.path.join(DATA_DIR, 'korean_shape_cache.pt')
JOINT_LABELS_JSON = os.path.join(DATA_DIR, 'training_labels.json')
JOINT_REGRESSOR_JSON = os.path.join(PROJECT_ROOT, 'models', 'k_mano_joint_regressor.json')  # legacy
KOREAN_SKELETON_JSON = os.path.join(PROJECT_ROOT, 'models', 'korean_hand_skeleton.json')
OUTPUT_MODEL      = os.path.join(MODELS_DIR, 'unified_hand_model.pth')
OUTPUT_PLOT       = os.path.join(OUTPUTS_DIR, 'unified_training.png')

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NUM_VERTS        = 11279
SCALE            = 100.0   # mm → dm; keeps vertices in ~[-1.5, 1.5]
BATCH_SIZE       = 64
LR               = 1e-4
EPOCHS           = 200
SHAPE_DIM        = 64
LAMBDA_SHAPE     = 1.0
LAMBDA_POSE      = 1.0
LAMBDA_CROSS     = 0.3    # supplementary vertex signal; kept < LAMBDA_POSE to avoid gradient conflict
LAMBDA_JOINT     = 0.1    # increased: joint signal now supported by L_cross so can raise weight
LAMBDA_LAP       = 0.1    # raised 10× — Laplacian now scale-normalized so needs higher weight
LAMBDA_PENET     = 0.005  # penetration loss; small so it doesn't dominate early training
LAMBDA_CORR      = 0.05   # raised for scaled-up CorrectionNet (1024 hidden) — prevent correction explosion
LAMBDA_CROSS_CORR= 0.3    # vertex loss through corrected mesh — gives CorrectionNet real signal
LAMBDA_REPROJ    = 0.3    # 2D reprojection loss on grasping dataset; balanced with L_cross
LAMBDA_TIP       = 0.1    # fingertip position loss; same scale as LAMBDA_JOINT
LAMBDA_EGO_MESH  = 0.1    # reduced from 0.5 — 134 frames too small to dominate training
LM_SCALE_JITTER  = 0.15   # landmark scale jitter ±15%; aggressive to reduce overfitting
LM_NOISE_STD     = 0.03   # landmark Gaussian noise std; ~3% of hand size
SCHED_T_MAX      = 400    # 2× epochs so LR never fully collapses within a 200-epoch run
LR_POSE_UNFREEZE = 1e-5   # lower LR for pose encoder + correction net when unfreezing
BETA_MAX         = 1e-3
KLD_WARMUP_START = 10
KLD_WARMUP_END   = 40
CROSS_START      = 0      # active from epoch 0: no reason to delay dense vertex supervision
POSE_UNFREEZE    = 30     # unfreeze at epoch 30 — let shape VAE stabilize first

# ---------------------------------------------------------------------------
# Rodrigues rotation (batch)
# ---------------------------------------------------------------------------

def rodrigues_batch(rvecs):
    """
    rvecs: (B, J, 3)
    Returns rotation matrices (B, J, 3, 3).
    """
    theta = rvecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (B,J,1)
    k = rvecs / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([ z,  -kz,  ky], dim=-1),
        torch.stack([ kz,  z,  -kx], dim=-1),
        torch.stack([-ky,  kx,  z ], dim=-1),
    ], dim=-2)  # (B,J,3,3)
    I = torch.eye(3, device=rvecs.device, dtype=rvecs.dtype).view(1, 1, 3, 3).expand_as(K)
    s = theta.unsqueeze(-1).sin()
    c = theta.unsqueeze(-1).cos()
    return I + s * K + (1 - c) * (K @ K)


# ---------------------------------------------------------------------------
# Linear Blend Skinning (differentiable)
# ---------------------------------------------------------------------------

def lbs_torch(v_rest, pose_params, J, weights, parents):
    """
    v_rest      : (B, V, 3)
    pose_params : (B, 48)
    J           : (16, 3) joint positions in rest pose (dm)
    weights     : (V, 16) blend weights
    parents     : list[int], length 16; root has parent -1

    Returns:
        posed_verts : (B, V, 3)
        joints_posed: (B, 16, 3) world-space joint centres after LBS transforms
    """
    B, V, _ = v_rest.shape
    device = v_rest.device

    rvecs = pose_params.view(B, 16, 3)
    Rs = rodrigues_batch(rvecs)  # (B, 16, 3, 3)

    G_list = []
    for j in range(16):
        p = parents[j]
        T = torch.zeros(B, 4, 4, device=device, dtype=v_rest.dtype)
        T[:, :3, :3] = Rs[:, j]
        T[:, :3, 3] = J[j] if p < 0 else (J[j] - J[p])
        T[:, 3, 3] = 1.0
        G_list.append(T if p < 0 else G_list[p] @ T)

    G = torch.stack(G_list, dim=1)   # (B, 16, 4, 4)

    # World-space joint positions: G[:, j] applied to rest joint J[j]
    J_h = torch.cat([J, torch.ones(16, 1, device=device, dtype=v_rest.dtype)], dim=-1)  # (16, 4)
    joints_posed = torch.einsum('bjkl,jl->bjk', G, J_h)[:, :, :3]  # (B, 16, 3)

    Gf = G.clone()                   # MUST clone — in-place mod below would corrupt G otherwise
    for j in range(16):
        Gf[:, j, :3, 3] = (G[:, j, :3, :3] @ (-J[j]).unsqueeze(-1)).squeeze(-1) + G[:, j, :3, 3]

    T_blend = torch.einsum('vj,bjkl->bvkl', weights, Gf)   # (B, V, 4, 4)
    ones = torch.ones(B, V, 1, device=device, dtype=v_rest.dtype)
    vh = torch.cat([v_rest, ones], dim=-1)                  # (B, V, 4)
    posed_verts = torch.einsum('bvkl,bvl->bvk', T_blend, vh)[:, :, :3]  # (B, V, 3)
    return posed_verts, joints_posed


# ---------------------------------------------------------------------------
# Beta schedule (KL annealing)
# ---------------------------------------------------------------------------

def get_beta(epoch, warmup_start=KLD_WARMUP_START, warmup_end=KLD_WARMUP_END, beta_max=BETA_MAX):
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return beta_max
    return beta_max * (epoch - warmup_start) / (warmup_end - warmup_start)


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------

class ShapeEncoder(nn.Module):
    """Encodes a flat Korean mesh (V*3) into a 64-dim VAE latent."""

    def __init__(self, num_verts=NUM_VERTS, latent_dim=SHAPE_DIM):
        super().__init__()
        in_dim = num_verts * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        """x: (B, V*3) — already normalized to dm scale."""
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ShapeDecoder(nn.Module):
    """Decodes a latent vector to vertex offsets from the Korean template."""

    def __init__(self, num_verts=NUM_VERTS, latent_dim=SHAPE_DIM):
        super().__init__()
        out_dim = num_verts * 3
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
            # No final activation — offsets are signed
        )

    def forward(self, z):
        """z: (B, latent_dim) → offsets (B, V*3) in dm scale."""
        return self.net(z)


class PoseEncoder(nn.Module):
    """
    Regresses MANO pose parameters from 2D landmarks.

    Wider + deeper than original (256→256→128) with residual connections
    to better handle the 2D→3D ambiguity. The residual blocks help
    gradients flow through the deeper network without degradation.
    """

    def __init__(self, in_dim=42, out_dim=48, dropout=0.3):
        super().__init__()
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )
        # Residual block 2
        self.res2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )
        # Output head
        self.output_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        """x: (B, 42) landmarks → (B, 48) pose params."""
        h = self.input_proj(x)
        h = F.relu(h + self.res1(h), inplace=True)
        h = F.relu(h + self.res2(h), inplace=True)
        return self.output_head(h)


class CorrectionNet(nn.Module):
    """
    Learnable corrective vertex offsets on top of LBS output.
    Adapted from DeepHandMesh's SkinRefineNet (temp_deephand/common/nets/module.py).

    Two branches:
      pose_corr  : (B, 48) pose params  → (B, V, 3) offsets in dm
      shape_corr : (B, 64) shape latent → (B, V, 3) offsets in dm

    Offsets are intentionally kept small by LAMBDA_CORR regularisation.
    This fixes the 'candy-wrapper' LBS artifacts at knuckles and finger joints.
    """

    def __init__(self, num_verts=NUM_VERTS, pose_dim=48, shape_dim=SHAPE_DIM):
        super().__init__()
        self.num_verts = num_verts
        self.fc_pose = nn.Sequential(
            nn.Linear(pose_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_verts * 3),
        )
        self.fc_shape = nn.Sequential(
            nn.Linear(shape_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_verts * 3),
        )
        # Init last layers near zero — use 1e-6 for the larger 1024-unit network
        # to prevent correction explosion in early epochs
        nn.init.normal_(self.fc_pose[-1].weight,  std=1e-6)
        nn.init.zeros_(self.fc_pose[-1].bias)
        nn.init.normal_(self.fc_shape[-1].weight, std=1e-6)
        nn.init.zeros_(self.fc_shape[-1].bias)

    def forward(self, pose_params, z_shape):
        """
        pose_params : (B, 48)
        z_shape     : (B, 64)
        Returns pose_corr (B, V, 3) and shape_corr (B, V, 3) in dm.
        Corrections clamped to ±0.05 dm (±5mm) to prevent explosion.
        """
        B = pose_params.shape[0]
        pose_corr  = self.fc_pose(pose_params).view(B, self.num_verts, 3)
        shape_corr = self.fc_shape(z_shape).view(B, self.num_verts, 3)
        # Clamp to prevent unbounded corrections
        pose_corr  = torch.clamp(pose_corr,  -0.05, 0.05)
        shape_corr = torch.clamp(shape_corr, -0.05, 0.05)
        return pose_corr, shape_corr


class DiffLBS(nn.Module):
    """Differentiable Linear Blend Skinning for the Korean mesh."""

    def __init__(self, weights_kmano, J, parents):
        """
        weights_kmano : (11279, 16) numpy array
        J             : (16, 3)    numpy array — fixed joint positions
        parents       : list[int], length 16; root has parent -1
        """
        super().__init__()
        self.register_buffer('weights', torch.from_numpy(weights_kmano).float())
        self.register_buffer('J', torch.from_numpy(J).float())
        self.parents = parents  # plain list, not a buffer

    def forward(self, v_rest, pose_params):
        """
        v_rest      : (B, V, 3) in dm scale
        pose_params : (B, 48)
        Returns posed_verts (B, V, 3) dm, joints_posed (B, 16, 3) dm.
        """
        return lbs_torch(v_rest, pose_params, self.J, self.weights, self.parents)


class UnifiedHandModel(nn.Module):
    """
    Full unified model combining VAE shape branch and pose regression branch.
    Korean template is stored as a registered buffer (dm scale).
    """

    def __init__(self, template_verts, weights_kmano, J, parents,
                 latent_dim=SHAPE_DIM, scale=SCALE, joint_vert_indices=None):
        """
        template_verts      : (11279, 3) numpy array in mm
        joint_vert_indices  : (21,) numpy int array — vertex index per joint
        """
        super().__init__()
        self.scale = scale
        # Store template in dm scale
        self.register_buffer('template',
                             torch.from_numpy(template_verts).float() / scale)
        # 21 vertex indices that represent each joint on the Korean mesh
        if joint_vert_indices is not None:
            self.register_buffer(
                'joint_vert_idx',
                torch.from_numpy(joint_vert_indices).long()
            )

        self.shape_encoder  = ShapeEncoder(num_verts=NUM_VERTS, latent_dim=latent_dim)
        self.shape_decoder  = ShapeDecoder(num_verts=NUM_VERTS, latent_dim=latent_dim)
        self.pose_encoder   = PoseEncoder(in_dim=42, out_dim=48)
        self.correction_net = CorrectionNet(num_verts=NUM_VERTS, pose_dim=48, shape_dim=latent_dim)
        # J must be in the same unit as vertices (dm), so divide by scale here.
        # lbs_torch receives v in dm and J in dm — units consistent.
        self.diff_lbs       = DiffLBS(weights_kmano, J / scale, parents)

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    # Shape branch
    # ------------------------------------------------------------------

    def encode_shape(self, scan_flat):
        """
        scan_flat : (B, V*3) in mm
        Returns (mu, logvar), each (B, latent_dim).
        """
        x = scan_flat / self.scale
        return self.shape_encoder(x)

    def decode_shape(self, z):
        """
        z : (B, latent_dim)
        Returns rest vertices (B, V, 3) in mm.
        """
        offsets = self.shape_decoder(z).view(-1, NUM_VERTS, 3)  # dm scale
        return (self.template.unsqueeze(0) + offsets) * self.scale

    # ------------------------------------------------------------------
    # Pose branch
    # ------------------------------------------------------------------

    def encode_pose(self, landmarks):
        """
        landmarks : (B, 42) 2D landmarks
        Returns (B, 48) pose params.
        """
        return self.pose_encoder(landmarks)

    # ------------------------------------------------------------------
    # LBS
    # ------------------------------------------------------------------

    def pose_mesh(self, rest_verts_mm, pose_params):
        """
        rest_verts_mm : (B, V, 3) in mm
        pose_params   : (B, 48)
        Returns:
            posed_verts : (B, V, 3) in mm
            joints_mm   : (B, 16, 3) in mm — true LBS joint centres
        """
        v = rest_verts_mm / self.scale                    # to dm
        posed_dm, joints_dm = self.diff_lbs(v, pose_params)
        return posed_dm * self.scale, joints_dm * self.scale


# ---------------------------------------------------------------------------
# LBS helpers
# ---------------------------------------------------------------------------

# Joint ordering must match MANO kintree_table row order
_JOINT_NAMES_16 = [
    'wrist',
    'index1',  'index2',  'index3',
    'middle1', 'middle2', 'middle3',
    'pinky1',  'pinky2',  'pinky3',
    'ring1',   'ring2',   'ring3',
    'thumb1',  'thumb2',  'thumb3',
]

# MediaPipe 21-joint indices that correspond to each MANO-16 joint (in kintree order).
# Used to select GT 2D keypoints from grasping dataset for the reprojection loss.
# MediaPipe: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
_MANO_TO_MP_IDX = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 2, 3, 4]


def _procrustes_umeyama(source, target):
    """
    Umeyama (1991) similarity transform: target ≈ s * R @ source + t
    source, target : (N, 3) float arrays of corresponding points.
    Returns (scale, R, t).
    """
    n = source.shape[0]
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    sc = source - mu_s
    tc = target - mu_t
    var_s = np.mean(np.sum(sc ** 2, axis=1))
    K = (tc.T @ sc) / n
    U, sigma, Vt = np.linalg.svd(K)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, float(np.sign(d))])
    R = U @ D @ Vt
    s = float(np.sum(sigma * np.diag(D)) / var_s)
    t = mu_t - s * (R @ mu_s)
    return s, R, t


def _seg_dist_batch(verts, a, b):
    """Vectorised distance from each row in verts (V,3) to segment a→b (3,)."""
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-8
    t = np.clip(((verts - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(verts - proj, axis=1)


def _compute_bone_weights(verts, J, parents, sigma=8.0):
    """
    Envelope (bone-distance) skinning with per-joint sigma.

    The wrist joint (J00, root) uses a large sigma so it owns the palm.
    Finger joints use a tight sigma so they only influence their own segment.
    Without this, the palm (which is physically rigid) gets pulled by all
    five finger base joints simultaneously → candy-wrapper collapse.

    Per-joint sigma (mm):
      wrist (J00)                 : 40mm — must own the entire palm
      finger base joints (J01,04,07,10,13) : 14mm — proximal phalanx + palm edge
      finger mid/distal joints    :  8mm — tight, one segment only
    Returns (V, 16) normalised float32.
    """
    # Joint index → sigma (mm)
    # MANO kintree order: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky,
    #                     10-12=ring, 13-15=thumb
    _SIGMA = {
        0:  40.0,   # wrist  — owns the full palm
        1:  14.0,   # index1
        4:  14.0,   # middle1
        7:  14.0,   # pinky1
        10: 14.0,   # ring1
        13: 14.0,   # thumb1
    }
    _DEFAULT_SIGMA = 8.0   # all other finger segments

    n_joints = len(parents)
    W = np.zeros((len(verts), n_joints), dtype=np.float64)
    for j, p in enumerate(parents):
        s = _SIGMA.get(j, _DEFAULT_SIGMA)
        if p < 0:
            dists = np.linalg.norm(verts - J[j], axis=1)
        else:
            dists = _seg_dist_batch(verts, J[p], J[j])
        W[:, j] = np.exp(-dists ** 2 / (2.0 * s ** 2))
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-8)
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Penetration loss
# ---------------------------------------------------------------------------

class PenetrationLoss(nn.Module):
    """
    Capsule-sphere penetration loss adapted from DeepHand's PenetLoss
    (temp_deephand/common/nets/loss.py, Facebook Research).

    Each bone (parent_j → child_j) is approximated by N_SPHERES evenly-spaced
    spheres.  Radii are pre-computed at init as the distance from each sphere
    centre to the nearest vertex of the Korean rest-pose mesh.

    At forward time, sphere centres are recomputed from the predicted 16 joint
    positions.  Any pair of non-adjacent bones whose spheres overlap contributes
    a penalty:  clamp(r_i + r_j - dist(c_i, c_j), min=0).

    N_SPHERES=10 matches DeepHand's bone_step_size=0.1 (arange(0,1,0.1) → 10 pts).
    """
    N_SPHERES = 10

    def __init__(self, template_verts_mm, J_rest_mm, parents):
        """
        template_verts_mm : (V, 3) Korean rest-pose vertices in mm
        J_rest_mm         : (16, 3) rest-pose joint positions in mm
        parents           : list[int] length 16, root=-1
        """
        super().__init__()

        bones = [(parents[j], j) for j in range(len(parents)) if parents[j] >= 0]
        self.bones = bones   # 15 bones for MANO 16-joint tree

        # Pre-compute sphere radii on the Korean rest-pose mesh
        steps = np.linspace(0.0, 1.0, self.N_SPHERES, endpoint=False)  # [0,.1,..,.9]
        V = np.asarray(template_verts_mm, dtype=np.float64)
        J = np.asarray(J_rest_mm, dtype=np.float64)

        radii = np.zeros((len(bones), self.N_SPHERES), dtype=np.float32)
        for bi, (pi, ci) in enumerate(bones):
            a, b = J[pi], J[ci]
            for si, s in enumerate(steps):
                centre = a + s * (b - a)
                radii[bi, si] = float(np.linalg.norm(V - centre, axis=1).min()) * 0.7

        self.register_buffer('radii', torch.from_numpy(radii))         # (n_bones, N)
        self.register_buffer('steps_t',
                             torch.from_numpy(steps.astype(np.float32)))  # (N,)

        # Non-adjacent bone pairs: skip any pair sharing a joint endpoint
        pairs = []
        for i in range(len(bones)):
            for j in range(i + 1, len(bones)):
                if not ({bones[i][0], bones[i][1]} & {bones[j][0], bones[j][1]}):
                    pairs.append((i, j))
        # Register pair index tensors for vectorized forward
        if pairs:
            pi_idx = torch.tensor([a for a, _ in pairs], dtype=torch.long)
            pj_idx = torch.tensor([b for _, b in pairs], dtype=torch.long)
        else:
            pi_idx = torch.zeros(0, dtype=torch.long)
            pj_idx = torch.zeros(0, dtype=torch.long)
        self.register_buffer('pair_i', pi_idx)
        self.register_buffer('pair_j', pj_idx)
        self.n_pairs = len(pairs)
        print(f'[PenetrationLoss] {len(bones)} bones, {len(pairs)} non-adjacent pairs, '
              f'{self.N_SPHERES} spheres/bone')

    def forward(self, joints_mm):
        """
        joints_mm : (B, 16, 3) joint positions in mm.
        Returns scalar mean penetration loss (fully vectorized, no Python loops).
        """
        if self.n_pairs == 0:
            return joints_mm.new_zeros(1)

        B = joints_mm.shape[0]
        N = self.N_SPHERES

        # All bone centres at once: (B, n_bones, N, 3)
        pi = torch.tensor([p for p, _ in self.bones], device=joints_mm.device)
        ci = torch.tensor([c for _, c in self.bones], device=joints_mm.device)
        a = joints_mm[:, pi, :]                                          # (B, n_bones, 3)
        b = joints_mm[:, ci, :]                                          # (B, n_bones, 3)
        centres = a[:, :, None, :] + self.steps_t[None, None, :, None] * (b - a)[:, :, None, :]
        # centres: (B, n_bones, N, 3)

        # Gather non-adjacent pairs
        c_i = centres[:, self.pair_i]   # (B, n_pairs, N, 3)
        c_j = centres[:, self.pair_j]   # (B, n_pairs, N, 3)

        # Pairwise distances via cdist: reshape to (B*n_pairs, N, 3)
        P = self.n_pairs
        dist = torch.cdist(c_i.view(B * P, N, 3),
                           c_j.view(B * P, N, 3))          # (B*P, N, N)
        dist = dist.view(B, P, N, N)

        # Radii sums: (n_pairs, N, N)
        r_i = self.radii[self.pair_i]   # (n_pairs, N)
        r_j = self.radii[self.pair_j]   # (n_pairs, N)
        r_sum = r_i[:, :, None] + r_j[:, None, :]          # (n_pairs, N, N)

        pen = torch.clamp(r_sum.unsqueeze(0) - dist, min=0.0)   # (B, n_pairs, N, N)
        return pen.mean()


# ---------------------------------------------------------------------------
# LBS data setup
# ---------------------------------------------------------------------------

def setup_lbs_data():
    """
    Returns (J_np, weights_kmano_np, parents_list).

    J              : (16, 3) joint positions in Korean-mesh mm space
    weights_kmano  : (11279, 16) bone-distance blend weights
    parents_list   : list[int] length 16, root=-1

    Fix applied:
      Old code computed J = J_reg @ mano_rest  (MANO ±47 mm space) then
      did KNN weight transfer to Korean mesh (±104 mm).  The ~2.2× scale
      mismatch caused median KNN distance of 27 mm → 63 % wrist dominance.

      New code:
        1. Joint positions → loaded from korean_hand_skeleton.json
           (built directly on Korean mesh geometry, correct scale).
        2. Skinning weights → bone-distance (envelope) skinning computed
           on Korean mesh vertices.  No MANO mesh, no scale mismatch.
        3. Procrustes (Umeyama) run for diagnostic: prints the MANO→Korean
           scale factor so you can verify alignment.
    """
    print('[LBS] Loading MANO PKL for kinematic tree ...')
    with open(MANO_PKL, 'rb') as f:
        mano = pickle.load(f, encoding='latin1')

    kintree = mano['kintree_table']
    parents_list = [int(kintree[0, j]) for j in range(16)]
    parents_list[0] = -1

    # ── Korean template vertices ─────────────────────────────────────────────
    print('[LBS] Loading Korean template mesh ...')
    kmesh      = trimesh.load(KOREAN_TEMPLATE, process=False)
    kmano_rest = np.array(kmesh.vertices, dtype=np.float32)   # (11279, 3)

    # ── Load MANO transferred blend weights ──────────────────────────────────
    print(f'[LBS] Loading MANO transferred weights from {MANO_WEIGHTS_NPY} ...')
    weights_kmano = np.load(MANO_WEIGHTS_NPY).astype(np.float32)  # (11279, 16)
    assert weights_kmano.shape == (NUM_VERTS, 16), \
        f'Expected ({NUM_VERTS}, 16), got {weights_kmano.shape}'

    # ── Compute joint positions from blend weights ───────────────────────────
    # Instead of loading from korean_hand_skeleton.json (Blender bone positions
    # that may not match the weight distribution), compute J as the weighted
    # centroid of vertices per joint. This guarantees J is where the weights
    # say each joint's center of influence is.
    print('[LBS] Computing joint positions from blend weight centroids ...')
    J_np = np.zeros((16, 3), dtype=np.float32)
    for j in range(16):
        w_j = weights_kmano[:, j]  # (V,) weight of this joint
        if w_j.sum() > 1e-6:
            J_np[j] = (w_j[:, None] * kmano_rest).sum(axis=0) / w_j.sum()
        else:
            # Fallback to JSON if joint has no weight
            with open(KOREAN_SKELETON_JSON) as f:
                skel = json.load(f)
            J_np[j] = np.array(skel[_JOINT_NAMES_16[j]]['position_mm'],
                               dtype=np.float32)
    print(f'[LBS] Weight-derived joint positions:')
    for j, name in enumerate(_JOINT_NAMES_16):
        print(f'  J{j:02d} {name:<12}: [{J_np[j,0]:+7.1f}, {J_np[j,1]:+7.1f}, {J_np[j,2]:+7.1f}]')

    dominant = weights_kmano.argmax(axis=1)
    hard = (weights_kmano.max(axis=1) > 0.95).sum()
    print(f'[LBS] MANO weights: {hard}/{len(weights_kmano)} ({100*hard/len(weights_kmano):.1f}%) hard assignment (>0.95)')
    print('[LBS] Dominant joint distribution:')
    for j, name in enumerate(_JOINT_NAMES_16):
        cnt = int((dominant == j).sum())
        if cnt > 0:
            print(f'  J{j:02d} {name:<12}: {cnt:5d} verts')

    print(f'[LBS] J={J_np.shape}, weights={weights_kmano.shape}')
    return J_np, weights_kmano, parents_list


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class KoreanShapeDataset(Dataset):
    """
    Loads all Korean flat-scan OBJ files.
    Each sample is a (V*3,) float32 tensor in mm (zero-centered per scan).
    Uses a disk cache to avoid re-parsing OBJ files on every run.
    """

    def __init__(self, scan_dir=KOREAN_SCAN_DIR, cache_path=SHAPE_CACHE):
        if os.path.exists(cache_path):
            print(f'[KoreanDataset] Loading cache from {cache_path} ...')
            self.data = torch.load(cache_path)
        else:
            print(f'[KoreanDataset] Parsing OBJ files in {scan_dir} ...')
            obj_files = sorted([
                os.path.join(scan_dir, f)
                for f in os.listdir(scan_dir)
                if f.lower().endswith('.obj')
            ])
            print(f'[KoreanDataset] Found {len(obj_files)} OBJ files.')
            all_verts = []
            t0 = time.time()
            for i, path in enumerate(obj_files):
                mesh = trimesh.load(path, process=False)
                v = np.array(mesh.vertices, dtype=np.float32)  # (11279, 3)
                v = v - v.mean(axis=0)                          # zero-center
                all_verts.append(v.reshape(-1))                 # (V*3,)
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    print(f'  Loaded {i+1}/{len(obj_files)}  ({elapsed:.1f}s)')
            self.data = torch.from_numpy(np.stack(all_verts, axis=0))  # (N, V*3)
            print(f'[KoreanDataset] Saving cache to {cache_path} ...')
            torch.save(self.data, cache_path)

        print(f'[KoreanDataset] {len(self.data)} scans, shape={self.data.shape}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (V*3,) float32 in mm


class FreiHANDPoseDataset(Dataset):
    """
    FreiHAND 2D landmarks, MANO pose parameters, and (for 20k samples)
    3D joint labels extracted from posed meshes.

    Returns (landmarks (42,), pose_params (48,), joints_mm (21, 3)).
    Samples without joint labels have joints filled with NaN.
    """

    def __init__(self, landmarks_path=LANDMARKS_NPY, pose_path=POSE_PARAMS_NPY,
                 joint_labels_path=JOINT_LABELS_JSON):
        print(f'[FreiHAND] Loading landmarks from {landmarks_path} ...')
        landmarks = np.load(landmarks_path).astype(np.float32)   # (N, 42)
        print(f'[FreiHAND] Loading pose params from {pose_path} ...')
        pose_params = np.load(pose_path).astype(np.float32)      # (N, 48)
        assert len(landmarks) == len(pose_params)
        N = len(landmarks)
        self.landmarks   = torch.from_numpy(landmarks)
        self.pose_params = torch.from_numpy(pose_params)

        # Joint labels: NaN for samples without labels
        self.joints = torch.full((N, 21, 3), float('nan'), dtype=torch.float32)
        if joint_labels_path and os.path.exists(joint_labels_path):
            print(f'[FreiHAND] Loading joint labels from {joint_labels_path} ...')
            with open(joint_labels_path) as f:
                label_dict = json.load(f)
            # posed_meshes were sampled with seed=42, N_SAMPLES=20000 from FreiHAND
            rng = np.random.RandomState(42)
            fh_indices = rng.choice(N, size=20000, replace=False)
            n_loaded = 0
            for pose_idx, (_, joints_mm) in enumerate(label_dict.items()):
                fh_idx = int(fh_indices[pose_idx])
                self.joints[fh_idx] = torch.tensor(joints_mm, dtype=torch.float32)
                n_loaded += 1
            print(f'[FreiHAND] Joint labels loaded for {n_loaded} samples.')
        else:
            print('[FreiHAND] No joint labels file found — joint loss disabled.')

        print(f'[FreiHAND] {N} samples total.')

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.pose_params[idx], self.joints[idx]


class GraspingPoseDataset(Dataset):
    """
    Grasping dataset with pseudo-GT MANO pose params from fit_mano_to_grasping.py.

    Loaded only when grasping_landmarks.npy + grasping_pose_params.npy exist.
    Samples with high fit error (top 10%) are filtered out before loading.

    Returns same format as FreiHANDPoseDataset:
      (landmarks (42,), pose_params (48,), joints (21, 3) — all NaN)
    So it can be concatenated with FreiHAND for direct L_pose supervision.
    """

    def __init__(self, landmarks_path=GRASP_LANDMARKS_NPY,
                 poses_path=GRASP_POSES_NPY,
                 errors_path=GRASP_ERRORS_NPY,
                 error_percentile=90):
        landmarks  = np.load(landmarks_path).astype(np.float32)   # (N, 42)
        poses      = np.load(poses_path).astype(np.float32)        # (N, 48)
        errors     = np.load(errors_path).astype(np.float32)       # (N,)

        # Filter: keep only the best error_percentile % of fits
        threshold = np.percentile(errors, error_percentile)
        good = errors < threshold
        landmarks = landmarks[good]
        poses     = poses[good]
        N = len(landmarks)

        self.landmarks   = torch.from_numpy(landmarks)                       # (N, 42)
        self.pose_params = torch.from_numpy(poses)                           # (N, 48)
        self.joints      = torch.full((N, 21, 3), float('nan'), dtype=torch.float32)
        print(f'[GraspingPose] {N} samples loaded '
              f'(fit error < {threshold:.4f}, top {error_percentile}%).')

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.pose_params[idx], self.joints[idx]


class EgoMeshDataset(Dataset):
    """
    Ego-vision dataset with real GT posed meshes registered to Korean topology.
    Produced by scripts/preprocess_ego_dataset.py.

    Returns:
      landmarks  (42,)         — normalised 2D keypoints (FreiHAND format)
      posed_verts (11279, 3)   — GT posed vertices in Korean topology (mm)
    """

    def __init__(self, landmarks_path=EGO_LANDMARKS_NPY, verts_path=EGO_POSED_VERTS_NPY):
        lm = np.load(landmarks_path).astype(np.float32)   # (N, 42)
        pv = np.load(verts_path).astype(np.float32)        # (N, 11279, 3)
        assert len(lm) == len(pv), 'ego_landmarks and ego_posed_verts length mismatch'
        self.landmarks   = torch.from_numpy(lm)
        self.posed_verts = torch.from_numpy(pv)
        print(f'[EgoMesh] {len(lm)} frames loaded.')

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.posed_verts[idx]


class GraspingDataset(Dataset):
    """
    손·팔 협조에 의한 파지-조작 동작 데이터
    (Hand-Arm Coordination Grasping-Manipulation Motion Dataset)

    Each sample returns:
      landmarks (42,)  : 21 hand keypoints flattened (x/W, y/H) — PoseEncoder input
      kp2d     (21, 2) : same keypoints as (21,2) — for reprojection loss
      vis      (21,)   : visibility flags (-1=absent, 0=occluded, 1=visible)
    """

    def __init__(self, root_dir=GRASPING_DIR):
        import glob as _glob
        json_dir = os.path.join(root_dir, '02.라벨링데이터')
        json_files = sorted(_glob.glob(
            os.path.join(json_dir, '**', '*.json'), recursive=True
        ))
        print(f'[Grasping] Scanning {len(json_files)} JSON files ...')

        landmarks_list, kp2d_list, vis_list = [], [], []
        skipped = 0
        for path in json_files:
            try:
                with open(path) as f:
                    d = json.load(f)
                W = float(d['image']['width'])
                H = float(d['image']['height'])
                kp2d_flat = d['gesture']['hand_gesture_data']['hand_keypoints']['2D']
                vis_flat  = d['gesture']['hand_gesture_data']['hand_keypoints']['visibility']

                kp2d = np.array(kp2d_flat, dtype=np.float32).reshape(21, 2)
                vis  = np.array(vis_flat,  dtype=np.float32)

                # Match FreiHAND landmark format exactly:
                #   1. Wrist-relative  (wrist = joint 0 → origin)
                #   2. Normalize by wrist→middle DIP distance (MediaPipe joint 11)
                #      so that hand size ≈ 1.0 unit, same as FreiHAND (mean=0.94)
                kp2d -= kp2d[0:1, :]                        # wrist at origin
                scale = np.linalg.norm(kp2d[11]) + 1e-6    # wrist→middle DIP
                kp2d /= scale                               # dimensionless, ~[-2, 2]

                kp2d_list.append(kp2d)
                landmarks_list.append(kp2d.reshape(-1))   # (42,)
                vis_list.append(vis)
            except Exception:
                skipped += 1

        self.landmarks = torch.from_numpy(np.stack(landmarks_list))  # (N, 42)
        self.kp2d      = torch.from_numpy(np.stack(kp2d_list))       # (N, 21, 2)
        self.vis       = torch.from_numpy(np.stack(vis_list))         # (N, 21)
        print(f'[Grasping] Loaded {len(self.landmarks)} samples ({skipped} skipped).')

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.kp2d[idx], self.vis[idx]


# ---------------------------------------------------------------------------
# Utility: load faces from OBJ (for Laplacian)
# ---------------------------------------------------------------------------

def load_template_faces(obj_path):
    """Return (F, 3) numpy int array of triangle face indices."""
    mesh = trimesh.load(obj_path, process=False)
    return np.array(mesh.faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad_(True)


def run_epoch(model, pose_loader, shape_iter, lap_loss_fn,
              optimizer, device, epoch, is_train=True, max_batches=None,
              penet_loss_fn=None, grasping_iter=None, ego_mesh_iter=None):
    """
    Run one training or validation epoch.
    For validation, shape_iter and grasping_iter are None.
    Returns dict of average losses.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_shape       = 0.0
    total_pose        = 0.0
    total_cross       = 0.0
    total_cross_corr  = 0.0
    total_joint       = 0.0
    total_lap         = 0.0
    total_penet       = 0.0
    total_corr        = 0.0
    total_reproj      = 0.0
    total_tip         = 0.0
    total_ego_mesh    = 0.0
    total_loss        = 0.0
    n_batches         = 0

    beta = get_beta(epoch)

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for pose_batch in pose_loader:
            if max_batches is not None and n_batches >= max_batches:
                break
            landmarks, gt_pose, gt_joints = [x.to(device) for x in pose_batch]
            B = landmarks.size(0)

            if is_train:
                shape_batch = next(shape_iter).to(device)   # (B, V*3) mm
                # Truncate / pad to match batch size if cycle mismatch
                if shape_batch.size(0) != B:
                    shape_batch = shape_batch[:B]
                    if shape_batch.size(0) == 0:
                        continue

            # ---- Shape loss (train only) ----
            if is_train:
                mu, logvar = model.encode_shape(shape_batch)
                z = model.reparameterize(mu, logvar)
                recon = model.decode_shape(z).view(B, -1)   # (B, V*3) mm
                # Normalize to dm so mse_shape is ~same scale as loss_pose (rad²)
                mse_shape = F.mse_loss(recon / SCALE, shape_batch / SCALE)
                kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
                loss_shape = mse_shape + beta * kld

                # Laplacian regularizer on decoded shape (normalized)
                decoded_verts = model.decode_shape(z) / SCALE      # (B, V, 3) dm
                loss_lap = lap_loss_fn(decoded_verts)
            else:
                loss_shape = torch.zeros(1, device=device)
                loss_lap   = torch.zeros(1, device=device)
                z          = None
                mu         = None
                logvar     = None

            # ---- Pose loss ----
            # Landmark augmentation (train only): scale jitter ±5% + Gaussian noise.
            # GT pose params are rotation-based so they are scale-invariant — safe to jitter.
            # This makes the pose encoder robust to small detection errors.
            if is_train:
                s = 1.0 + LM_SCALE_JITTER * (2 * torch.rand(B, 1, device=device) - 1)
                landmarks_aug = landmarks * s + LM_NOISE_STD * torch.randn_like(landmarks)
            else:
                landmarks_aug = landmarks
            pred_pose = model.encode_pose(landmarks_aug)
            loss_pose = F.mse_loss(pred_pose, gt_pose)

            # ---- Cross loss (train only, after warm-up) ----
            # Pose the Korean template with GT pose and predicted pose, compare vertices.
            # This gives dense per-vertex supervision (V×3) vs the sparse joint signal.
            # Using the same template for both removes shape as a confound — only pose differs.
            tmpl_mm = model.template.unsqueeze(0).expand(B, -1, -1) * model.scale  # (B, V, 3) mm
            if is_train and epoch >= CROSS_START:
                gt_posed_verts,   _ = model.pose_mesh(tmpl_mm, gt_pose)    # (B, V, 3) mm
                pred_posed_verts, pred_joints = model.pose_mesh(tmpl_mm, pred_pose)  # (B, V, 3) mm
                loss_cross = F.mse_loss(pred_posed_verts / SCALE, gt_posed_verts / SCALE)
            else:
                _, pred_joints = model.pose_mesh(tmpl_mm, pred_pose)
                loss_cross = torch.zeros(1, device=device)

            # ---- Joint position loss + Penetration loss ----
            # pred_joints already computed above in cross loss block
            valid = ~gt_joints.isnan().any(dim=-1).any(dim=-1)  # (B,)

            if valid.any() and pred_joints is not None:
                # Normalize joints: wrist-relative + scale by hand size.
                # FreiHAND hands are ~1.3x larger than Korean template, so
                # absolute and even wrist-relative distances differ by scale.
                # Dividing by hand size (wrist→middle3 distance) makes the
                # loss purely about pose/angles, not bone length.
                pj = pred_joints[valid]                    # (N, 16, 3) mm
                gj = gt_joints[valid, :16, :]              # (N, 16, 3) mm
                pj_rel = pj - pj[:, :1, :]
                gj_rel = gj - gj[:, :1, :]
                # middle3 = joint index 6 in both MANO and Korean skeleton
                p_scale = pj_rel[:, 6:7, :].norm(dim=-1, keepdim=True).clamp(min=1.0)
                g_scale = gj_rel[:, 6:7, :].norm(dim=-1, keepdim=True).clamp(min=1.0)
                loss_joint = F.mse_loss(pj_rel / p_scale, gj_rel / g_scale)
            else:
                loss_joint = torch.zeros(1, device=device)

            # Fingertip loss — uses the 5 fingertip vertices from the posed template.
            # GT fingertips in MediaPipe 21-joint: joints 4,8,12,16,20 (not 16:21).
            # Model fingertip vertex order: thumb,index,middle,ring,pinky (joint_vert_idx[16:]).
            # Same wrist-relative + scale normalization as L_joint.
            # L_tip removed: Korean template fingertip vertices are fixed LBS points
            # ~10-15mm from true GT fingertip positions — loss was stuck at 0.43 every run.
            # L_cross already supervises all 11279 vertices including fingertip regions.
            loss_tip = torch.zeros(1, device=device)

            # Penetration loss on the 16 true LBS joint centres
            if is_train and penet_loss_fn is not None and pred_joints is not None:
                loss_penet = penet_loss_fn(pred_joints)
            else:
                loss_penet = torch.zeros(1, device=device)

            # ---- Correction loss (train only, when z is available) ----
            # CorrectionNet learns pose+shape vertex offsets to fix LBS candy-wrapper
            # artifacts at knuckles.
            #
            # Key fix: route LAMBDA_CROSS_CORR through the corrected posed mesh so
            # CorrectionNet gets a real vertex-level gradient, not just Laplacian
            # smoothness which the regulariser easily overpowers.
            #
            # Flow:
            #   decoded_rest + corrections → corrected_rest_mm
            #   pose_mesh(corrected_rest_mm, pred_pose) → corr_posed_verts
            #   loss_cross_corr = MSE(corr_posed_verts, gt_posed_verts)
            #   → gradient flows: corr_posed ← pose_mesh ← corrections ← CorrectionNet
            loss_cross_corr = torch.zeros(1, device=device)
            if is_train and z is not None:
                pose_corr, shape_corr = model.correction_net(pred_pose.detach(), z)
                corr_verts_dm  = decoded_verts + pose_corr + shape_corr   # (B, V, 3) dm
                corr_verts_mm  = corr_verts_dm * SCALE                    # dm → mm

                # Laplacian on corrected rest shape
                loss_lap = lap_loss_fn(corr_verts_dm)

                # Vertex-level signal: pose the corrected mesh, compare to GT posed template
                if epoch >= CROSS_START:
                    corr_posed, _ = model.pose_mesh(corr_verts_mm, pred_pose)
                    loss_cross_corr = F.mse_loss(
                        corr_posed / SCALE,
                        gt_posed_verts.detach() / SCALE   # detach GT path
                    )

                # Tiny regulariser: just prevents corrections from exploding
                loss_corr = pose_corr.pow(2).mean() + shape_corr.pow(2).mean()
            else:
                loss_corr = torch.zeros(1, device=device)

            # ---- Reprojection loss (grasping dataset, train only) ----
            # No MANO pose params available → use 2D geometric consistency instead.
            # Strategy: feed grasping 2D landmarks to PoseEncoder → LBS joints →
            # orthographic projection (drop Z) → compare wrist-relative normalised
            # XY to GT 2D keypoints.  Both sides normalised by wrist→middle3 distance
            # so the loss is purely about finger angles, not absolute scale.
            if is_train and grasping_iter is not None:
                g_lm, g_kp2d, g_vis = [x.to(device) for x in next(grasping_iter)]
                # Truncate to batch size
                g_lm   = g_lm[:B];  g_kp2d = g_kp2d[:B];  g_vis = g_vis[:B]
                Bg = g_lm.shape[0]

                g_pred_pose = model.encode_pose(g_lm)          # (Bg, 48)
                tmpl_g = model.template.unsqueeze(0).expand(Bg, -1, -1) * model.scale
                _, g_joints = model.pose_mesh(tmpl_g, g_pred_pose)  # (Bg, 16, 3) mm

                # Orthographic projection: take X, Y from predicted 3D joints
                g_pred_2d = g_joints[:, :, :2]                 # (Bg, 16, 2) mm

                # Select corresponding MediaPipe GT joints for the 16 MANO joints
                mp_idx = torch.tensor(_MANO_TO_MP_IDX, device=device)
                g_gt_2d = g_kp2d[:, mp_idx, :]                # (Bg, 16, 2) in [0,1]

                # GT is already wrist-relative and scale-normalised (matches FreiHAND format).
                # Predicted joints are in mm with wrist near origin (LBS wrist-centred).
                # Normalise predicted by its wrist→middle DIP distance (MANO joint 6)
                # so it's in the same ~1.0-unit scale as GT.
                g_pred_rel = g_pred_2d - g_pred_2d[:, :1, :]  # (Bg,16,2) mm, wrist=0
                g_gt_rel   = g_gt_2d                           # already wrist-relative, ~1.0 scale

                p_sc = g_pred_rel[:, 6:7, :].norm(dim=-1, keepdim=True).clamp(min=1.0)
                loss_reproj = F.mse_loss(g_pred_rel / p_sc, g_gt_rel)
            else:
                loss_reproj = torch.zeros(1, device=device)

            # ---- Ego-mesh loss (ego-vision dataset, train only) ----
            # Real GT posed meshes (registered to Korean topology via NN-map).
            # Supervises the full output path: landmarks → PoseEncoder → CorrectionNet
            # (pose branch only; shape identity unknown so z_shape=zeros) → LBS.
            # This breaks the circular LBS-supervised-by-LBS problem in L_ccorr.
            loss_ego_mesh = torch.zeros(1, device=device)
            if is_train and ego_mesh_iter is not None:
                eg_lm, eg_gt_verts = [x.to(device) for x in next(ego_mesh_iter)]
                eg_lm       = eg_lm[:B]
                eg_gt_verts = eg_gt_verts[:B]      # (Beg, 11279, 3) mm
                Beg = eg_lm.shape[0]

                eg_pred_pose = model.encode_pose(eg_lm)   # (Beg, 48)

                # Detach pose from ego mesh loss — 134 frames is too few to train the
                # pose encoder (38k FreiHAND+grasping already handles that).
                # L_ego only supervises CorrectionNet: teach it to fix LBS artifacts
                # given a pose, without pulling pose encoder weights toward ego distribution.
                eg_pose_detached = eg_pred_pose.detach()

                # Pose correction only — shape identity unknown for ego samples
                z_zero = torch.zeros(Beg, SHAPE_DIM, device=device)
                pose_corr_eg, _ = model.correction_net(eg_pose_detached, z_zero)

                # Corrected template → pose → compare to GT
                eg_rest_dm   = model.template.unsqueeze(0).expand(Beg, -1, -1) + pose_corr_eg
                eg_corr_posed, _ = model.pose_mesh(eg_rest_dm * model.scale, eg_pose_detached)

                loss_ego_mesh = F.mse_loss(eg_corr_posed / SCALE, eg_gt_verts / SCALE)

            # ---- Total loss ----
            loss = (LAMBDA_SHAPE       * loss_shape
                    + LAMBDA_POSE        * loss_pose
                    + LAMBDA_CROSS       * loss_cross
                    + LAMBDA_CROSS_CORR  * loss_cross_corr
                    + LAMBDA_JOINT       * loss_joint
                    + LAMBDA_TIP         * loss_tip
                    + LAMBDA_LAP         * loss_lap
                    + LAMBDA_PENET       * loss_penet
                    + LAMBDA_CORR        * loss_corr
                    + LAMBDA_REPROJ      * loss_reproj
                    + LAMBDA_EGO_MESH    * loss_ego_mesh)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_shape      += loss_shape.item()
            total_pose       += loss_pose.item()
            total_cross      += loss_cross.item()
            total_cross_corr += loss_cross_corr.item()
            total_joint      += loss_joint.item()
            total_lap        += loss_lap.item()
            total_penet      += loss_penet.item()
            total_corr       += loss_corr.item()
            total_reproj     += loss_reproj.item()
            total_tip        += loss_tip.item()
            total_ego_mesh   += loss_ego_mesh.item()
            total_loss       += loss.item()
            n_batches        += 1

    n = max(n_batches, 1)
    return {
        'loss_shape':      total_shape      / n,
        'loss_pose':       total_pose       / n,
        'loss_cross':      total_cross      / n,
        'loss_cross_corr': total_cross_corr / n,
        'loss_joint':      total_joint      / n,
        'loss_lap':        total_lap        / n,
        'loss_penet':      total_penet      / n,
        'loss_corr':       total_corr       / n,
        'loss_reproj':     total_reproj     / n,
        'loss_tip':        total_tip        / n,
        'loss_ego_mesh':   total_ego_mesh   / n,
        'total':           total_loss       / n,
    }


def main(epochs=EPOCHS, max_batches=None, resume=False):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[main] Using device: {device}')

    # ------------------------------------------------------------------ #
    # 1. LBS data
    # ------------------------------------------------------------------ #
    J_np, weights_kmano_np, parents_list = setup_lbs_data()

    # ------------------------------------------------------------------ #
    # 2. Korean template vertices
    # ------------------------------------------------------------------ #
    print('[main] Loading Korean template mesh ...')
    tmesh = trimesh.load(KOREAN_TEMPLATE, process=False)
    template_verts = np.array(tmesh.vertices, dtype=np.float32)  # (11279, 3)
    korean_faces   = np.array(tmesh.faces,    dtype=np.int64)    # (F, 3)
    print(f'[main] Template: {template_verts.shape}, faces: {korean_faces.shape}')

    # ------------------------------------------------------------------ #
    # 3. Joint vertex indices — 16 LBS joints from korean_hand_skeleton.json
    #    + 5 fingertip vertices detected from Korean mesh geometry (max-Y
    #    per finger X-band).  Total = 21 to match FreiHAND GT annotations.
    # ------------------------------------------------------------------ #
    print('[main] Building 21 joint vertex indices from Korean mesh ...')
    with open(KOREAN_SKELETON_JSON) as f:
        skel = json.load(f)

    # 16 LBS joints in MANO kintree order
    joint_vert_indices_16 = np.array(
        [skel[n]['vertex_index'] for n in _JOINT_NAMES_16], dtype=np.int64
    )

    # 5 fingertip vertices: among the 50 nearest verts to each distal joint,
    # pick the one with the highest Y (furthest from wrist).
    # This guarantees unique vertices per finger regardless of X overlap.
    distal_joints = {
        'thumb':  np.array(skel['thumb3']['position_mm'],  dtype=np.float32),
        'index':  np.array(skel['index3']['position_mm'],  dtype=np.float32),
        'middle': np.array(skel['middle3']['position_mm'], dtype=np.float32),
        'ring':   np.array(skel['ring3']['position_mm'],   dtype=np.float32),
        'pinky':  np.array(skel['pinky3']['position_mm'],  dtype=np.float32),
    }
    tip_indices = []
    used = set()
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        j3 = distal_joints[finger]
        dists = np.linalg.norm(template_verts - j3, axis=1)
        candidates = np.argsort(dists)[:50]
        # Pick highest-Y among candidates not already used
        candidates = [v for v in candidates if v not in used]
        tip_v = int(candidates[np.argmax(template_verts[candidates, 1])])
        used.add(tip_v)
        tip_indices.append(tip_v)
        print(f'  {finger}_tip → vert {tip_v}  '
              f'pos=({template_verts[tip_v,0]:.1f}, {template_verts[tip_v,1]:.1f}, {template_verts[tip_v,2]:.1f})')

    joint_vert_indices = np.concatenate(
        [joint_vert_indices_16, np.array(tip_indices, dtype=np.int64)]
    )  # (21,)
    print(f'[main] Joint vertex indices: {joint_vert_indices.shape}')

    # ------------------------------------------------------------------ #
    # 4. Model
    # ------------------------------------------------------------------ #
    model = UnifiedHandModel(
        template_verts      = template_verts,
        weights_kmano       = weights_kmano_np,
        J                   = J_np,
        parents             = parents_list,
        latent_dim          = SHAPE_DIM,
        scale               = SCALE,
        joint_vert_indices  = joint_vert_indices,
    ).to(device)

    # ------------------------------------------------------------------ #
    # 5. Load pretrained pose lifter weights (skipped when resuming)
    # ------------------------------------------------------------------ #
    start_epoch   = 0
    best_val_pose = float('inf')
    history = {
        'train_shape': [], 'train_pose': [], 'train_cross': [],
        'train_joint': [], 'val_pose': [], 'total': [], 'epochs': [],
    }

    if resume and os.path.exists(OUTPUT_MODEL):
        print(f'[main] Resuming from checkpoint {OUTPUT_MODEL} ...')
        ckpt = torch.load(OUTPUT_MODEL, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_pose = ckpt['val_pose_loss']
        # Restore history; add 'epochs' key if it was missing in old checkpoint
        old_hist = ckpt.get('history', {})
        for k in ('train_shape', 'train_pose', 'train_cross', 'train_joint', 'val_pose', 'total'):
            history[k] = old_hist.get(k, [])
        n = len(history['train_shape'])
        history['epochs'] = old_hist.get('epochs', list(range(n)))
        print(f'[main] Checkpoint loaded: best val_pose={best_val_pose:.4f}, resuming from epoch {start_epoch}.')
    else:
        print(f'[main] Loading pretrained pose lifter from {POSE_LIFTER_PTH} ...')
        pose_lifter_state = torch.load(POSE_LIFTER_PTH, map_location=device)
        if isinstance(pose_lifter_state, dict) and 'model_state_dict' in pose_lifter_state:
            pose_lifter_state = pose_lifter_state['model_state_dict']
        # strict=False: new PoseEncoder has different architecture than old PoseLifter.
        # Matching layers will be loaded, new layers init randomly.
        missing, unexpected = model.pose_encoder.load_state_dict(pose_lifter_state, strict=False)
        if missing:
            print(f'[main] Pose lifter: {len(missing)} new params (randomly initialized).')
        if unexpected:
            print(f'[main] Pose lifter: {len(unexpected)} old params skipped (arch changed).')
        print('[main] Pose lifter weights loaded (strict=False — new architecture).')

    # ------------------------------------------------------------------ #
    # 5b. Freeze/unfreeze pose encoder based on start_epoch
    # ------------------------------------------------------------------ #
    if start_epoch >= POSE_UNFREEZE:
        unfreeze_module(model.pose_encoder)
        unfreeze_module(model.correction_net)
        pose_encoder_frozen = False
        print(f'[main] PoseEncoder + CorrectionNet unfrozen (start_epoch={start_epoch} >= POSE_UNFREEZE={POSE_UNFREEZE}).')
    else:
        freeze_module(model.pose_encoder)
        freeze_module(model.correction_net)
        pose_encoder_frozen = True
        print(f'[main] PoseEncoder + CorrectionNet frozen until epoch {POSE_UNFREEZE}.')

    # ------------------------------------------------------------------ #
    # 6. Datasets & loaders
    # ------------------------------------------------------------------ #
    print('[main] Setting up datasets ...')
    shape_dataset = KoreanShapeDataset(scan_dir=KOREAN_SCAN_DIR, cache_path=SHAPE_CACHE)
    pose_dataset  = FreiHANDPoseDataset(landmarks_path=LANDMARKS_NPY, pose_path=POSE_PARAMS_NPY)

    # 90/10 train/val split for FreiHAND (val stays FreiHAND-only for clean comparison)
    n_val   = max(1, int(0.1 * len(pose_dataset)))
    n_train = len(pose_dataset) - n_val
    pose_train_freihand, pose_val = random_split(
        pose_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Optionally augment training set with grasping pseudo-GT pose params
    # (produced by scripts/fit_mano_to_grasping.py — only loaded if files exist)
    if (os.path.exists(GRASP_LANDMARKS_NPY) and os.path.exists(GRASP_POSES_NPY)
            and os.path.exists(GRASP_ERRORS_NPY)):
        from torch.utils.data import ConcatDataset
        grasp_pose_dataset = GraspingPoseDataset(
            landmarks_path=GRASP_LANDMARKS_NPY,
            poses_path=GRASP_POSES_NPY,
            errors_path=GRASP_ERRORS_NPY,
        )
        pose_train = ConcatDataset([pose_train_freihand, grasp_pose_dataset])
        print(f'[main] Pose train: {len(pose_train_freihand)} FreiHAND + '
              f'{len(grasp_pose_dataset)} grasping = {len(pose_train)} total.')
    else:
        pose_train = pose_train_freihand
        print(f'[main] Grasping pose params not found — using FreiHAND only '
              f'({len(pose_train)} samples). Run scripts/fit_mano_to_grasping.py first.')

    shape_loader = DataLoader(shape_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)
    pose_train_loader = DataLoader(pose_train, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0, drop_last=False)
    pose_val_loader   = DataLoader(pose_val, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=0, drop_last=False)

    print(f'[main] shape_loader: {len(shape_loader)} batches/epoch')
    print(f'[main] pose_train_loader: {len(pose_train_loader)} batches/epoch')
    print(f'[main] pose_val_loader: {len(pose_val_loader)} batches/epoch')

    # ------------------------------------------------------------------ #
    # 6b. Grasping dataset (reprojection loss)
    # ------------------------------------------------------------------ #
    grasping_loader = None
    if os.path.isdir(GRASPING_DIR):
        try:
            grasping_dataset = GraspingDataset(root_dir=GRASPING_DIR)
            grasping_loader  = DataLoader(grasping_dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0, drop_last=True)
            print(f'[main] grasping_loader: {len(grasping_loader)} batches/epoch')
        except Exception as e:
            print(f'[main] Grasping dataset failed to load ({e}), skipping.')
    else:
        print(f'[main] Grasping data not found at {GRASPING_DIR}, reprojection loss disabled.')

    # ------------------------------------------------------------------ #
    # 6c. Ego-vision dataset (real GT posed-mesh loss)
    # ------------------------------------------------------------------ #
    ego_mesh_loader = None
    if os.path.exists(EGO_LANDMARKS_NPY) and os.path.exists(EGO_POSED_VERTS_NPY):
        try:
            ego_mesh_dataset = EgoMeshDataset(
                landmarks_path=EGO_LANDMARKS_NPY,
                verts_path=EGO_POSED_VERTS_NPY,
            )
            ego_mesh_loader = DataLoader(ego_mesh_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=0, drop_last=True)
            print(f'[main] ego_mesh_loader: {len(ego_mesh_loader)} batches/epoch '
                  f'({len(ego_mesh_dataset)} frames)')
        except Exception as e:
            print(f'[main] Ego mesh dataset failed to load ({e}), skipping.')
    else:
        print(f'[main] Ego mesh data not found — run scripts/preprocess_ego_dataset.py first.')

    # ------------------------------------------------------------------ #
    # 7. Laplacian loss
    # ------------------------------------------------------------------ #
    lap_loss_fn = LaplacianLoss(korean_faces, num_vertices=NUM_VERTS).to(device)

    # ------------------------------------------------------------------ #
    # 8. Penetration loss (10-sphere capsule method, adapted from DeepHand)
    # ------------------------------------------------------------------ #
    penet_loss_fn = PenetrationLoss(template_verts, J_np, parents_list).to(device)

    # ------------------------------------------------------------------ #
    # 9. Optimizer & scheduler
    # ------------------------------------------------------------------ #
    # Two param groups: base (full LR) and pose+correction (lower LR at unfreeze).
    # Splitting upfront avoids the "appears in more than one group" error.
    pose_corr_ids = set(
        id(p) for p in list(model.pose_encoder.parameters()) +
                        list(model.correction_net.parameters())
    )
    base_params = [p for p in model.parameters() if id(p) not in pose_corr_ids]
    pose_corr_params = list(model.pose_encoder.parameters()) + \
                       list(model.correction_net.parameters())
    optimizer = torch.optim.AdamW([
        {'params': base_params,       'lr': LR,               'weight_decay': 1e-5},
        {'params': pose_corr_params,  'lr': LR_POSE_UNFREEZE, 'weight_decay': 1e-4},
    ])
    if resume and os.path.exists(OUTPUT_MODEL):
        ckpt_opt = torch.load(OUTPUT_MODEL, map_location=device)
        if 'optimizer_state' in ckpt_opt:
            optimizer.load_state_dict(ckpt_opt['optimizer_state'])
            print('[main] Optimizer state restored from checkpoint.')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHED_T_MAX, last_epoch=start_epoch - 1
    )

    # ------------------------------------------------------------------ #
    # 9. Training loop
    # ------------------------------------------------------------------ #
    print(f'\n[main] Starting training for {epochs} epochs (from epoch {start_epoch}) ...\n')

    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        # Unfreeze pose encoder when scheduled
        if pose_encoder_frozen and epoch >= POSE_UNFREEZE:
            unfreeze_module(model.pose_encoder)
            unfreeze_module(model.correction_net)
            pose_encoder_frozen = False
            print(f'[Epoch {epoch}] PoseEncoder + CorrectionNet UNFROZEN (lr={LR_POSE_UNFREEZE:.0e}).')

        # Cycle shape, grasping, and ego loaders to match length of pose loader
        shape_iter     = cycle(shape_loader)
        grasping_iter  = cycle(grasping_loader)  if grasping_loader  is not None else None
        ego_mesh_iter  = cycle(ego_mesh_loader)  if ego_mesh_loader  is not None else None

        # ---- Train ----
        train_stats = run_epoch(
            model, pose_train_loader, shape_iter, lap_loss_fn,
            optimizer, device, epoch, is_train=True, max_batches=max_batches,
            penet_loss_fn=penet_loss_fn, grasping_iter=grasping_iter,
            ego_mesh_iter=ego_mesh_iter,
        )

        # ---- Validation (pose only) ----
        val_stats = run_epoch(
            model, pose_val_loader, None, lap_loss_fn,
            optimizer, device, epoch, is_train=False, max_batches=max_batches,
        )
        val_pose_loss = val_stats['loss_pose']

        scheduler.step()

        # Record history
        history['epochs'].append(epoch)
        history['train_shape'].append(train_stats['loss_shape'])
        history['train_pose'].append(train_stats['loss_pose'])
        history['train_cross'].append(train_stats['loss_cross'])
        history['train_joint'].append(train_stats['loss_joint'])
        history['val_pose'].append(val_pose_loss)
        history['total'].append(train_stats['total'])

        beta = get_beta(epoch)
        elapsed = time.time() - t_start
        print(
            f'Epoch [{epoch:>3d}/{epochs}]  '
            f'L_shape={train_stats["loss_shape"]:.4f}  '
            f'L_pose={train_stats["loss_pose"]:.4f}  '
            f'L_joint={train_stats["loss_joint"]:.4f}  '
            f'L_cross={train_stats["loss_cross"]:.4f}  '
            f'L_ccorr={train_stats["loss_cross_corr"]:.4f}  '
            f'L_reproj={train_stats["loss_reproj"]:.4f}  '
            f'L_tip={train_stats["loss_tip"]:.4f}  '
            f'L_ego={train_stats["loss_ego_mesh"]:.4f}  '
            f'L_lap={train_stats["loss_lap"]:.4f}  '
            f'L_penet={train_stats["loss_penet"]:.4f}  '
            f'L_corr={train_stats["loss_corr"]:.4f}  '
            f'beta={beta:.2e}  '
            f'val_pose={val_pose_loss:.4f}  '
            f'lr={scheduler.get_last_lr()[0]:.2e}  '
            f't={elapsed:.1f}s'
        )

        # Save best model
        if val_pose_loss < best_val_pose:
            best_val_pose = val_pose_loss
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state':  optimizer.state_dict(),
                'val_pose_loss':    val_pose_loss,
                'history':          history,
            }, OUTPUT_MODEL)
            print(f'  -> Best model saved (val_pose={val_pose_loss:.4f})')

    print(f'\n[main] Training complete. Best val_pose_loss={best_val_pose:.4f}')
    print(f'[main] Best model saved to {OUTPUT_MODEL}')

    # ------------------------------------------------------------------ #
    # 10. Plot training curves
    # ------------------------------------------------------------------ #
    print('[main] Saving training curves ...')
    ep = history['epochs'] if history['epochs'] else list(range(len(history['train_shape'])))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep, history['train_shape'], label='L_shape (train)', color='steelblue')
    axes[0].set_title('Shape Loss (MSE + β·KLD)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history['train_pose'], label='L_pose (train)', color='darkorange')
    axes[1].plot(ep, history['val_pose'],   label='L_pose (val)',   color='firebrick',
                 linestyle='--')
    axes[1].set_title('Pose Loss (MSE)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, history['train_cross'], label='L_cross (train)', color='seagreen')
    axes[2].set_title('Cross Loss (Shape × Pose consistency)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(fig)
    print(f'[main] Plot saved to {OUTPUT_PLOT}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the semi-supervised Unified Shape+Pose hand model.'
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS}).'
    )
    parser.add_argument(
        '--max-batches', type=int, default=None,
        help='Limit batches per epoch (e.g. 10 for quick smoke test).'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from the saved checkpoint (models/unified_hand_model.pth).'
    )
    args = parser.parse_args()
    main(epochs=args.epochs, max_batches=args.max_batches, resume=args.resume)
