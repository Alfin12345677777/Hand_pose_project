"""
inference_demo.py
=================
Full pipeline: image/webcam → MediaPipe → PoseLifter → K-MANO mesh

Usage:
  python inference_demo.py --image path/to/hand.jpg   # single image
  python inference_demo.py --webcam                   # live webcam

Output: saves rendered mesh to outputs/inference_result.png
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import trimesh
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
MODEL_PATH     = '../models/pose_lifter.pth'
MANO_PKL       = '../models/MANO_RIGHT.pkl'
MANO_ALIGNED   = '../meshes/MANO_ALIGNED_TO_KOREA.obj'
KMANO_MESH     = '../meshes/AVERAGE_KOREAN_HAND_CENTERED.obj'
SKELETON_JSON  = '../models/korean_hand_skeleton.json'
WEIGHTS_NPY    = '../data/mano_transferred_weights.npy'
OUTPUT_IMG     = '../outputs/inference_result.png'

MIDDLE_TIP_IDX = 12   # MediaPipe / MANO joint index for middle fingertip

_JOINT_NAMES_16 = [
    'wrist',
    'index1',  'index2',  'index3',
    'middle1', 'middle2', 'middle3',
    'pinky1',  'pinky2',  'pinky3',
    'ring1',   'ring2',   'ring3',
    'thumb1',  'thumb2',  'thumb3',
]

# ──────────────────────────────────────────
# MODEL (must match train_pose_lifter.py)
# ──────────────────────────────────────────
class PoseLifter(nn.Module):
    """Must match PoseEncoder in train_unified_model.py (42 landmarks + 10 bone features = 52)."""
    def __init__(self, in_dim=52, out_dim=48, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res1 = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512),
        )
        self.res2 = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512),
        )
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )
    def forward(self, x):
        h = self.input_proj(x)
        h = torch.relu(h + self.res1(h))
        h = torch.relu(h + self.res2(h))
        return self.output_head(h)


# ──────────────────────────────────────────
# LBS SETUP  (same as generate_posed_meshes)
# ──────────────────────────────────────────
def load_lbs():
    import json

    with open(MANO_PKL, 'rb') as f:
        mano = pickle.load(f, encoding='latin1')

    kintree  = np.array(mano['kintree_table'], dtype=np.int64)
    parents  = kintree[0].copy(); parents[0] = -1

    # Load Korean joint positions (correct scale, on the mesh)
    with open(SKELETON_JSON) as f:
        skel = json.load(f)
    J = np.array([skel[n]['position_mm'] for n in _JOINT_NAMES_16],
                 dtype=np.float64)  # (16, 3) in mm

    # Load Korean mesh
    kmano_mesh = trimesh.load(KMANO_MESH, process=False)
    kmano_rest = np.array(kmano_mesh.vertices, dtype=np.float64)
    kmano_faces= np.array(kmano_mesh.faces, dtype=np.int32)

    # Load blend weights (from Blender or transfer script)
    if os.path.exists(WEIGHTS_NPY):
        weights_kmano = np.load(WEIGHTS_NPY).astype(np.float64)
    else:
        # Fallback: KNN transfer from MANO (not ideal but functional)
        mano_rest = np.array(trimesh.load(MANO_ALIGNED, process=False).vertices, dtype=np.float64)
        weights   = np.array(mano['weights'], dtype=np.float64)
        tree = cKDTree(mano_rest)
        _, kmano_to_mano = tree.query(kmano_rest, workers=-1)
        weights_kmano = weights[kmano_to_mano]

    return J, weights_kmano, parents, kmano_rest, kmano_faces


def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-8: return np.eye(3)
    k = r / theta
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


def lbs_custom(v_rest, pose_params, J, weights, parents):
    Rs = np.stack([rodrigues(pose_params[j*3:(j+1)*3]) for j in range(16)])
    G  = [None] * 16
    for j in range(16):
        p = parents[j]
        T = np.eye(4); T[:3,:3]=Rs[j]; T[:3,3]=J[j] if p<0 else J[j]-J[p]
        G[j] = T if p<0 else G[p]@T
    G = np.stack(G)
    Gf = G.copy()
    for j in range(16): Gf[j,:3,3] = G[j,:3,:3]@(-J[j]) + G[j,:3,3]
    T_blend = np.einsum('vj,jab->vab', weights, Gf)
    vh = np.c_[v_rest, np.ones(len(v_rest))]
    return np.einsum('vab,vb->va', T_blend, vh)[:,:3]


# ──────────────────────────────────────────
# NORMALIZE LANDMARKS  (same as training)
# ──────────────────────────────────────────
def normalize_landmarks(landmarks_2d):
    """
    landmarks_2d: (21, 2) pixel coordinates from MediaPipe
    returns     : (42,)   normalized, ready for model
    """
    lm = landmarks_2d.copy().astype(np.float32)
    lm -= lm[0]                                         # center at wrist
    scale = np.linalg.norm(lm[MIDDLE_TIP_IDX])
    if scale > 1e-6:
        lm /= scale
    return np.clip(lm.flatten(), -5.0, 5.0)


def _extract_bone_features_np(landmarks_flat):
    """
    NumPy version of extract_bone_features for inference.
    landmarks_flat: (42,) normalized 2D landmarks.
    Returns: (10,) bone proportion features.
    """
    lm = landmarks_flat.reshape(21, 2)

    def dist(a, b):
        return max(np.linalg.norm(lm[a] - lm[b]), 1e-6)

    thumb_len = dist(1, 4)
    index_len = dist(5, 8)
    middle_len = dist(9, 12)
    ring_len = dist(13, 16)
    pinky_len = dist(17, 20)
    hand_len = max(dist(0, 12), 1e-6)
    palm_len = dist(0, 9)

    thumb_dir = np.arctan2(lm[2, 1] - lm[0, 1], lm[2, 0] - lm[0, 0])
    index_dir = np.arctan2(lm[5, 1] - lm[0, 1], lm[5, 0] - lm[0, 0])
    pinky_dir = np.arctan2(lm[17, 1] - lm[0, 1], lm[17, 0] - lm[0, 0])

    return np.array([
        thumb_len / middle_len,
        index_len / middle_len,
        ring_len / middle_len,
        pinky_len / middle_len,
        palm_len / hand_len,
        dist(0, 5) / hand_len,
        dist(0, 17) / hand_len,
        thumb_dir - index_dir,
        index_dir - pinky_dir,
        dist(5, 17) / hand_len,
    ], dtype=np.float32)


# ──────────────────────────────────────────
# RENDER RESULT
# ──────────────────────────────────────────
def render_mesh(v, faces, image=None, title='K-MANO Reconstruction'):
    fig = plt.figure(figsize=(14, 5), facecolor='#111')

    if image is not None:
        ax0 = fig.add_subplot(1, 3, 1)
        ax0.imshow(image)
        ax0.set_title('Input Image', color='white', fontsize=10)
        ax0.axis('off')
        start_col = 2
    else:
        start_col = 1

    views = [(20, -60, 'Front'), (0, -90, 'Side')]
    for i, (elev, azim, lbl) in enumerate(views):
        ax = fig.add_subplot(1, 3 if image is not None else 2,
                             start_col + i, projection='3d')
        ax.scatter(v[::4,0], v[::4,1], v[::4,2], s=0.8,
                   c=v[::4,2], cmap='plasma')
        ax.set_facecolor('#111')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'K-MANO — {lbl}', color='white', fontsize=9)
        ax.axis('off')

    plt.suptitle(title, color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=130, bbox_inches='tight', facecolor='#111')
    print(f"Saved → {OUTPUT_IMG}")


# ──────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ──────────────────────────────────────────
def run_inference(landmarks_2d, image=None):
    """
    landmarks_2d : (21, 2) MediaPipe pixel coords
    image        : optional (H, W, 3) for visualization
    """
    device = torch.device('cpu')

    # Load model
    model = PoseLifter()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Normalize + extract bone features + predict
    x = normalize_landmarks(landmarks_2d)
    x_t = torch.tensor(x).unsqueeze(0)  # (1, 42)
    bone_feat = _extract_bone_features_np(x)  # (10,)
    x_full = torch.cat([x_t, torch.tensor(bone_feat).unsqueeze(0)], dim=-1)  # (1, 52)
    with torch.no_grad():
        pose_params = model(x_full).squeeze(0).numpy()

    print(f"Predicted pose range: [{pose_params.min():.3f}, {pose_params.max():.3f}] rad")

    # Load LBS components
    J, weights_kmano, parents, kmano_rest, kmano_faces = load_lbs()

    # Apply pose to K-MANO
    v_posed = lbs_custom(kmano_rest, pose_params, J, weights_kmano, parents)

    # Save mesh
    out_mesh = trimesh.Trimesh(vertices=v_posed, faces=kmano_faces, process=False)
    out_mesh.export(OUTPUT_IMG.replace('.png', '.obj'))

    # Render
    render_mesh(v_posed, kmano_faces, image=image)
    return v_posed, pose_params


# ──────────────────────────────────────────
# DEMO: test with a FreiHAND sample
# ──────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',  type=str, help='Path to hand image')
    parser.add_argument('--sample', type=int, default=0,
                        help='FreiHAND sample index to test with (default: 0)')
    args = parser.parse_args()

    if args.image:
        # Real image — run MediaPipe
        try:
            import mediapipe as mp
            import cv2
        except ImportError:
            print("Install mediapipe and opencv: pip install mediapipe opencv-python")
            sys.exit(1)

        img = cv2.imread(args.image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            print("No hand detected in image.")
            sys.exit(1)

        h, w = img.shape[:2]
        lm = result.multi_hand_landmarks[0]
        landmarks_2d = np.array([[l.x * w, l.y * h]
                                  for l in lm.landmark])   # (21, 2) pixels
        run_inference(landmarks_2d, image=img_rgb)

    else:
        # Test with FreiHAND ground-truth joints (no camera needed)
        import json
        print(f"Testing with FreiHAND sample #{args.sample}...")

        with open('../data/training_xyz.json') as f:
            all_xyz = json.load(f)
        with open('../data/training_K.json') as f:
            all_K = json.load(f)

        joints_3d = np.array(all_xyz[args.sample])
        K = np.array(all_K[args.sample])
        Z = joints_3d[:, 2]
        u = (K[0,0] * joints_3d[:,0] / Z) + K[0,2]
        v = (K[1,1] * joints_3d[:,1] / Z) + K[1,2]
        landmarks_2d = np.stack([u, v], axis=1)   # (21, 2)

        run_inference(landmarks_2d)
        print("\nDone! To test with a real image: python inference_demo.py --image your_hand.jpg")
