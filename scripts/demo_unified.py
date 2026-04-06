"""
demo_unified.py
===============
Visualise 5 posed hand meshes from the trained UnifiedHandModel.

Shows front + side view per sample. Applies CorrectionNet on top of LBS.

Usage:
  my_venv/bin/python scripts/demo_unified.py
  my_venv/bin/python scripts/demo_unified.py --samples 0 100 500 1000 5000

Output:
  outputs/unified_demo.png         — rendered figure
  outputs/unified_demo_0..4.obj    — exportable meshes
"""

import os, sys, argparse
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(__file__))
from train_unified_model import (
    UnifiedHandModel, setup_lbs_data,
    NUM_VERTS, SHAPE_DIM, SCALE,
    KOREAN_SKELETON_JSON, _JOINT_NAMES_16,
)

PROJECT_ROOT = '/home/user/Documents/Handpose_project'
CHECKPOINT   = os.path.join(PROJECT_ROOT, 'models',  'unified_hand_model.pth')
LANDMARKS    = os.path.join(PROJECT_ROOT, 'data',    'landmarks_2d.npy')
TEMPLATE_OBJ = os.path.join(PROJECT_ROOT, 'meshes',  'AVERAGE_KOREAN_HAND_CENTERED.obj')
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')

N_SAMPLES = 5


def load_model(device):
    import json, pickle
    J_np, weights_np, parents = setup_lbs_data()

    tmesh = trimesh.load(TEMPLATE_OBJ, process=False)
    template_verts = np.array(tmesh.vertices, dtype=np.float32)

    # Build joint_vert_indices (21,) — same logic as main()
    with open(KOREAN_SKELETON_JSON) as f:
        skel = json.load(f)
    joint_vert_indices_16 = np.array(
        [skel[n]['vertex_index'] for n in _JOINT_NAMES_16], dtype=np.int64
    )
    distal_joints = {
        'thumb':  np.array(skel['thumb3']['position_mm'],  dtype=np.float32),
        'index':  np.array(skel['index3']['position_mm'],  dtype=np.float32),
        'middle': np.array(skel['middle3']['position_mm'], dtype=np.float32),
        'ring':   np.array(skel['ring3']['position_mm'],   dtype=np.float32),
        'pinky':  np.array(skel['pinky3']['position_mm'],  dtype=np.float32),
    }
    tip_indices, used = [], set()
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        j3 = distal_joints[finger]
        dists = np.linalg.norm(template_verts - j3, axis=1)
        cands = [v for v in np.argsort(dists)[:50] if v not in used]
        tip_v = int(cands[np.argmax(template_verts[cands, 1])])
        used.add(tip_v)
        tip_indices.append(tip_v)
    joint_vert_indices = np.concatenate(
        [joint_vert_indices_16, np.array(tip_indices, dtype=np.int64)]
    )

    model = UnifiedHandModel(
        template_verts=template_verts,
        weights_kmano=weights_np,
        J=J_np,
        parents=parents,
        latent_dim=SHAPE_DIM,
        scale=SCALE,
        joint_vert_indices=joint_vert_indices,
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch     = ckpt.get('epoch', '?')
    val_pose  = ckpt.get('val_pose_loss', float('nan'))
    print(f'[demo] Checkpoint: epoch={epoch}, best val_pose={val_pose:.4f}')
    return model, np.array(tmesh.faces, dtype=np.int64)


def infer(model, landmarks_flat, device):
    """
    landmarks_flat : (42,) normalised 2D landmarks
    Returns final posed+corrected vertices (V, 3) mm and pose params (48,).
    """
    with torch.no_grad():
        lm = torch.tensor(landmarks_flat, dtype=torch.float32, device=device).unsqueeze(0)

        # Pose
        pred_pose = model.encode_pose(lm)                    # (1, 48)

        # Shape — use mean latent (template shape)
        z = torch.zeros(1, SHAPE_DIM, device=device)
        rest_mm = model.decode_shape(z)                      # (1, V, 3) mm

        # CorrectionNet on top of decoded shape
        pose_corr, shape_corr = model.correction_net(pred_pose, z)  # (1, V, 3) dm each
        rest_corr_mm = rest_mm + (pose_corr + shape_corr) * SCALE   # (1, V, 3) mm

        # LBS
        posed_mm, joints_mm = model.pose_mesh(rest_corr_mm, pred_pose)  # (1,V,3), (1,16,3)

    return (posed_mm.squeeze(0).cpu().numpy(),
            joints_mm.squeeze(0).cpu().numpy(),
            pred_pose.squeeze(0).cpu().numpy())


def render_mesh(ax, verts, faces, joints, elev, azim, title, color='#4499cc'):
    ax.set_facecolor('#111111')
    ax.set_title(title, color='white', fontsize=8, pad=3)

    # Subsample faces for speed
    step = max(1, len(faces) // 1500)
    tri  = verts[faces[::step]]
    poly = Poly3DCollection(tri, alpha=0.85, linewidth=0)
    poly.set_facecolor(color)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    # Joint dots
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=12, c='#ffaa44', zorder=5)

    c = verts.mean(0)
    r = verts.ptp(0).max() * 0.6
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, nargs='+', default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[demo] device={device}')

    model, faces = load_model(device)
    landmarks_all = np.load(LANDMARKS)   # (N, 42)

    if args.samples:
        indices = args.samples[:N_SAMPLES]
    else:
        rng = np.random.default_rng(7)
        indices = rng.choice(len(landmarks_all), size=N_SAMPLES, replace=False).tolist()

    print(f'[demo] Samples: {indices}')

    colors = ['#4499cc', '#55cc55', '#ee5555', '#ffaa44', '#9467bd']

    # 2 rows × N_SAMPLES cols: front + side view
    fig = plt.figure(figsize=(4 * N_SAMPLES, 9))
    fig.patch.set_facecolor('#111111')

    for col, idx in enumerate(indices):
        lm = landmarks_all[idx]
        verts, joints, pose = infer(model, lm, device)

        print(f'  sample {idx}: pose_range=[{pose.min():.3f}, {pose.max():.3f}] '
              f'verts_y=[{verts[:,1].min():.1f}, {verts[:,1].max():.1f}]mm')

        # Export OBJ
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        obj_path = os.path.join(OUTPUTS_DIR, f'unified_demo_{col}.obj')
        mesh.export(obj_path)

        # Front view
        ax_front = fig.add_subplot(2, N_SAMPLES, col + 1, projection='3d')
        render_mesh(ax_front, verts, faces, joints,
                    elev=20, azim=-70,
                    title=f'Sample {idx}\nFront',
                    color=colors[col])

        # Side view
        ax_side = fig.add_subplot(2, N_SAMPLES, N_SAMPLES + col + 1, projection='3d')
        render_mesh(ax_side, verts, faces, joints,
                    elev=10, azim=0,
                    title=f'Sample {idx}\nSide',
                    color=colors[col])

    fig.suptitle('UnifiedHandModel — 5 samples with CorrectionNet',
                 color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    out_png = os.path.join(OUTPUTS_DIR, 'unified_demo.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f'[demo] Saved → {out_png}')
    print(f'[demo] OBJ files → {OUTPUTS_DIR}/unified_demo_0..{N_SAMPLES-1}.obj')


if __name__ == '__main__':
    main()
