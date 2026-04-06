"""
visualize_lbs_weights.py
========================
Render the Korean hand template with vertices colored by dominant LBS joint.
Four views: front, back, left side, right side.

Output: outputs/lbs_weights_visualization.png

Usage:
  my_venv/bin/python scripts/visualize_lbs_weights.py
"""

import os
import sys
import json
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

PROJECT_ROOT = '/home/user/Documents/Handpose_project'
KOREAN_TMPL  = os.path.join(PROJECT_ROOT, 'meshes', 'AVERAGE_KOREAN_HAND_CENTERED.obj')
KOREAN_SKEL  = os.path.join(PROJECT_ROOT, 'models', 'korean_hand_skeleton.json')
OUTPUT_PNG   = os.path.join(PROJECT_ROOT, 'outputs', 'lbs_weights_visualization.png')

JOINT_NAMES_16 = [
    'wrist',
    'index1', 'index2', 'index3',
    'middle1','middle2','middle3',
    'pinky1', 'pinky2', 'pinky3',
    'ring1',  'ring2',  'ring3',
    'thumb1', 'thumb2', 'thumb3',
]

# One distinct color per joint — grouped by finger for easy reading
JOINT_COLORS = [
    '#888888',  # J00 wrist       — grey
    '#1f77b4',  # J01 index1      — blue (light)
    '#4499cc',  # J02 index2
    '#77bbee',  # J03 index3
    '#2ca02c',  # J04 middle1     — green
    '#55cc55',  # J05 middle2
    '#88ee88',  # J06 middle3
    '#9467bd',  # J07 pinky1      — purple
    '#bb88dd',  # J08 pinky2
    '#ddbbff',  # J09 pinky3
    '#d62728',  # J10 ring1       — red
    '#ee5555',  # J11 ring2
    '#ff9999',  # J12 ring3
    '#ff7f0e',  # J13 thumb1      — orange
    '#ffaa44',  # J14 thumb2
    '#ffcc88',  # J15 thumb3
]


def seg_dist(v, a, b):
    """Distance from each vertex v to line segment a–b."""
    ab = b - a
    t  = np.clip(((v - a) @ ab) / (np.dot(ab, ab) + 1e-8), 0, 1)
    return np.linalg.norm(v - (a + t[:, None] * ab), axis=1)


def compute_bone_weights(verts, J, parents, sigma=15.0):
    W = np.zeros((len(verts), 16), dtype=np.float64)
    for j, p in enumerate(parents):
        if p < 0:
            d = np.linalg.norm(verts - J[j], axis=1)
        else:
            d = seg_dist(verts, J[p], J[j])
        W[:, j] = np.exp(-d ** 2 / (2 * sigma ** 2))
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-8)
    return W.astype(np.float32)


def render_view(ax, verts, faces, vert_colors, joint_positions,
                elev, azim, title):
    """Render the mesh from one viewpoint using Poly3DCollection."""
    ax.set_title(title, fontsize=9, pad=4)

    # Build face vertex arrays and face colors (mean of 3 corner colors)
    tri_verts = verts[faces]                          # (F, 3, 3)
    face_rgb  = vert_colors[faces].mean(axis=1)      # (F, 3)  mean corner colour

    poly = Poly3DCollection(tri_verts, alpha=0.92, linewidth=0)
    poly.set_facecolor(face_rgb)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    # Joint positions as scatter
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
               c=JOINT_COLORS, s=40, zorder=5, edgecolors='k', linewidths=0.5)

    # Axis limits
    centre = verts.mean(axis=0)
    r = max(verts.max(axis=0) - verts.min(axis=0)) * 0.55
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] - r, centre[1] + r)
    ax.set_zlim(centre[2] - r, centre[2] + r)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X', fontsize=7); ax.set_ylabel('Y', fontsize=7); ax.set_zlabel('Z', fontsize=7)
    ax.tick_params(labelsize=6)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

    # ── Load mesh ──────────────────────────────────────────────────────────
    print('Loading Korean template mesh ...')
    mesh  = trimesh.load(KOREAN_TMPL, process=False)
    verts = np.array(mesh.vertices, dtype=np.float32)   # (V, 3) mm
    faces = np.array(mesh.faces,    dtype=np.int64)     # (F, 3)
    print(f'  {len(verts)} vertices, {len(faces)} faces')

    # ── Load skeleton ──────────────────────────────────────────────────────
    print('Loading skeleton ...')
    with open(KOREAN_SKEL) as f:
        skel = json.load(f)
    J = np.array([skel[n]['position_mm'] for n in JOINT_NAMES_16], dtype=np.float32)

    # Parents: MANO kintree order
    import pickle
    MANO_PKL = os.path.join(PROJECT_ROOT, 'models', 'MANO_RIGHT.pkl')
    with open(MANO_PKL, 'rb') as f:
        mano = pickle.load(f, encoding='latin1')
    kintree = mano['kintree_table']
    parents = [int(kintree[0, j]) for j in range(16)]
    parents[0] = -1

    # ── Compute weights & dominant joint ──────────────────────────────────
    print('Computing bone weights (sigma=8 mm) ...')
    W = compute_bone_weights(verts, J, parents, sigma=8.0)
    dominant = W.argmax(axis=1)   # (V,) int in [0, 15]

    # Print distribution (same as training output)
    print('\nDominant joint distribution:')
    for j, name in enumerate(JOINT_NAMES_16):
        count = (dominant == j).sum()
        print(f'  J{j:02d} {name:<10}: {count:5d} verts')

    # Vertex RGB colors from dominant joint
    vert_colors = np.array([
        matplotlib.colors.to_rgb(JOINT_COLORS[d]) for d in dominant
    ], dtype=np.float32)   # (V, 3)

    # ── Four-view plot ─────────────────────────────────────────────────────
    print('\nRendering four-view figure ...')
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a1a')

    views = [
        (20,  -90, 'Front'),
        (20,   90, 'Back'),
        (20,    0, 'Right side'),
        (20,  180, 'Left side'),
    ]

    axes = []
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        ax.set_facecolor('#1a1a1a')
        render_view(ax, verts, faces, vert_colors, J, elev, azim, title)
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(colors='white')
        axes.append(ax)

    # Legend
    patches = [
        mpatches.Patch(color=JOINT_COLORS[j],
                       label=f'J{j:02d} {JOINT_NAMES_16[j]}')
        for j in range(16)
    ]
    fig.legend(handles=patches, loc='lower center', ncol=8,
               fontsize=7, framealpha=0.2,
               facecolor='#333333', edgecolor='white',
               labelcolor='white',
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle('Korean Hand Template — LBS Dominant Joint per Vertex  (σ = 8 mm)',
                 color='white', fontsize=13, y=0.97)

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'\nSaved → {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
