"""
eda_grasping_data.py
====================
Exploratory Data Analysis for the 손·팔 협조에 의한 파지-조작 동작 데이터
(Hand-Arm Coordination Grasping-Manipulation Motion Data)

Produces: outputs/eda_grasping_data.png
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from collections import Counter, defaultdict
import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLE_ROOT   = '/media/user/My Passport/Sample'
IMG_DIR       = os.path.join(SAMPLE_ROOT, '01.원천데이터')
JSON_DIR      = os.path.join(SAMPLE_ROOT, '02.라벨링데이터')
OUTPUT_PATH   = '/home/user/Documents/Handpose_project/outputs/eda_grasping_data.png'

# 21-point hand skeleton connections (MediaPipe convention)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm
]
FINGER_COLORS = ['#FF6B6B','#FFA07A','#98D8C8','#87CEEB','#DDA0DD']

# ---------------------------------------------------------------------------
# Load all JSON annotations
# ---------------------------------------------------------------------------
def load_all_annotations(json_dir):
    json_files = sorted(glob.glob(os.path.join(json_dir, '**', '*.json'), recursive=True))
    print(f'[EDA] Found {len(json_files)} JSON files.')

    records = []
    for path in json_files:
        try:
            with open(path) as f:
                d = json.load(f)

            img_id   = d['image']['image_ID']
            width    = int(d['image']['width'])
            height   = int(d['image']['height'])
            mission  = d['mission']['name']
            obj_name = d['object']['object_name']
            light    = int(d['light_source']['light_degree'])

            gd       = d['gesture']['hand_gesture_data']
            n_fingers = gd['grasp_finger_count']
            kp2d_flat = gd['hand_keypoints']['2D']
            kp3d_flat = gd['hand_keypoints']['3D']
            vis_flat  = gd['hand_keypoints']['visibility']
            intrinsic = d['object']['intrinsic']

            kp2d = np.array(kp2d_flat).reshape(-1, 2)   # (21, 2) pixels
            kp3d = np.array(kp3d_flat).reshape(-1, 3)   # (21, 3) normalized
            vis  = np.array(vis_flat)                    # (21,)

            # Corresponding image path
            stem = os.path.splitext(os.path.basename(path))[0]
            img_path = path.replace('02.라벨링데이터', '01.원천데이터').replace('.json', '.jpg')

            records.append({
                'path':     path,
                'img_path': img_path,
                'img_id':   img_id,
                'width':    width,
                'height':   height,
                'mission':  mission,
                'object':   obj_name,
                'light':    light,
                'n_fingers': n_fingers,
                'kp2d':     kp2d,
                'kp3d':     kp3d,
                'vis':      vis,
                'intrinsic': intrinsic,
            })
        except Exception as e:
            print(f'  [WARN] {path}: {e}')

    print(f'[EDA] Loaded {len(records)} valid records.')
    return records

# ---------------------------------------------------------------------------
# Draw skeleton on image
# ---------------------------------------------------------------------------
def draw_skeleton(ax, img, kp2d, vis, title=''):
    ax.imshow(img)
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis('off')

    # Bones
    finger_ranges = [(0,4),(5,8),(9,12),(13,16),(17,20)]
    for ci, (conn) in enumerate(HAND_CONNECTIONS):
        i, j = conn
        if vis[i] >= 0 and vis[j] >= 0:
            color_idx = 0
            for fi, (start, end) in enumerate(finger_ranges):
                if i >= start or j <= end:
                    color_idx = fi
            ax.plot([kp2d[i,0], kp2d[j,0]], [kp2d[i,1], kp2d[j,1]],
                    '-', color='white', lw=1.5, alpha=0.7)

    # Joints
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 21))
    for k in range(21):
        if vis[k] >= 0:
            ax.scatter(kp2d[k,0], kp2d[k,1], c=[colors[k]], s=15, zorder=5)

# ---------------------------------------------------------------------------
# Main EDA
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    records = load_all_annotations(JSON_DIR)
    if not records:
        print('[EDA] No records found. Check path.')
        return

    # ── Aggregate stats ──────────────────────────────────────────────────────
    missions   = Counter(r['mission']  for r in records)
    objects    = Counter(r['object']   for r in records)
    lights     = Counter(r['light']    for r in records)
    n_fingers  = Counter(r['n_fingers'] for r in records)
    all_vis    = np.concatenate([r['vis'] for r in records])  # (N*21,)

    all_kp2d   = np.stack([r['kp2d'] for r in records])  # (N, 21, 2)
    all_kp3d   = np.stack([r['kp3d'] for r in records])  # (N, 21, 3)
    all_depths = all_kp3d[:, :, 2].ravel()               # all z values

    # Normalized 2D (0-1)
    widths  = np.array([r['width']  for r in records])
    heights = np.array([r['height'] for r in records])
    kp2d_norm_x = all_kp2d[:, :, 0] / widths[:, None]
    kp2d_norm_y = all_kp2d[:, :, 1] / heights[:, None]

    # Wrist (joint 0) position heatmap
    wrist_x = kp2d_norm_x[:, 0]
    wrist_y = kp2d_norm_y[:, 0]

    print(f'\n[EDA] === Dataset Statistics ===')
    print(f'  Total samples     : {len(records)}')
    print(f'  Missions          : {dict(missions)}')
    print(f'  Objects           : {dict(objects)}')
    print(f'  Lighting levels   : {dict(lights)}')
    print(f'  Finger counts     : {dict(n_fingers)}')
    print(f'  Visibility -1/0/1 : {(all_vis==-1).sum()} / {(all_vis==0).sum()} / {(all_vis==1).sum()}')
    print(f'  Depth Z range     : {all_depths.min():.4f} — {all_depths.max():.4f} (mean {all_depths.mean():.4f})')
    print(f'  2D x norm range   : {kp2d_norm_x.min():.3f} — {kp2d_norm_x.max():.3f}')
    print(f'  2D y norm range   : {kp2d_norm_y.min():.3f} — {kp2d_norm_y.max():.3f}')

    # ── Per-keypoint visibility ───────────────────────────────────────────────
    JOINT_NAMES = [
        'Wrist','Thumb1','Thumb2','Thumb3','Thumb4',
        'Index1','Index2','Index3','Index4',
        'Mid1','Mid2','Mid3','Mid4',
        'Ring1','Ring2','Ring3','Ring4',
        'Pinky1','Pinky2','Pinky3','Pinky4',
    ]
    vis_stack = np.stack([r['vis'] for r in records])  # (N, 21)
    vis_rate  = (vis_stack >= 0).mean(axis=0)           # visible = 0 or 1

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#1a1a2e')
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    title_kw = dict(fontsize=10, color='white', fontweight='bold', pad=6)
    label_kw = dict(color='#aaaacc', fontsize=8)

    # ── Row 0: Sample images with skeleton ───────────────────────────────────
    sample_indices = np.linspace(0, len(records)-1, 4, dtype=int)
    for col, idx in enumerate(sample_indices):
        r = records[idx]
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#0d0d1a')
        if os.path.exists(r['img_path']):
            img = cv2.cvtColor(cv2.imread(r['img_path']), cv2.COLOR_BGR2RGB)
            draw_skeleton(ax, img, r['kp2d'], r['vis'],
                          title=f"{r['object']}\nLight={r['light']} | Fingers={r['n_fingers']}")
        else:
            ax.text(0.5, 0.5, 'Image\nnot found', ha='center', va='center',
                    color='white', fontsize=9, transform=ax.transAxes)
            ax.set_facecolor('#111')
            ax.axis('off')

    # ── Row 1-col 0: Mission distribution ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('#0d0d1a')
    missions_short = {k[:8]+'…' if len(k)>8 else k: v for k,v in missions.items()}
    bars = ax.bar(range(len(missions_short)), list(missions_short.values()),
                  color='#4ecdc4', edgecolor='none')
    ax.set_xticks(range(len(missions_short)))
    ax.set_xticklabels(list(missions_short.keys()), rotation=30, ha='right', **label_kw)
    ax.set_title('Mission Types', **title_kw)
    ax.set_ylabel('Count', **label_kw)
    ax.tick_params(colors='#aaaacc')
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0d0d1a')
    for spine in ax.spines.values(): spine.set_color('#333355')

    # ── Row 1-col 1: Lighting distribution ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('#0d0d1a')
    light_keys = sorted(lights.keys())
    light_vals = [lights[k] for k in light_keys]
    cmap = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(light_keys)))
    ax.bar(light_keys, light_vals, color=cmap, edgecolor='none')
    ax.set_title('Lighting Levels', **title_kw)
    ax.set_xlabel('Light Degree', **label_kw)
    ax.set_ylabel('Count', **label_kw)
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values(): spine.set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Row 1-col 2: Finger count ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('#0d0d1a')
    fc_keys = sorted(n_fingers.keys())
    fc_vals = [n_fingers[k] for k in fc_keys]
    ax.bar([str(k) for k in fc_keys], fc_vals, color='#ff6b6b', edgecolor='none')
    ax.set_title('Grasp Finger Count', **title_kw)
    ax.set_xlabel('Fingers Used', **label_kw)
    ax.set_ylabel('Count', **label_kw)
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values(): spine.set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Row 1-col 3: Object distribution ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor('#0d0d1a')
    obj_items = objects.most_common(8)
    obj_names = [o[:10]+'…' if len(o)>10 else o for o,_ in obj_items]
    obj_counts = [c for _,c in obj_items]
    ax.barh(range(len(obj_names)), obj_counts, color='#a29bfe', edgecolor='none')
    ax.set_yticks(range(len(obj_names)))
    ax.set_yticklabels(obj_names, **label_kw)
    ax.set_title('Object Types (top 8)', **title_kw)
    ax.set_xlabel('Count', **label_kw)
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values(): spine.set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Row 2-col 0-1: Keypoint visibility per joint ─────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    ax.set_facecolor('#0d0d1a')
    colors_vis = plt.cm.RdYlGn(vis_rate)
    bars = ax.bar(range(21), vis_rate * 100, color=colors_vis, edgecolor='none')
    ax.set_xticks(range(21))
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=7, color='#aaaacc')
    ax.set_ylabel('Visibility Rate (%)', **label_kw)
    ax.set_title('Per-Joint Visibility Rate', **title_kw)
    ax.set_ylim(0, 110)
    ax.axhline(100, color='#4ecdc4', lw=1, linestyle='--', alpha=0.5)
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values(): spine.set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(vis_rate):
        ax.text(i, v*100+1.5, f'{v*100:.0f}%', ha='center', fontsize=5.5, color='white', rotation=90)

    # ── Row 2-col 2-3: Depth (Z) distribution ────────────────────────────────
    ax = fig.add_subplot(gs[2, 2:])
    ax.set_facecolor('#0d0d1a')
    ax.hist(all_depths, bins=60, color='#fd79a8', edgecolor='none', alpha=0.85)
    ax.axvline(all_depths.mean(), color='white', lw=1.5, linestyle='--',
               label=f'Mean={all_depths.mean():.4f}')
    ax.set_title('3D Joint Depth (Z) Distribution', **title_kw)
    ax.set_xlabel('Normalized Depth Z', **label_kw)
    ax.set_ylabel('Count', **label_kw)
    ax.legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e', edgecolor='none')
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values(): spine.set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Row 3-col 0-1: Wrist position heatmap ────────────────────────────────
    ax = fig.add_subplot(gs[3, :2])
    ax.set_facecolor('#0d0d1a')
    h2d, xedge, yedge = np.histogram2d(wrist_x, wrist_y, bins=40,
                                         range=[[0,1],[0,1]])
    im = ax.imshow(h2d.T, origin='upper', cmap='hot', aspect='auto',
                   extent=[0,1,1,0])
    ax.set_title('Wrist Position Heatmap (normalized image space)', **title_kw)
    ax.set_xlabel('X (left→right)', **label_kw)
    ax.set_ylabel('Y (top→bottom)', **label_kw)
    plt.colorbar(im, ax=ax, label='Count').ax.yaxis.label.set_color('#aaaacc')
    ax.tick_params(colors='#aaaacc')

    # ── Row 3-col 2-3: All 2D keypoint scatter ───────────────────────────────
    ax = fig.add_subplot(gs[3, 2:])
    ax.set_facecolor('#0d0d1a')
    finger_joint_groups = {
        'Thumb':  [1,2,3,4],
        'Index':  [5,6,7,8],
        'Middle': [9,10,11,12],
        'Ring':   [13,14,15,16],
        'Pinky':  [17,18,19,20],
        'Wrist':  [0],
    }
    fcolors = ['#FF6B6B','#FFA07A','#98D8C8','#87CEEB','#DDA0DD','#FFD700']
    for (fname, joints), fc in zip(finger_joint_groups.items(), fcolors):
        xs = kp2d_norm_x[:, joints].ravel()
        ys = kp2d_norm_y[:, joints].ravel()
        ax.scatter(xs, ys, c=fc, s=0.5, alpha=0.3, label=fname)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # flip y so image orientation matches
    ax.set_title('All 2D Keypoints (normalized)', **title_kw)
    ax.set_xlabel('X (0=left, 1=right)', **label_kw)
    ax.set_ylabel('Y (0=top, 1=bottom)', **label_kw)
    leg = ax.legend(fontsize=7, markerscale=8, labelcolor='white',
                    facecolor='#0d0d1a', edgecolor='#333355', loc='upper right')
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values(): spine.set_color('#333355')

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        '손·팔 협조에 의한 파지-조작 동작 데이터  —  EDA\n'
        f'{len(records):,} samples  |  21 keypoints (2D + depth)  |  {len(missions)} mission types  |  {len(objects)} object types',
        fontsize=13, color='white', fontweight='bold', y=0.98
    )

    plt.savefig(OUTPUT_PATH, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n[EDA] Plot saved to {OUTPUT_PATH}')

    # ── Print training integration summary ───────────────────────────────────
    print('\n[EDA] === Training Integration Summary ===')
    print(f'  Input to PoseEncoder : hand_keypoints.2D  → 21×2 = 42 values (normalize by image size)')
    print(f'  Joint supervision    : hand_keypoints.3D  → 21×3 (x=px/W, y=py/H, z=depth)')
    print(f'  Depth Z range        : {all_depths.min():.5f} – {all_depths.max():.5f}')
    print(f'  Avg visibility rate  : {vis_rate.mean()*100:.1f}%')
    print(f'  Samples in this set  : {len(records):,}')
    print(f'  Full dataset scale   : ~5,000,000 samples')
    print(f'  Key advantage        : 150× more data than FreiHAND ({len(records)} sample)')
    print(f'\n  !! NOTE: No MANO pose_params (48D) — need reprojection loss instead of L_pose')
    print(f'  !! Plan: 2D kpts → PoseEncoder → LBS → project joints → compare to GT 2D')


if __name__ == '__main__':
    main()
