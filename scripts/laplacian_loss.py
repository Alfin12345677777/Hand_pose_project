"""
Laplacian Regularizer — identical formulation to DeepHandMesh:

    L_lap = (1/V) * sum_v || v - (1/|N(v)|) * sum_{v' in N(v)} v' ||²

Builds the sparse (V×V) uniform Laplacian matrix L once from mesh topology,
then computes the loss as || L @ vertices ||² at training time.

Usage:
    from laplacian_loss import LaplacianLoss
    lap_loss = LaplacianLoss(faces, num_vertices).to(device)   # once, before training
    loss = lap_loss(vertices)                                   # each forward pass
"""

import torch
import torch.nn as nn
import numpy as np
import trimesh


class LaplacianLoss(nn.Module):
    def __init__(self, faces, num_vertices):
        """
        faces        : (F, 3) numpy array or torch.LongTensor of triangle indices
        num_vertices : V
        """
        super().__init__()

        # ==========================================
        # Build the uniform Laplacian matrix L
        # L = I - D^{-1} A
        # where A[i,j]=1 if edge (i,j) exists, D[i,i]=degree(i)
        # Stored as a sparse COO tensor for fast GPU matmul.
        # ==========================================
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        faces = np.array(faces, dtype=np.int64)

        # Collect all undirected edges from triangle faces
        edges = set()
        for f in faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                edges.add((min(a, b), max(a, b)))

        # Build adjacency: rows/cols of A
        rows, cols = [], []
        for a, b in edges:
            rows += [a, b]
            cols += [b, a]

        # Degree of each vertex
        degree = np.zeros(num_vertices, dtype=np.float32)
        for r in rows:
            degree[r] += 1.0

        # L = I - D^{-1} A  →  L[i,j] = -1/degree[i]  for neighbors j
        #                       L[i,i] = +1
        lap_rows, lap_cols, lap_vals = [], [], []
        for r, c in zip(rows, cols):
            lap_rows.append(r)
            lap_cols.append(c)
            lap_vals.append(-1.0 / degree[r])

        # Diagonal entries (+1)
        for i in range(num_vertices):
            lap_rows.append(i)
            lap_cols.append(i)
            lap_vals.append(1.0)

        indices = torch.tensor([lap_rows, lap_cols], dtype=torch.long)
        values  = torch.tensor(lap_vals, dtype=torch.float32)
        L = torch.sparse_coo_tensor(indices, values,
                                    size=(num_vertices, num_vertices))
        self.register_buffer('L', L)
        self.num_vertices = num_vertices

    def forward(self, vertices):
        """
        vertices : (B, V, 3) or (V, 3)  — raw (un-normalized) vertex positions
        returns  : scalar loss
        """
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)          # (1, V, 3)

        B = vertices.shape[0]
        # L @ V  →  each row is: v_i - mean(neighbors of v_i)
        Lv = torch.stack([
            torch.sparse.mm(self.L, vertices[b])      # (V, 3)
            for b in range(B)
        ])                                             # (B, V, 3)

        # Mean squared norm across all vertices and batch
        return (Lv ** 2).sum(dim=-1).mean()


def build_from_mesh_file(mesh_path):
    """Convenience: load an OBJ and return a LaplacianLoss ready to use."""
    mesh = trimesh.load(mesh_path, process=False)
    return LaplacianLoss(faces=mesh.faces, num_vertices=len(mesh.vertices))
