"""
SIGNBRIDGE TITAN: Spatial Graph Topology & Normalization
Defines anatomical hand skeleton graph, adjacency matrices, and normalized Graph Laplacians.
"""

from typing import List, Tuple
import numpy as np
import torch

# MediaPipe 21 Landmark Indices
NUM_HAND_LANDMARKS = 21

# Anatomical segments (bones)
ANATOMICAL_BONES: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Transverse metacarpal (palm) arch connections
PALM_CONNECTIONS: List[Tuple[int, int]] = [
    (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
]

# Inter-digital spatial interaction edges (closure / pinch / spread)
CROSS_FINGER_EDGES: List[Tuple[int, int]] = [
    (4, 8),   # Thumb tip - Index tip
    (8, 12),  # Index tip - Middle tip
    (12, 16), # Middle tip - Ring tip
    (16, 20), # Ring tip - Pinky tip
    (4, 12),  # Thumb tip - Middle tip
    (5, 17),  # Hand width / palm span
]

# Complete TITAN hand graph edges
ALL_EDGES: List[Tuple[int, int]] = ANATOMICAL_BONES + PALM_CONNECTIONS + CROSS_FINGER_EDGES


def build_adjacency_matrix(
    num_nodes: int = NUM_HAND_LANDMARKS,
    include_cross_finger: bool = True,
    add_self_loops: bool = True,
) -> np.ndarray:
    """Builds symmetric binary adjacency matrix for hand landmarks."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    edges = ANATOMICAL_BONES + PALM_CONNECTIONS
    if include_cross_finger:
        edges = edges + CROSS_FINGER_EDGES

    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    if add_self_loops:
        np.fill_diagonal(adj, 1.0)

    return adj


def compute_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    Computes Kipf-Welling symmetric normalized adjacency:
    D_tilde^{-1/2} * A_tilde * D_tilde^{-1/2}
    """
    # Degree matrix
    degrees = np.sum(adj, axis=1)
    degrees_inv_sqrt = np.zeros_like(degrees)
    pos = degrees > 0
    degrees_inv_sqrt[pos] = np.power(degrees[pos], -0.5)
    d_mat_inv_sqrt = np.diag(degrees_inv_sqrt)

    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return norm_adj.astype(np.float32)


def get_default_normalized_adjacency(device: str = "cpu") -> torch.Tensor:
    """Returns normalized adjacency as PyTorch tensor."""
    adj = build_adjacency_matrix(add_self_loops=True)
    norm_adj = compute_normalized_laplacian(adj)
    return torch.tensor(norm_adj, dtype=torch.float32, device=device)
