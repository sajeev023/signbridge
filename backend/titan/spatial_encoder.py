"""
SIGNBRIDGE TITAN: Spatial Graph Encoder (SGE)
Implements Graph Convolutional Networks (GCN) with anatomical positional embeddings,
scale-invariant joint coordinate representations, and auxiliary biomechanical feature integration.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_topology import (
    NUM_HAND_LANDMARKS,
    build_adjacency_matrix,
    compute_normalized_laplacian,
    get_default_normalized_adjacency,
)

# Kinematic parent for each landmark (root = 0: wrist)
KINEMATIC_PARENT = [
    0,  # 0: wrist -> self
    0, 1, 2, 3,    # thumb: 1<-0, 2<-1, 3<-2, 4<-3
    0, 5, 6, 7,    # index: 5<-0, 6<-5, 7<-6, 8<-7
    0, 9, 10, 11,  # middle: 9<-0, 10<-9, 11<-10, 12<-11
    0, 13, 14, 15, # ring: 13<-0, 14<-13, 15<-14, 16<-15
    0, 17, 18, 19, # pinky: 17<-0, 18<-17, 19<-18, 20<-19
]


def landmarks_to_node_features(
    landmarks: List[Dict[str, float]],
    pos_embed_dim: int = 16,
) -> np.ndarray:
    """
    Converts 21 MediaPipe landmarks to a (21, 6) tensor:
    - 3 normalized coords relative to wrist
    - 3 relative displacement vectors from parent joint
    """
    if len(landmarks) < NUM_HAND_LANDMARKS:
        return np.zeros((NUM_HAND_LANDMARKS, 6), dtype=np.float32)

    coords = np.array([[lm["x"], lm["y"], lm.get("z", 0.0)] for lm in landmarks[:NUM_HAND_LANDMARKS]], dtype=np.float32)
    wrist = coords[0]

    # Palm scale: distance from wrist (0) to middle MCP (9)
    palm_vec = coords[9] - wrist
    palm_scale = np.linalg.norm(palm_vec)
    if palm_scale < 1e-4:
        palm_scale = 1.0

    # Scale-normalized wrist-relative coordinates
    rel_wrist = (coords - wrist) / palm_scale

    # Kinematic parent relative displacements
    parent_coords = coords[KINEMATIC_PARENT]
    rel_parent = (coords - parent_coords) / palm_scale

    # (21, 6) feature matrix
    node_features = np.hstack([rel_wrist, rel_parent])
    
    # Occlusion-aware robust sanitization: clip extreme outliers and replace NaNs
    node_features = np.nan_to_num(node_features, nan=0.0, posinf=5.0, neginf=-5.0)
    node_features = np.clip(node_features, -10.0, 10.0)
    return node_features.astype(np.float32)


class GraphConvolutionLayer(nn.Module):
    """
    Kipf & Welling Graph Convolutional Layer with residual skip and LayerNorm:
    H' = LayerNorm(GELU(A_norm * H * W_neighbor + H * W_self))
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.w_neighbor = nn.Linear(in_features, out_features, bias=False)
        self.w_self = nn.Linear(in_features, out_features, bias=True)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Nodes, In_Features)
            adj_norm: Tensor of shape (Nodes, Nodes) or (Batch, Nodes, Nodes)
        """
        # Neighbor aggregation: A_norm * X
        if adj_norm.dim() == 2:
            h_neighbor = torch.matmul(adj_norm, x)  # (B, N, C)
        else:
            h_neighbor = torch.bmm(adj_norm, x)

        out = self.w_neighbor(h_neighbor) + self.w_self(x)
        out = F.gelu(out)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class SpatialGraphEncoder(nn.Module):
    """
    TITAN Spatial Graph Encoder:
    Encodes 21 hand landmarks into a dense spatial latent vector z_spatial in R^d.
    Incorporates anatomical positional embeddings and auxiliary biomechanical features.
    """
    def __init__(
        self,
        node_in_dim: int = 6,
        pos_embed_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        use_aux_features: bool = True,
        aux_dim: int = 42,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_aux_features = use_aux_features

        # Anatomical node positional embedding (joint identity 0..20)
        self.joint_embed = nn.Embedding(NUM_HAND_LANDMARKS, pos_embed_dim)
        in_dim = node_in_dim + pos_embed_dim

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Stacked Graph Convolution layers
        self.gcn_layers = nn.ModuleList([
            GraphConvolutionLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Readout projection (mean + max pooling)
        self.readout_proj = nn.Linear(hidden_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # Optional auxiliary 42-feature biomechanical pathway
        if use_aux_features:
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
                nn.LayerNorm(out_dim),
            )
            self.fusion_norm = nn.LayerNorm(out_dim)

        # Register default normalized adjacency matrix as buffer
        norm_adj = get_default_normalized_adjacency()
        self.register_buffer("norm_adj", norm_adj)

        # Adaptive data-driven graph topology (2s-AGCN / ST-GCN++):
        # Learned parameter matrix B (21x21) capturing non-local functional dependencies
        # Initialized to zeros for strict backward compatibility with fixed normalized Laplacian
        self.learned_adj = nn.Parameter(torch.zeros(NUM_HAND_LANDMARKS, NUM_HAND_LANDMARKS))

    def forward(
        self,
        node_features: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
        custom_adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (B, 21, 6) normalized coordinates & parent displacements
            aux_features: (B, 42) optional canonical 42-feature biomechanical vectors
            custom_adj: (21, 21) or (B, 21, 21) optional adjacency override
        Returns:
            z_spatial: (B, out_dim) latent representation of static handshape
        """
        B, N, _ = node_features.shape
        base_adj = custom_adj if custom_adj is not None else self.norm_adj
        
        # Effective adjacency matrix: A_eff = A_norm + B_learned
        effective_adj = base_adj + self.learned_adj

        # Joint positional embeddings
        joint_ids = torch.arange(N, device=node_features.device).unsqueeze(0).expand(B, -1)
        j_emb = self.joint_embed(joint_ids)  # (B, N, pos_embed_dim)

        # Combined node features
        h = torch.cat([node_features, j_emb], dim=-1)  # (B, N, 6 + pos_embed_dim)
        h = F.gelu(self.input_proj(h))

        # GCN forward with residual connections using adaptive graph
        for layer in self.gcn_layers:
            h = h + layer(h, effective_adj)

        # Readout: Mean and Max Pooling across all 21 nodes
        h_mean = torch.mean(h, dim=1)  # (B, hidden_dim)
        h_max, _ = torch.max(h, dim=1)  # (B, hidden_dim)
        h_pool = torch.cat([h_mean, h_max], dim=-1)  # (B, hidden_dim * 2)

        z_spatial = self.norm(F.gelu(self.readout_proj(h_pool)))  # (B, out_dim)

        # Auxiliary biomechanical feature fusion if provided
        if self.use_aux_features and aux_features is not None:
            z_aux = self.aux_proj(aux_features)
            z_spatial = self.fusion_norm(z_spatial + z_aux)

        return z_spatial
