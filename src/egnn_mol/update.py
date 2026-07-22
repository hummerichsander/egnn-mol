"""The equivariant per-pair update shared by both backbones.

Both the dense and the sparse backbone reduce, after edge construction, to work over a single
flat axis ``P`` of directed pairs (i <- j): dense neighbor tensors ``(B, N, K, .)`` flatten to
``(P, .)``, sparse edge tensors are already ``(E, .)``. :class:`EquivariantUpdate` owns every
learnable piece and the distance-encoding choice, and exposes three aggregation-free methods on
``(P, .)`` tensors. The backbones only differ in how they gather the per-pair inputs and how they
reduce the per-pair outputs back to nodes.

The update is E(3)-equivariant (features depend only on invariants; coordinate updates are linear
combinations of relative-coordinate vectors). The optional SE(3) triple-product term of the
original sparse backbone is not reproduced here: it needs graph-level aggregation rather than a
pure per-pair op and is unused by current configs. :func:`egnn_mol.geometry.signed_volume` is
provided for adding it later."""

import torch
from torch import Tensor, nn

from .encodings import Encoding, encode_distance, encoding_width
from .nn import MLP, CoorsNorm


def _init_mlp(mlp: nn.Module, gain: float = 1e-3) -> None:
    """Xavier-init every linear in ``mlp`` with a small gain and zero bias."""
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _init_coord_head(
    mlp: nn.Module, gain: float = 1e-3, final_scale: float = 0.01
) -> None:
    """Xavier-init the coordinate head and shrink its last layer for near-identity updates."""
    linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if i == len(linears) - 1:
            m.weight.data *= final_scale
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class EquivariantUpdate(nn.Module):
    """Learnable pieces and per-pair primitives of one EGNN layer, shared across backbones."""

    def __init__(
        self,
        *,
        dim: int,
        encoding: Encoding = "bessel",
        encoding_features: int = 8,
        cutoff: float = 10.0,
        m_dim: int = 16,
        edge_dim: int = 0,
        soft_edges: bool = False,
        norm_feats: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        dropout: float = 0.0,
        coor_weights_clamp_value: float | None = None,
    ) -> None:
        """Build the update.

        :param dim: Node feature dimensionality.
        :param encoding: Radial distance encoding to use.
        :param encoding_features: Number of basis functions / frequency bands for the encoding.
        :param cutoff: Radial length scale of the encoding.
        :param m_dim: Hidden message dimensionality.
        :param edge_dim: Extra per-edge feature dimensionality (0 if none).
        :param soft_edges: Gate messages by a learned scalar in [0, 1].
        :param norm_feats: LayerNorm node features before the node update.
        :param norm_coors: Normalize displacement vectors in the coordinate update.
        :param norm_coors_scale_init: Initial scale of :class:`CoorsNorm`.
        :param dropout: Dropout probability inside the MLPs.
        :param coor_weights_clamp_value: Optional symmetric clamp on coordinate weights."""

        super().__init__()
        self.encoding = encoding
        self.encoding_features = encoding_features
        self.cutoff = cutoff
        self.coor_weights_clamp_value = coor_weights_clamp_value

        dist_width = encoding_width(encoding, encoding_features)
        edge_input_dim = dim * 2 + dist_width + edge_dim

        self.edge_mlp = MLP(
            edge_input_dim,
            edge_input_dim * 2,
            m_dim,
            dropout=dropout,
            final_activation=True,
        )
        self.edge_gate = (
            nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edges else None
        )

        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.node_mlp = MLP(dim + m_dim, dim * 2, dim, dropout=dropout)

        self.coors_mlp = MLP(m_dim, m_dim * 4, 1, dropout=dropout)
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )

        _init_mlp(self.edge_mlp)
        _init_mlp(self.node_mlp)
        _init_coord_head(self.coors_mlp)

    def message(
        self,
        feats_i: Tensor,
        feats_j: Tensor,
        dist: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Per-pair messages.

        :param feats_i: Target-node features (P, dim).
        :param feats_j: Source-node features (P, dim).
        :param dist: True L2 distances (P, 1).
        :param edge_attr: Optional per-pair edge features (P, edge_dim).
        :return: Messages (P, m_dim)."""

        enc = encode_distance(dist, self.encoding, self.encoding_features, self.cutoff)
        parts = [feats_i, feats_j, enc]
        if edge_attr is not None:
            parts.append(edge_attr)
        m_ij = self.edge_mlp(torch.cat(parts, dim=-1))
        if self.edge_gate is not None:
            m_ij = m_ij * self.edge_gate(m_ij)
        return m_ij

    def coord_delta(self, m_ij: Tensor, rel_coors: Tensor) -> Tensor:
        """Per-pair coordinate contributions; the caller masks padded pairs and reduces to nodes.

        :param m_ij: Messages (P, m_dim).
        :param rel_coors: Relative coordinates (P, 3), already minimum-image wrapped.
        :return: Coordinate contributions (P, 3)."""

        weight = self.coors_mlp(m_ij)
        if self.coor_weights_clamp_value is not None:
            c = self.coor_weights_clamp_value
            weight = weight.clamp(min=-c, max=c)
        return weight * self.coors_norm(rel_coors)

    def update_feats(self, feats: Tensor, m_pooled: Tensor) -> Tensor:
        """Residual node-feature update from already-reduced messages.

        :param feats: Node features (num_nodes, dim).
        :param m_pooled: Reduced messages per node (num_nodes, m_dim).
        :return: Updated node features (num_nodes, dim)."""

        normed = self.node_norm(feats)
        return self.node_mlp(torch.cat([normed, m_pooled], dim=-1)) + feats
