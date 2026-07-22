import torch
from torch import Tensor, nn

from .encodings import Encoding, encode_distance, encoding_width
from .nn import MLP, PosNorm


def _init_mlp(mlp: nn.Module, gain: float = 1e-3) -> None:
    """Xavier-init every linear in ``mlp`` with a small gain and zero bias."""
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _init_pos_head(
    mlp: nn.Module, gain: float = 1e-3, final_scale: float = 0.01
) -> None:
    """Xavier-init the position head and shrink its last layer for near-identity updates."""
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
        norm_x: bool = False,
        norm_pos: bool = False,
        norm_pos_scale_init: float = 1e-2,
        dropout: float = 0.0,
        pos_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
    ) -> None:
        """Build the update.

        :param dim: Node feature dimensionality.
        :param encoding: Radial distance encoding to use.
        :param encoding_features: Number of basis functions / frequency bands for the encoding.
        :param cutoff: Radial length scale of the encoding.
        :param m_dim: Hidden message dimensionality.
        :param edge_dim: Extra per-edge feature dimensionality (0 if none).
        :param soft_edges: Gate messages by a learned scalar in [0, 1].
        :param norm_x: LayerNorm node features before the node update.
        :param norm_pos: Normalize displacement vectors in the position update.
        :param norm_pos_scale_init: Initial scale of :class:`PosNorm`.
        :param dropout: Dropout probability inside the MLPs.
        :param pos_weights_clamp_value: Optional symmetric clamp on position weights.
        :param tripp_num_layers: Depth of the triple-product MLP; > 0 turns on the SE(3)
            chirality term (0 keeps the update E(3)-equivariant)."""

        super().__init__()
        self.encoding = encoding
        self.encoding_features = encoding_features
        self.cutoff = cutoff
        self.pos_weights_clamp_value = pos_weights_clamp_value
        self.tripp = tripp_num_layers > 0

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

        self.node_norm = nn.LayerNorm(dim) if norm_x else nn.Identity()
        self.node_mlp = MLP(dim + m_dim, dim * 2, dim, dropout=dropout)

        # With the triple-product term the position head also sees the chirality scalar of
        # both endpoints, so its input widens by 2.
        self.pos_mlp = MLP(
            m_dim + (2 if self.tripp else 0), m_dim * 4, 1, dropout=dropout
        )
        self.pos_norm = (
            PosNorm(scale_init=norm_pos_scale_init) if norm_pos else nn.Identity()
        )
        self.triple_mlp = (
            MLP(m_dim, m_dim, 3, num_layers=tripp_num_layers, dropout=dropout)
            if self.tripp
            else None
        )

        _init_mlp(self.edge_mlp)
        _init_mlp(self.node_mlp)
        _init_pos_head(self.pos_mlp)
        if self.triple_mlp is not None:
            _init_mlp(self.triple_mlp)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        dist: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Per-pair messages.

        :param x_i: Target-node features (P, dim).
        :param x_j: Source-node features (P, dim).
        :param dist: True L2 distances (P, 1).
        :param edge_attr: Optional per-pair edge features (P, edge_dim).
        :return: Messages (P, m_dim)."""

        enc = encode_distance(dist, self.encoding, self.encoding_features, self.cutoff)
        parts = [x_i, x_j, enc]
        if edge_attr is not None:
            parts.append(edge_attr)
        m_ij = self.edge_mlp(torch.cat(parts, dim=-1))
        if self.edge_gate is not None:
            m_ij = m_ij * self.edge_gate(m_ij)
        return m_ij

    def normalize_rel(self, rel_pos: Tensor) -> Tensor:
        """Direction-normalize relative positions (identity if ``norm_pos`` is off).

        :param rel_pos: Relative positions (P, 3), already minimum-image wrapped.
        :return: Normalized relative positions (P, 3)."""

        return self.pos_norm(rel_pos)

    def triple_abc(self, m_ij: Tensor) -> Tensor:
        """Per-pair scalar weights for the three chirality vector fields (SE(3) term only).

        :param m_ij: Messages (P, m_dim).
        :return: Three scalar weights per pair (P, 3)."""

        return self.triple_mlp(m_ij)

    def pos_weight(
        self, m_ij: Tensor, chi_i: Tensor | None = None, chi_j: Tensor | None = None
    ) -> Tensor:
        """Per-pair scalar position weight.

        With the triple-product term the per-node chirality scalars of both endpoints are
        appended to the message before the position head.

        :param m_ij: Messages (P, m_dim).
        :param chi_i: Chirality scalar of the target node, gathered per pair (P, 1); only with SE(3).
        :param chi_j: Chirality scalar of the source node, gathered per pair (P, 1); only with SE(3).
        :return: Position weights (P, 1)."""

        inp = torch.cat([m_ij, chi_i, chi_j], dim=-1) if self.tripp else m_ij
        weight = self.pos_mlp(inp)
        if self.pos_weights_clamp_value is not None:
            c = self.pos_weights_clamp_value
            weight = weight.clamp(min=-c, max=c)
        return weight

    def update_x(self, x: Tensor, m_pooled: Tensor) -> Tensor:
        """Residual node-feature update from already-reduced messages.

        :param x: Node features (num_nodes, dim).
        :param m_pooled: Reduced messages per node (num_nodes, m_dim).
        :return: Updated node features (num_nodes, dim)."""

        normed = self.node_norm(x)
        return self.node_mlp(torch.cat([normed, m_pooled], dim=-1)) + x
