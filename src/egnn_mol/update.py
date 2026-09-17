import torch
from torch import Tensor, nn

from .encodings import Encoding, encode_distance, encoding_width
from .nn import MLP, DisplacementNorm


def _init_mlp(mlp: nn.Module, gain: float = 1e-3) -> None:
    """Xavier-init every linear in ``mlp`` with a small gain and zero bias."""
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _init_x_head(mlp: nn.Module, gain: float = 1e-3, final_scale: float = 0.01) -> None:
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
        norm_h_node: bool = False,
        norm_displacement: bool = False,
        norm_displacement_scale_init: float = 1.0,
        dropout: float = 0.0,
        x_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
        mlp_depth: int = 1,
        vector_channels: int = 0,
    ) -> None:
        """Build the update.

        :param dim: Node feature dimensionality.
        :param encoding: Radial distance encoding to use.
        :param encoding_features: Number of basis functions / frequency bands for the encoding.
        :param cutoff: Radial length scale of the encoding.
        :param m_dim: Hidden message dimensionality.
        :param edge_dim: Extra per-edge feature dimensionality (0 if none).
        :param soft_edges: Gate messages by a learned scalar in [0, 1].
        :param norm_h_node: LayerNorm node features before the node update.
        :param norm_displacement: Normalize displacement vectors in the position update.
        :param norm_displacement_scale_init: Initial scale of :class:`DisplacementNorm`.
        :param dropout: Dropout probability inside the MLPs.
        :param x_weights_clamp_value: Optional symmetric clamp on position weights.
        :param tripp_num_layers: Depth of the triple-product MLP; > 0 turns on the SE(3)
            chirality term (0 keeps the update E(3)-equivariant).
        :param mlp_depth: Number of hidden blocks in the edge, node and position MLPs. It buys
            capacity without widening the receptive field, so it leaves the Jacobian sparsity
            pattern untouched.
        :param vector_channels: Number of PaiNN-style equivariant vector features per node; 0
            keeps the plain scalar update. Each channel is a 3-vector that rotates with the
            system and persists across layers, contracted to rotation-invariant scalars that
            feed the node update. A layer still reads one hop, so the sparsity pattern is
            unchanged; the contractions are dot products only, which keeps the update
            reflection-equivariant like the rest of the E(3) path."""

        super().__init__()
        self.encoding = encoding
        self.encoding_features = encoding_features
        self.cutoff = cutoff
        self.x_weights_clamp_value = x_weights_clamp_value
        self.tripp = tripp_num_layers > 0
        self.vector_channels = vector_channels

        dist_width = encoding_width(encoding, encoding_features)
        edge_input_dim = dim * 2 + dist_width + edge_dim

        self.edge_mlp = MLP(
            edge_input_dim,
            edge_input_dim * 2,
            m_dim,
            num_layers=mlp_depth,
            dropout=dropout,
            final_activation=True,
        )
        self.edge_gate = (
            nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edges else None
        )

        self.node_norm = nn.LayerNorm(dim) if norm_h_node else nn.Identity()
        # the vector channels reach the scalars only as their invariants, two per channel.
        self.node_mlp = MLP(
            dim + m_dim + 2 * vector_channels,
            dim * 2,
            dim,
            num_layers=mlp_depth,
            dropout=dropout,
        )

        # With the triple-product term the position head also sees the chirality scalar of
        # both endpoints, so its input widens by 2.
        self.x_mlp = MLP(
            m_dim + (2 if self.tripp else 0),
            m_dim * 4,
            1,
            num_layers=mlp_depth,
            dropout=dropout,
        )
        self.displacement_norm = (
            DisplacementNorm(scale_init=norm_displacement_scale_init)
            if norm_displacement
            else nn.Identity()
        )
        self.triple_mlp = (
            MLP(m_dim, m_dim, 3, num_layers=tripp_num_layers, dropout=dropout)
            if self.tripp
            else None
        )

        if vector_channels:
            self.vec_message = MLP(
                m_dim,
                m_dim * 2,
                2 * vector_channels,
                num_layers=mlp_depth,
                dropout=dropout,
            )
            self.vec_mix = nn.Linear(vector_channels, 2 * vector_channels, bias=False)
            self.vec_gate = MLP(
                dim + 2 * vector_channels,
                m_dim * 2,
                vector_channels,
                num_layers=mlp_depth,
                dropout=dropout,
            )
        else:
            self.vec_message = self.vec_mix = self.vec_gate = None

        _init_mlp(self.edge_mlp)
        _init_mlp(self.node_mlp)
        _init_x_head(self.x_mlp)
        if self.triple_mlp is not None:
            _init_mlp(self.triple_mlp)
        if vector_channels:
            _init_mlp(self.vec_message)
            _init_mlp(self.vec_gate)
            nn.init.xavier_uniform_(self.vec_mix.weight, gain=1e-3)

    def message(
        self,
        h_node_i: Tensor,
        h_node_j: Tensor,
        dist: Tensor,
        h_edge: Tensor | None = None,
    ) -> Tensor:
        """Per-pair messages.

        :param h_node_i: Target-node features (P, dim).
        :param h_node_j: Source-node features (P, dim).
        :param dist: True L2 distances (P, 1).
        :param h_edge: Optional per-pair edge features (P, edge_dim).
        :return: Messages (P, m_dim)."""

        enc = encode_distance(dist, self.encoding, self.encoding_features, self.cutoff)
        parts = [h_node_i, h_node_j, enc]
        if h_edge is not None:
            parts.append(h_edge)
        m_ij = self.edge_mlp(torch.cat(parts, dim=-1))
        if self.edge_gate is not None:
            m_ij = m_ij * self.edge_gate(m_ij)
        return m_ij

    def normalize_rel(self, rel_x: Tensor) -> Tensor:
        """Direction-normalize relative positions (identity if ``norm_displacement`` is off).

        :param rel_x: Relative positions (P, 3), already minimum-image wrapped.
        :return: Normalized relative positions (P, 3)."""

        return self.displacement_norm(rel_x)

    def triple_abc(self, m_ij: Tensor) -> Tensor:
        """Per-pair scalar weights for the three chirality vector fields (SE(3) term only).

        :param m_ij: Messages (P, m_dim).
        :return: Three scalar weights per pair (P, 3)."""

        return self.triple_mlp(m_ij)

    def x_weight(
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
        weight = self.x_mlp(inp)
        if self.x_weights_clamp_value is not None:
            c = self.x_weights_clamp_value
            weight = weight.clamp(min=-c, max=c)
        return weight

    def vector_coefficients(self, m_ij: Tensor) -> tuple[Tensor, Tensor]:
        """Per-pair scalars weighting a neighbour's vectors and the edge's own direction.

        :param m_ij: Messages (P, m_dim).
        :return: Weights for the neighbour's vectors and for the direction, each
            (P, vector_channels)."""

        a_vv, a_vs = self.vec_message(m_ij).chunk(2, dim=-1)
        return a_vv, a_vs

    def vector_invariants(self, vec: Tensor) -> tuple[Tensor, Tensor]:
        """Channel-mix the vectors and contract them to rotation-invariant scalars.

        Dot products only: a cross or triple product would be a pseudoscalar and would drop the
        update from E(3) to SE(3), which is what ``tripp_num_layers`` is for.

        :param vec: Vector features (num_nodes, vector_channels, 3).
        :return: The mixed vectors (num_nodes, vector_channels, 3) and the invariants
            (num_nodes, 2 * vector_channels)."""

        mixed = self.vec_mix(vec.transpose(-2, -1)).transpose(-2, -1)
        u, v = mixed.chunk(2, dim=-2)

        # squared norms, not norms: a norm has no derivative where a channel is exactly zero,
        # which is where every channel starts, and the divergence needs a C^1 field.
        return u, torch.cat([(u * v).sum(-1), (v * v).sum(-1)], dim=-1)

    def gate_vec(self, h_node: Tensor, invariants: Tensor) -> Tensor:
        """Node-local gate on the mixed vectors, the one place the scalars steer them.

        :param h_node: Node features (num_nodes, dim).
        :param invariants: Vector invariants (num_nodes, 2 * vector_channels).
        :return: One scalar per node and channel (num_nodes, vector_channels)."""

        return self.vec_gate(torch.cat([h_node, invariants], dim=-1))

    def update_h_node(
        self, h_node: Tensor, m_pooled: Tensor, invariants: Tensor | None = None
    ) -> Tensor:
        """Residual node-feature update from already-reduced messages.

        :param h_node: Node features (num_nodes, dim).
        :param m_pooled: Reduced messages per node (num_nodes, m_dim).
        :param invariants: Vector invariants (num_nodes, 2 * vector_channels), or None without
            vector channels.
        :return: Updated node features (num_nodes, dim)."""

        parts = [self.node_norm(h_node), m_pooled]
        if invariants is not None:
            parts.append(invariants)

        return self.node_mlp(torch.cat(parts, dim=-1)) + h_node
