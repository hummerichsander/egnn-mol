import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .encodings import (
    Encoding,
    encode_distance,
    encode_distance_derivative,
    encoding_width,
    polynomial_envelope,
    polynomial_envelope_derivative,
)
from .geometry import minimum_image, squared_distance
from .nn import MLP
from .sparse import build_edges


class RadialField(nn.Module):
    """Equivariant radial velocity field with a closed-form divergence.

    Follows Köhler, Klein & Noé (https://arxiv.org/abs/2006.02425): every node moves along a
    weighted sum of its relative-position vectors, ``v_i = sum_j phi(d_ij) r_ij``, whose
    divergence is available in closed form as ``sum_ij [d(phi)/d(d_ij) * d_ij + D * phi]``. Their
    derivation only touches the diagonal Jacobian blocks ``dv_i/dx_i``, so the closed form needs
    nothing beyond ``phi`` reading positions solely through ``d_ij`` -- node and edge features may
    condition it freely, since neither depends on positions.

    Time is one such feature and gets no argument of its own: a flow passes it in as an extra
    node-feature channel, exactly as :class:`~egnn_mol.sparse.GeometricEGNN` requires, which is
    what keeps the two backbones' ``forward`` signatures interchangeable.

    Two consequences worth knowing before using it. There is no ``depth``: stacking these updates
    would make the total Jacobian a *product*, whose log-determinant is not a sum of traces, so
    the single pairwise sum is the whole receptive field and capacity has to come from the basis
    width and the coefficient head. And ``phi`` is linear in the radial basis by construction --
    the coefficient head predicts the basis weights, never ``phi`` itself -- which is what keeps
    ``d(phi)/d(d)`` exact rather than another autograd pass."""

    def __init__(
        self,
        *,
        dim: int,
        encoding: Encoding = "gaussian",
        encoding_features: int = 16,
        cutoff: float = 10.0,
        m_dim: int = 64,
        head_depth: int = 2,
        edge_dim: int = 0,
        distance_cutoff: float = 0.0,
        num_nearest_neighbors: int = 0,
        envelope_exponent: int = 6,
    ) -> None:
        """Build the field.

        :param dim: Node feature dimensionality (features must already be this wide, time
            channel included).
        :param encoding: Radial basis the pair weight is expanded in.
        :param encoding_features: Number of basis functions / frequency bands. With no depth to
            stack, this is the primary capacity knob.
        :param cutoff: Radial length scale of the encoding.
        :param m_dim: Hidden width of the coefficient head.
        :param head_depth: Number of hidden blocks in the coefficient head. It never reads
            positions, so it may be as deep as wanted without touching the closed form.
        :param edge_dim: Static edge-feature dimensionality (0 if no edge features).
        :param distance_cutoff: If > 0, add a radius graph of dynamic edges, and apply the
            polynomial envelope at that radius so the field stays smooth where edges enter and
            leave it -- without which the divergence would only hold away from the boundary.
        :param num_nearest_neighbors: If > 0, add a kNN graph of dynamic edges.
        :param envelope_exponent: Polynomial exponent of that envelope."""

        super().__init__()
        self.encoding = encoding
        self.encoding_features = encoding_features
        self.cutoff = cutoff
        self.edge_dim = edge_dim
        self.distance_cutoff = distance_cutoff
        self.num_nearest_neighbors = num_nearest_neighbors
        self.envelope_exponent = envelope_exponent

        self.coefficients = MLP(
            dim + edge_dim,
            m_dim,
            encoding_width(encoding, encoding_features),
            num_layers=head_depth,
        )

        # zero output weights start the field at v = 0, i.e. an identity flow whose Jacobian
        # determinant is exactly 1; the head still receives gradient and trains out of it.
        out = [m for m in self.coefficients.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)

    def forward(
        self,
        h_node: Tensor,
        x: Tensor,
        edge_index: Tensor | None = None,
        h_edge: Tensor | None = None,
        batch: Tensor | None = None,
        box: Tensor | None = None,
        distance_cutoff: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the velocity and its exact divergence on a packed graph.

        The divergence always comes back: it costs one derivative-basis evaluation and one dot
        product on top of the velocity, and a caller that drops it drops its graph with it.

        :param h_node: Node features (N, dim), carrying the time channel if the caller uses one.
        :param x: Node positions (N, 3).
        :param edge_index: Static edge connectivity (2, E) as ``[source/neighbor, target/center]``,
            or None.
        :param h_edge: Static edge features (E, edge_dim), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param box: Per-node box lengths (N, 3), or None.
        :param distance_cutoff: Radius of the dynamic graph for this call, overriding the
            constructed one; None uses it. The envelope and its derivative taper at the same
            radius, so the closed form keeps matching the field. There is no separate envelope
            flag here, so ``0.0`` drops the radius graph and its taper together, leaving the
            static edges. The encoding length scale ``cutoff`` is unaffected -- a radius past it
            aliases long edges onto short-range basis values.
        :return: Velocity (N, 3) and one divergence per graph (num_graphs,)."""

        # named `radius`, not `cutoff`: `self.cutoff` two lines below is the encoding scale.
        radius = self.distance_cutoff if distance_cutoff is None else distance_cutoff

        edge_index, h_edge, static = build_edges(
            x,
            edge_index,
            h_edge,
            batch,
            box,
            edge_dim=self.edge_dim,
            distance_cutoff=radius,
            num_nearest_neighbors=self.num_nearest_neighbors,
        )
        src, dst = edge_index[0], edge_index[1]

        rel_x = minimum_image(x[dst] - x[src], box[dst] if box is not None else None)
        dist = squared_distance(rel_x).clamp(min=1e-8).sqrt()

        b = encode_distance(dist, self.encoding, self.encoding_features, self.cutoff)
        db = encode_distance_derivative(
            dist, self.encoding, self.encoding_features, self.cutoff
        )
        if radius > 0:
            env = polynomial_envelope(dist, radius, self.envelope_exponent)
            d_env = polynomial_envelope_derivative(dist, radius, self.envelope_exponent)
            # a static edge never enters or leaves at the cutoff, so it needs no taper; d_env must
            # go to zero with it or the closed-form derivative stops matching the field.
            env = torch.where(static[:, None], torch.ones_like(env), env)
            d_env = torch.where(static[:, None], torch.zeros_like(d_env), d_env)
            b, db = env * b, d_env * b + env * db

        # summing the two endpoints keeps phi_ij == phi_ji, so the field stays the gradient
        # field of a pairwise potential as in the paper; the closed form holds either way.
        parts = [h_node[dst] + h_node[src]]
        if h_edge is not None:
            parts.append(h_edge)
        w = self.coefficients(torch.cat(parts, dim=-1))

        phi = (w * b).sum(dim=-1, keepdim=True)
        dphi = (w * db).sum(dim=-1, keepdim=True)

        v = scatter(phi * rel_x, dst, dim=0, dim_size=x.shape[0], reduce="sum")

        graph = batch[dst] if batch is not None else torch.zeros_like(dst)
        num_graphs = int(batch.max()) + 1 if batch is not None else 1
        div = scatter(
            dphi * dist + x.shape[-1] * phi,
            graph,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )

        return v, div.squeeze(-1)
