from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch_cluster import knn_graph, radius_graph
from torch_geometric.utils import coalesce, scatter

from .encodings import Encoding, polynomial_envelope
from .geometry import minimum_image, signed_volume, squared_distance
from .sparsity import composed_closure, greedy_colouring
from .update import EquivariantUpdate

Aggregation = Literal["sum", "mean", "vp"]


def radius_graph_pbc(
    x: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    box: Tensor | None = None,
    include_self: bool = False,
) -> Tensor:
    """Build an undirected within-cutoff graph under the minimum-image convention.

    Uses a dense ``(N, N)`` pairwise pass, so it is intended for small systems (the typical
    periodic ``distance_cutoff`` use case); it is not a linear-scaling neighbor list.

    :param x: Node positions (N, 3).
    :param cutoff: Distance cutoff.
    :param batch: Graph membership (N,), or None for a single graph.
    :param box: Per-node box lengths (N, 3), or None for open boundaries.
    :param include_self: Whether to keep self-loops.
    :return: Edge index (2, E) with the convention ``[source/neighbor, target/center]``."""

    n = x.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=x.device)

    rel = x[:, None, :] - x[None, :, :]
    rel = minimum_image(rel, box[:, None, :] if box is not None else None)
    dist_sq = (rel**2).sum(dim=-1)
    within = (batch[:, None] == batch[None, :]) & (dist_sq < cutoff**2)
    if not include_self:
        within = within & ~torch.eye(n, dtype=torch.bool, device=x.device)

    center, neighbor = within.nonzero(as_tuple=True)
    return torch.stack([neighbor, center], dim=0)


def knn_graph_pbc(
    x: Tensor,
    k: int,
    batch: Tensor | None = None,
    box: Tensor | None = None,
) -> Tensor:
    """Build a directed k-nearest-neighbor graph under the minimum-image convention.

    Dense ``(N, N)`` pairwise pass — intended for small systems.

    :param x: Node positions (N, 3).
    :param k: Number of nearest neighbors per node.
    :param batch: Graph membership (N,), or None for a single graph.
    :param box: Per-node box lengths (N, 3), or None for open boundaries.
    :return: Edge index (2, E) with the convention ``[source/neighbor, target/center]``."""

    n = x.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=x.device)

    rel = x[:, None, :] - x[None, :, :]
    rel = minimum_image(rel, box[:, None, :] if box is not None else None)
    dist_sq = (rel**2).sum(dim=-1)
    same_graph = batch[:, None] == batch[None, :]
    dist_sq = dist_sq.masked_fill(~same_graph, float("inf"))
    dist_sq.fill_diagonal_(float("inf"))

    k = min(k, n)
    neighbor = dist_sq.topk(k, dim=-1, largest=False).indices  # (N, k)
    center = torch.arange(n, device=x.device)[:, None].expand(-1, k)
    return torch.stack([neighbor.reshape(-1), center.reshape(-1)], dim=0)


def radius_edges(
    x: Tensor, cutoff: float, batch: Tensor | None, box: Tensor | None
) -> Tensor:
    """Radius graph: minimum-image dense pass under PBC (``box`` given), else ``torch_cluster``.

    :param x: Node positions (N, 3).
    :param cutoff: Distance cutoff.
    :param batch: Graph membership (N,), or None.
    :param box: Per-node box lengths (N, 3) for PBC, or None for open boundaries.
    :return: Edge index (2, E), ``[source/neighbor, target/center]``."""

    if box is not None:
        return radius_graph_pbc(x, cutoff, batch, box)
    return radius_graph(
        x, r=cutoff, batch=batch, loop=False, max_num_neighbors=x.shape[0]
    )


def knn_edges(x: Tensor, k: int, batch: Tensor | None, box: Tensor | None) -> Tensor:
    """kNN graph: minimum-image dense pass under PBC (``box`` given), else ``torch_cluster``.

    :param x: Node positions (N, 3).
    :param k: Number of nearest neighbors per node.
    :param batch: Graph membership (N,), or None.
    :param box: Per-node box lengths (N, 3) for PBC, or None for open boundaries.
    :return: Edge index (2, E), ``[source/neighbor, target/center]``."""

    if box is not None:
        return knn_graph_pbc(x, k, batch, box)
    return knn_graph(x, k=k, batch=batch, loop=False, flow="source_to_target")


def build_edges(
    x: Tensor,
    edge_index: Tensor | None,
    h_edge: Tensor | None,
    batch: Tensor | None,
    box: Tensor | None,
    *,
    edge_dim: int = 0,
    distance_cutoff: float = 0.0,
    num_nearest_neighbors: int = 0,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Union static bonds with internal distance-based (radius / kNN) edges.

    Dynamic edges get zero edge features; duplicates are coalesced (summing attributes, so
    static features survive). With no static edges and no distance graph the result is
    all-pairs within each graph. Shared by every packed backbone, so they provably see the
    same edge set for the same arguments.

    The returned mask marks the edges the caller supplied. Those exist independently of the
    positions, so nothing about them appears or disappears at a cutoff and there is nothing to
    taper; :meth:`GeometricEGNN.edge_envelope` exempts them.

    :param x: Node positions (N, 3).
    :param edge_index: Static edge connectivity (2, E) as ``[source/neighbor, target/center]``,
        or None.
    :param h_edge: Static edge features (E, edge_dim), or None.
    :param batch: Graph membership (N,), or None for a single graph.
    :param box: Per-node box lengths (N, 3), or None.
    :param edge_dim: Edge-feature width; 0 returns no edge features.
    :param distance_cutoff: If > 0, add a radius graph of dynamic edges.
    :param num_nearest_neighbors: If > 0, add a kNN graph of dynamic edges.
    :return: The combined ``(edge_index, h_edge, static)``, the last a per-edge mask (E,) that is
        True on the caller-supplied edges."""

    n = x.shape[0]
    dynamic: list[Tensor] = []
    if distance_cutoff > 0:
        dynamic.append(radius_edges(x, distance_cutoff, batch, box))
    if num_nearest_neighbors > 0:
        dynamic.append(knn_edges(x, num_nearest_neighbors, batch, box))
    if edge_index is None and not dynamic:
        dynamic.append(radius_graph_pbc(x, float("inf"), batch, box))

    indices = ([edge_index] if edge_index is not None else []) + dynamic

    # the flag has to ride through `coalesce`, which re-sorts the edge index; a mask built
    # beforehand would no longer line up with the rows it describes.
    flags = [x.new_ones(edge_index.shape[1], 1)] if edge_index is not None else []
    flags += [x.new_zeros(extra.shape[1], 1) for extra in dynamic]
    flag = torch.cat(flags, dim=0)

    if edge_dim > 0:
        attrs = [h_edge] if edge_index is not None else []
        attrs += [x.new_zeros(extra.shape[1], edge_dim) for extra in dynamic]
        index, (h_edge, flag) = coalesce(
            torch.cat(indices, dim=1),
            [torch.cat(attrs, dim=0), flag],
            num_nodes=n,
            reduce="sum",
        )
        return index, h_edge, flag.squeeze(-1) > 0

    index, flag = coalesce(torch.cat(indices, dim=1), flag, num_nodes=n, reduce="sum")

    return index, None, flag.squeeze(-1) > 0


class SparseEGNNLayer(nn.Module):
    """One packed-graph message-passing layer wrapping the shared :class:`EquivariantUpdate`."""

    def __init__(
        self,
        *,
        dim: int,
        encoding: Encoding = "bessel",
        encoding_features: int = 8,
        cutoff: float = 10.0,
        m_dim: int = 16,
        edge_dim: int = 0,
        aggr: Aggregation = "sum",
        soft_edges: bool = False,
        norm_h_node: bool = False,
        norm_displacement: bool = False,
        norm_displacement_scale_init: float = 1.0,
        dropout: float = 0.0,
        x_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
        mlp_depth: int = 1,
        vector_channels: int = 0,
        vector_chirality: bool = False,
    ) -> None:
        """See :class:`GeometricEGNN` for the shared arguments.

        :param aggr: How to aggregate messages onto nodes (``"sum"``, ``"mean"`` or ``"vp"``)."""

        super().__init__()
        self.aggr = aggr
        self.core = EquivariantUpdate(
            dim=dim,
            encoding=encoding,
            encoding_features=encoding_features,
            cutoff=cutoff,
            m_dim=m_dim,
            edge_dim=edge_dim,
            soft_edges=soft_edges,
            norm_h_node=norm_h_node,
            norm_displacement=norm_displacement,
            norm_displacement_scale_init=norm_displacement_scale_init,
            dropout=dropout,
            x_weights_clamp_value=x_weights_clamp_value,
            tripp_num_layers=tripp_num_layers,
            mlp_depth=mlp_depth,
            vector_channels=vector_channels,
            vector_chirality=vector_chirality,
        )

    def forward(
        self,
        h_node: Tensor,
        x: Tensor,
        edge_index: Tensor,
        h_edge: Tensor | None = None,
        box: Tensor | None = None,
        env: Tensor | None = None,
        vec: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Message-passing update on a packed graph.

        :param h_node: Node features (N, dim).
        :param x: Node positions (N, 3).
        :param edge_index: Edge connectivity (2, E) as ``[source/neighbor, target/center]``.
        :param h_edge: Edge features (E, edge_dim), or None.
        :param box: Per-node box lengths (N, 3), or None.
        :param env: Per-edge cutoff envelope (E, 1), or None for an untapered update.
        :param vec: Vector features (N, vector_channels, 3), or None without them.
        :return: Updated features (N, dim), positions (N, 3) and vector features."""

        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]

        rel_x = minimum_image(x[dst] - x[src], box[dst] if box is not None else None)
        dist = squared_distance(rel_x).clamp(min=1e-8).sqrt()

        m_ij = self.core.message(h_node[dst], h_node[src], dist, h_edge)

        # applied to the displacement and the message, the two quantities every aggregation
        # here is built from, so an edge fades out of all three at once.
        normed = self.core.normalize_rel(rel_x)
        if env is not None:
            normed, m_ij = env * normed, env * m_ij

        chis = []
        if self.core.tripp:
            abc = self.core.triple_abc(m_ij)  # (E, 3)
            v = torch.stack(
                [
                    scatter(abc[:, k : k + 1] * normed, dst, dim=0, dim_size=n)
                    for k in range(3)
                ],
                dim=1,
            )  # (N, 3=k, 3=xyz)
            chis.append(signed_volume(v[:, 0], v[:, 1], v[:, 2]))  # (N, 1)
        if self.core.vector_chirality:
            # off the vectors this layer was handed, so it adds no hop of its own.
            chis.append(self.core.vector_chirality_scalar(vec))

        if chis:
            chi = torch.cat(chis, dim=-1)
            weight = self.core.x_weight(m_ij, chi[dst], chi[src])
        else:
            weight = self.core.x_weight(m_ij)

        x_out = x + scatter(weight * normed, dst, dim=0, dim_size=n, reduce="sum")

        invariants, vec_out = None, vec
        if self.core.vector_channels:
            # one hop, like the message: the neighbour's *incoming* vectors, plus this edge's own
            # direction. Reading a neighbour's updated vectors instead would put the layer two
            # hops out, the way the chirality term does, and `layer_adjacencies` counts one.
            a_vv, a_vs = self.core.vector_coefficients(m_ij)
            message = a_vv[..., None] * vec[src] + a_vs[..., None] * normed[:, None, :]
            vec_out = vec + scatter(message, dst, dim=0, dim_size=n, reduce="sum")

            u, invariants = self.core.vector_invariants(vec_out)
            vec_out = vec_out + self.core.gate_vec(h_node, invariants)[..., None] * u

        m_pooled = self.pool(m_ij, dst, n, env)
        h_node_out = self.core.update_h_node(h_node, m_pooled, invariants)
        return h_node_out, x_out, vec_out

    def pool(self, m_ij: Tensor, dst: Tensor, n: int, env: Tensor | None) -> Tensor:
        """Reduce the per-edge messages onto their target nodes.

        :param m_ij: Messages (E, m_dim), already tapered by ``env`` if there is one.
        :param dst: Target node of every edge (E,).
        :param n: Number of nodes.
        :param env: Per-edge cutoff envelope (E, 1), or None.
        :return: Pooled messages (n, m_dim)."""

        if self.aggr != "vp":
            return scatter(m_ij, dst, dim=0, dim_size=n, reduce=self.aggr)

        # sum / sqrt(degree), so the pooled message keeps a degree-independent scale: plain "sum"
        # grows its variance with degree and "mean" shrinks it, and `use_residue` puts sparse bond
        # nodes and dense residue cliques in one graph. The degree counts the envelope, squared,
        # rather than edges: an integer count would jump as an edge crosses `distance_cutoff`,
        # reintroducing the discontinuity the envelope exists to remove, and sum(env^2) is also
        # the variance the taper actually leaves. Offset by one rather than clamped, to stay
        # smooth and to keep a lone tapered edge tapered instead of rescaling it back to full.
        weights = torch.ones_like(m_ij[:, :1]) if env is None else env
        degree = scatter(weights * weights, dst, dim=0, dim_size=n, reduce="sum")
        total = scatter(m_ij, dst, dim=0, dim_size=n, reduce="sum")

        return total / (1.0 + degree).sqrt()


class GeometricEGNN(nn.Module):
    """E(3)-equivariant GNN on packed torch-geometric tensors.

    Expects node features already projected to ``dim`` (embed upstream, as with the dense
    backbone). The neighborhood is the union of static bonds (``edge_index`` / ``h_edge``,
    given from outside) and internal distance-based edges (``distance_cutoff`` radius and/or
    ``num_nearest_neighbors`` kNN), with self-loops excluded; with none of these it is all-pairs
    within each graph. Dynamic graphs use ``torch_cluster`` for open boundaries and the dense
    minimum-image builder under periodic boundaries (``box`` given)."""

    def __init__(
        self,
        *,
        depth: int,
        dim: int,
        encoding: Encoding = "bessel",
        encoding_features: int = 8,
        cutoff: float = 10.0,
        edge_dim: int = 0,
        distance_cutoff: float = 0.0,
        num_nearest_neighbors: int = 0,
        dynamic_layers: Sequence[int] | None = None,
        aggr: Aggregation = "sum",
        envelope: bool = False,
        envelope_exponent: int = 6,
        **kwargs,
    ) -> None:
        """Build the network.

        :param depth: Number of message-passing layers.
        :param dim: Node feature dimensionality (features must already be this wide).
        :param encoding: Radial distance encoding for every layer.
        :param encoding_features: Number of basis functions / frequency bands for the encoding.
        :param cutoff: Radial length scale of the encoding.
        :param edge_dim: Static edge-feature dimensionality (0 if no edge features).
        :param distance_cutoff: If > 0, add a radius graph of dynamic edges.
        :param num_nearest_neighbors: If > 0, add a kNN graph of dynamic edges.
        :param dynamic_layers: Indices of the layers that see the dynamic edges; the rest see the
            static graph alone. None (the default) gives every layer the full neighborhood. The
            dynamic edges are what make the position Jacobian's sparsity pattern a ball of radius
            ``receptive_hops * distance_cutoff``, so restricting them to a few layers decouples
            the depth of the stack from the radius of that ball.
        :param aggr: Message aggregation onto nodes. "sum" and "mean" are the plain reductions;
            "vp" is variance-preserving, dividing the sum by the square root of the
            envelope-weighted degree so the pooled message's scale does not track node degree.
        :param envelope: Taper every edge's contribution to zero at ``distance_cutoff``, so the
            field stays C^1 where an edge enters or leaves the radius graph. Off by default: a
            network trained without it computes a different function, so turning it on silently
            would change what an existing checkpoint evaluates.
        :param envelope_exponent: Polynomial exponent of that envelope.
        :param kwargs: Extra keyword arguments forwarded to every :class:`SparseEGNNLayer`."""

        super().__init__()
        self.edge_dim = edge_dim
        self.distance_cutoff = distance_cutoff
        self.num_nearest_neighbors = num_nearest_neighbors

        if dynamic_layers is not None:
            dynamic_layers = tuple(sorted(set(dynamic_layers)))
            if not all(0 <= layer < depth for layer in dynamic_layers):
                raise ValueError(
                    f"dynamic_layers must index the {depth} layers, got {dynamic_layers}."
                )

        self.dynamic_layers = dynamic_layers

        if envelope and distance_cutoff <= 0:
            raise ValueError(
                "the envelope tapers edges at `distance_cutoff`, so it needs one; got "
                f"distance_cutoff={distance_cutoff}."
            )

        self.envelope_cutoff = distance_cutoff if envelope else None
        self.envelope_exponent = envelope_exponent

        self.layers = nn.ModuleList(
            [
                SparseEGNNLayer(
                    dim=dim,
                    encoding=encoding,
                    encoding_features=encoding_features,
                    cutoff=cutoff,
                    edge_dim=edge_dim,
                    aggr=aggr,
                    **kwargs,
                )
                for _ in range(depth)
            ]
        )

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
        """Run all message-passing layers on a packed graph.

        :param h_node: Node features (N, dim).
        :param x: Node positions (N, 3).
        :param edge_index: Static edge connectivity (2, E) as ``[source/neighbor, target/center]``,
            or None.
        :param h_edge: Static edge features (E, edge_dim), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param box: Per-node box lengths (N, 3), or None.
        :param distance_cutoff: Radius of the dynamic graph for this call, overriding the
            constructed one; None uses it. The envelope tapers at the same radius, so the field
            stays C^1 where an edge enters or leaves. The encoding length scale ``cutoff`` is
            unaffected -- a radius past it aliases long edges onto short-range basis values.
        :return: Updated features (N, dim) and positions (N, 3)."""

        edge_index, h_edge, static = self.build_neighborhood(
            x, edge_index, h_edge, batch, box, distance_cutoff
        )

        return self.run_layers(
            h_node,
            x,
            edge_index,
            h_edge,
            box,
            self.edge_envelope(x, edge_index, box, static, distance_cutoff),
            static,
        )

    def build_neighborhood(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        h_edge: Tensor | None = None,
        batch: Tensor | None = None,
        box: Tensor | None = None,
        distance_cutoff: float | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Build the edge set the layers run on, from this module's graph settings or a given radius.

        Split out of :meth:`forward` so the divergence can colour exactly the edges the layers
        went on to see, rather than a reconstruction of them.

        :param x: Node positions (N, 3).
        :param edge_index: Static edge connectivity (2, E), or None.
        :param h_edge: Static edge features (E, edge_dim), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param box: Per-node box lengths (N, 3), or None.
        :param distance_cutoff: Radius of the dynamic graph for this call, overriding the
            constructed one; None uses it. The envelope tapers at the same radius, so the field
            stays C^1 where an edge enters or leaves. The encoding length scale ``cutoff`` is
            unaffected -- a radius past it aliases long edges onto short-range basis values.
        :return: The combined ``(edge_index, h_edge, static)``, see :func:`build_edges`."""

        return build_edges(
            x,
            edge_index,
            h_edge,
            batch,
            box,
            edge_dim=self.edge_dim,
            distance_cutoff=(
                self.distance_cutoff if distance_cutoff is None else distance_cutoff
            ),
            num_nearest_neighbors=self.num_nearest_neighbors,
        )

    def edge_envelope(
        self,
        x: Tensor,
        edge_index: Tensor,
        box: Tensor | None = None,
        static: Tensor | None = None,
        distance_cutoff: float | None = None,
    ) -> Tensor | None:
        """The cutoff envelope of every edge, evaluated on the positions the edges were built from.

        Evaluated **once**, on the input positions, and shared by every layer -- exactly as the
        edge set is. Letting each layer taper by its own distances instead would break the very
        continuity the envelope exists for: an edge appears at ``distance_cutoff`` with zero
        weight, but by the second layer the pair has moved, so it would re-enter with a finite
        weight and the field would jump as the edge set changed.

        :param x: Node positions (N, 3), the ones the edge set was built from.
        :param edge_index: Edge connectivity (2, E).
        :param box: Per-node box lengths (N, 3), or None.
        :param static: Per-edge mask (E,) of caller-supplied edges to exempt, or None.
        :param distance_cutoff: Radius the edges were built at for this call, or None for the
            constructed one. The taper has to reach zero exactly where the graph ends, so this is
            the same value :meth:`build_neighborhood` was given.
        :return: Envelope weights (E, 1), or None when no envelope is configured."""

        if self.envelope_cutoff is None:
            return None

        cutoff = self.envelope_cutoff if distance_cutoff is None else distance_cutoff
        if cutoff <= 0:
            raise ValueError(
                "the envelope tapers edges at `distance_cutoff`, so it needs one; got "
                f"distance_cutoff={cutoff}."
            )

        src, dst = edge_index[0], edge_index[1]
        rel_x = minimum_image(x[dst] - x[src], box[dst] if box is not None else None)
        dist = squared_distance(rel_x).clamp(min=1e-8).sqrt()
        env = polynomial_envelope(dist, cutoff, self.envelope_exponent)

        # a static edge set is position-independent, so no edge enters or leaves at the cutoff and
        # there is nothing to taper; tapering it would silently delete every static edge longer
        # than the cutoff, which is exactly what a topological graph is for.
        if static is not None:
            env = torch.where(static[:, None], torch.ones_like(env), env)

        return env

    def run_layers(
        self,
        h_node: Tensor,
        x: Tensor,
        edge_index: Tensor,
        h_edge: Tensor | None,
        box: Tensor | None,
        env: Tensor | None = None,
        static: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run every message-passing layer on an already-built edge set.

        :param h_node: Node features (N, dim).
        :param x: Node positions (N, 3).
        :param edge_index: Edge connectivity (2, E).
        :param h_edge: Edge features (E, edge_dim), or None.
        :param box: Per-node box lengths (N, 3), or None.
        :param env: Per-edge cutoff envelope (E, 1), or None.
        :param static: Per-edge mask (E,) of the caller-supplied edges, needed only to honour
            ``dynamic_layers``; without it every layer reads the full edge set.
        :return: Updated features (N, dim) and positions (N, 3)."""

        # the vector channels are internal state of one pass: they start at zero, carry the
        # directional part between layers, and are read out as invariants rather than returned.
        channels = self.layers[0].core.vector_channels
        vec = x.new_zeros(x.shape[0], channels, 3) if channels else None

        for layer, (edges, features, envelope) in zip(
            self.layers, self.layer_edges(edge_index, h_edge, env, static)
        ):
            h_node, x, vec = layer(
                h_node, x, edges, features, box=box, env=envelope, vec=vec
            )

        return h_node, x

    def reads_dynamic(self, layer: int) -> bool:
        """Whether a layer sees the dynamic edges on top of the static graph.

        :param layer: Index of the layer.
        :return: True if it does."""

        return self.dynamic_layers is None or layer in self.dynamic_layers

    def layer_edges(
        self,
        edge_index: Tensor,
        h_edge: Tensor | None,
        env: Tensor | None,
        static: Tensor | None,
    ) -> list[tuple[Tensor, Tensor | None, Tensor | None]]:
        """The edge set every layer runs on, one entry per layer.

        A static-only layer carries no envelope: :meth:`edge_envelope` exempts static edges
        anyway, so its slice of ``env`` is all ones.

        :param edge_index: The full edge connectivity (2, E).
        :param h_edge: The full edge features (E, edge_dim), or None.
        :param env: The full per-edge envelope (E, 1), or None.
        :param static: Per-edge mask (E,) of the caller-supplied edges, or None.
        :return: One ``(edge_index, h_edge, env)`` triple per layer."""

        full = (edge_index, h_edge, env)
        if self.dynamic_layers is None or static is None:
            return [full] * len(self.layers)

        # sliced once here rather than per layer: every static-only layer reads this same view.
        bonded = (
            edge_index[:, static],
            h_edge[static] if h_edge is not None else None,
            None,
        )

        return [
            full if self.reads_dynamic(layer) else bonded
            for layer in range(len(self.layers))
        ]

    def layer_adjacencies(self, edge_index: Tensor, static: Tensor) -> list[Tensor]:
        """One edge index per hop of the stack, in order, for the composed sparsity pattern.

        A layer reads one hop of its own edge set, or two with the triple-product chirality term
        -- see :attr:`receptive_hops` for that recursion. ``vector_chirality`` is chirality of the
        other kind and stays at one: it contracts vectors the layer was handed rather than ones it
        aggregates, so its pseudoscalar sits at the same generation as ``h_j``, which the message
        already reads. A layer outside ``dynamic_layers`` therefore
        contributes hops of the static graph rather than of the full neighborhood, which is what
        keeps the composed ball small while the stack stays deep.

        :param edge_index: The full edge connectivity (2, E).
        :param static: Per-edge mask (E,) of the caller-supplied edges.
        :return: One edge index per hop."""

        hops = 2 if self.layers[0].core.tripp else 1
        bonded = edge_index[:, static]

        return [
            edge_index if self.reads_dynamic(layer) else bonded
            for layer in range(len(self.layers))
            for _ in range(hops)
        ]

    @property
    def receptive_hops(self) -> int:
        """Graph distance over which one position update can reach, across the whole stack.

        A layer's feature update reads one hop. Its position update reads one hop too, unless
        the triple-product chirality term is on: ``x_weight`` then also sees ``chi`` at both
        endpoints, and that ``chi`` is itself a one-hop aggregate, which puts the position update
        two hops out. ``vector_chirality`` feeds the same head a pseudoscalar without that cost,
        because the vectors it contracts were aggregated by earlier layers: reading a neighbour's
        *incoming* state is one hop, the same one ``m_ij`` already pays for ``h_j``.
        Writing the two radii as ``a_l`` (positions) and ``b_l`` (features), the recursion is
        ``a_l = max(a, b)_{l-1} + (2 if tripp else 1)`` and ``b_l = max(a, b)_{l-1} + 1``, so
        after ``depth`` layers the positions reach ``2 * depth`` hops with the chirality term
        and ``depth`` without.

        With ``dynamic_layers`` set this is still the number of hops walked, but they are no
        longer hops of one graph, so it is only an upper bound on the reach through the full
        neighborhood; :meth:`sparsity_pattern` composes the per-layer graphs instead.

        :return: The number of hops."""

        return len(self.layers) * (2 if self.layers[0].core.tripp else 1)

    def sparsity_pattern(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
        box: Tensor | None = None,
        distance_cutoff: float | None = None,
    ) -> Tensor:
        """The sparsity pattern of the position Jacobian, as an edge index.

        The edge set is built once per forward pass by non-differentiable neighbor searches and
        shared by every layer, so within one call the graph is fixed and ``dx_out_i / dx_j``
        vanishes structurally beyond the reach of the stack -- :attr:`receptive_hops` hops of the
        full neighborhood, or, under ``dynamic_layers``, the composition of what each layer
        actually reads.

        :param x: Node positions (N, 3).
        :param edge_index: Static edge connectivity (2, E), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param box: Per-node box lengths (N, 3), or None.
        :param distance_cutoff: Radius of the dynamic graph for this call, overriding the
            constructed one; None uses it. A wider radius costs more colours, which is
            what makes pricing one worth doing before running at it.
        :return: Edge index (2, E') of the pattern, carrying every self-loop."""

        # edge_dim=0: the pattern is set by which edges exist, never by what they carry, and
        # asking for features here would oblige the caller to supply them.
        edges, _, static = build_edges(
            x,
            edge_index,
            None,
            batch,
            box,
            edge_dim=0,
            distance_cutoff=(
                self.distance_cutoff if distance_cutoff is None else distance_cutoff
            ),
            num_nearest_neighbors=self.num_nearest_neighbors,
        )

        return composed_closure(self.layer_adjacencies(edges, static), x.shape[0])

    def jacobian_colouring(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
        box: Tensor | None = None,
        distance_cutoff: float | None = None,
    ) -> Tensor:
        """Group the nodes so one derivative pass reads every diagonal block of a group at once.

        :param x: Node positions (N, 3).
        :param edge_index: Static edge connectivity (2, E), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param box: Per-node box lengths (N, 3), or None.
        :param distance_cutoff: Radius of the dynamic graph for this call, overriding the
            constructed one; None uses it. A wider radius costs more colours.
        :return: Colours (N,), numbered contiguously from zero."""

        pattern = self.sparsity_pattern(x, edge_index, batch, box, distance_cutoff)

        return greedy_colouring(pattern, x.shape[0])

    def forward_and_divergence(
        self,
        h_node: Tensor,
        x: Tensor,
        edge_index: Tensor | None = None,
        h_edge: Tensor | None = None,
        batch: Tensor | None = None,
        box: Tensor | None = None,
        create_graph: bool = False,
        distance_cutoff: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the stack and take the exact divergence of its displacement field.

        The divergence is of ``x_out - x``, not of ``x_out`` -- the two differ by ``3N`` per
        graph -- matching :class:`~egnn_mol.radial.RadialField`, whose velocity is likewise a
        displacement.

        Exact, and costs ``3 * colours`` backward passes rather than the ``3N`` of a coordinate-
        at-a-time trace. Seeding a whole colour class at once gives, at node ``k`` of the class,
        ``sum_i dv_(i,a) / dx_(k,a)`` over the class; every term but ``i = k`` is structurally
        zero because the class is an independent set in the pattern, so the sum *is* the
        diagonal entry. The colour count tracks the size of a ``receptive_hops`` ball rather
        than the system, so this stops scaling with N.

        :param h_node: Node features (N, dim).
        :param x: Node positions (N, 3).
        :param edge_index: Static edge connectivity (2, E), or None.
        :param h_edge: Static edge features (E, edge_dim), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param box: Per-node box lengths (N, 3), or None.
        :param create_graph: Whether to build the second-order graph so the divergence itself is
            differentiable. Leave False unless backpropagating through it.
        :param distance_cutoff: Radius of the dynamic graph for this call, overriding the
            constructed one; None uses it. The envelope tapers at the same radius, so the field
            stays C^1 where an edge enters or leaves. The encoding length scale ``cutoff`` is
            unaffected -- a radius past it aliases long edges onto short-range basis values.
            The colouring is taken from the edges this radius produced, so it follows too.
        :return: Updated features (N, dim), positions (N, 3), and one divergence per graph
            (num_graphs,)."""

        edge_index, h_edge, static = self.build_neighborhood(
            x, edge_index, h_edge, batch, box, distance_cutoff
        )
        colours = greedy_colouring(
            composed_closure(self.layer_adjacencies(edge_index, static), x.shape[0]),
            x.shape[0],
        )

        graph = batch if batch is not None else torch.zeros_like(colours)
        num_graphs = int(graph.max()) + 1

        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            h_node, x_out = self.run_layers(
                h_node,
                x,
                edge_index,
                h_edge,
                box,
                self.edge_envelope(x, edge_index, box, static, distance_cutoff),
                static,
            )
            displacement = x_out - x

            div = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
            for colour in range(int(colours.max()) + 1):
                group = colours == colour
                for axis in range(x.shape[-1]):
                    seed = torch.zeros_like(displacement)
                    seed[group, axis] = 1.0
                    # value-only backward reuses the graph without a second-order one.
                    row = torch.autograd.grad(
                        displacement,
                        x,
                        grad_outputs=seed,
                        retain_graph=True,
                        create_graph=create_graph,
                    )[0]
                    div = div.index_add(0, graph[group], row[group, axis])

        if create_graph:
            return h_node, x_out, div

        return h_node.detach(), x_out.detach(), div.detach()
