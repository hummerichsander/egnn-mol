from typing import Literal

import torch
from torch import Tensor, nn
from torch_cluster import knn_graph, radius_graph
from torch_geometric.utils import coalesce, scatter

from .encodings import Encoding
from .geometry import minimum_image, signed_volume, squared_distance
from .update import EquivariantUpdate

Aggregation = Literal["sum", "mean"]


def radius_graph_pbc(
    pos: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    box: Tensor | None = None,
    include_self: bool = False,
) -> Tensor:
    """Build an undirected within-cutoff graph under the minimum-image convention.

    Uses a dense ``(N, N)`` pairwise pass, so it is intended for small systems (the typical
    periodic ``distance_cutoff`` use case); it is not a linear-scaling neighbor list.

    :param pos: Node positions (N, 3).
    :param cutoff: Distance cutoff.
    :param batch: Graph membership (N,), or None for a single graph.
    :param box: Per-node box lengths (N, 3), or None for open boundaries.
    :param include_self: Whether to keep self-loops.
    :return: Edge index (2, E) with the convention ``[source/neighbor, target/center]``."""

    n = pos.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=pos.device)

    rel = pos[:, None, :] - pos[None, :, :]
    rel = minimum_image(rel, box[:, None, :] if box is not None else None)
    dist_sq = (rel**2).sum(dim=-1)
    within = (batch[:, None] == batch[None, :]) & (dist_sq < cutoff**2)
    if not include_self:
        within = within & ~torch.eye(n, dtype=torch.bool, device=pos.device)

    center, neighbor = within.nonzero(as_tuple=True)
    return torch.stack([neighbor, center], dim=0)


def knn_graph_pbc(
    pos: Tensor,
    k: int,
    batch: Tensor | None = None,
    box: Tensor | None = None,
) -> Tensor:
    """Build a directed k-nearest-neighbor graph under the minimum-image convention.

    Dense ``(N, N)`` pairwise pass — intended for small systems.

    :param pos: Node positions (N, 3).
    :param k: Number of nearest neighbors per node.
    :param batch: Graph membership (N,), or None for a single graph.
    :param box: Per-node box lengths (N, 3), or None for open boundaries.
    :return: Edge index (2, E) with the convention ``[source/neighbor, target/center]``."""

    n = pos.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=pos.device)

    rel = pos[:, None, :] - pos[None, :, :]
    rel = minimum_image(rel, box[:, None, :] if box is not None else None)
    dist_sq = (rel**2).sum(dim=-1)
    same_graph = batch[:, None] == batch[None, :]
    dist_sq = dist_sq.masked_fill(~same_graph, float("inf"))
    dist_sq.fill_diagonal_(float("inf"))

    k = min(k, n)
    neighbor = dist_sq.topk(k, dim=-1, largest=False).indices  # (N, k)
    center = torch.arange(n, device=pos.device)[:, None].expand(-1, k)
    return torch.stack([neighbor.reshape(-1), center.reshape(-1)], dim=0)


def radius_edges(
    pos: Tensor, cutoff: float, batch: Tensor | None, box: Tensor | None
) -> Tensor:
    """Radius graph: minimum-image dense pass under PBC (``box`` given), else ``torch_cluster``.

    :param pos: Node positions (N, 3).
    :param cutoff: Distance cutoff.
    :param batch: Graph membership (N,), or None.
    :param box: Per-node box lengths (N, 3) for PBC, or None for open boundaries.
    :return: Edge index (2, E), ``[source/neighbor, target/center]``."""

    if box is not None:
        return radius_graph_pbc(pos, cutoff, batch, box)
    return radius_graph(
        pos, r=cutoff, batch=batch, loop=False, max_num_neighbors=pos.shape[0]
    )


def knn_edges(pos: Tensor, k: int, batch: Tensor | None, box: Tensor | None) -> Tensor:
    """kNN graph: minimum-image dense pass under PBC (``box`` given), else ``torch_cluster``.

    :param pos: Node positions (N, 3).
    :param k: Number of nearest neighbors per node.
    :param batch: Graph membership (N,), or None.
    :param box: Per-node box lengths (N, 3) for PBC, or None for open boundaries.
    :return: Edge index (2, E), ``[source/neighbor, target/center]``."""

    if box is not None:
        return knn_graph_pbc(pos, k, batch, box)
    return knn_graph(pos, k=k, batch=batch, loop=False, flow="source_to_target")


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
        norm_x: bool = False,
        norm_pos: bool = False,
        norm_pos_scale_init: float = 1.0,
        dropout: float = 0.0,
        pos_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
    ) -> None:
        """See :class:`GeometricEGNN` for the shared arguments.

        :param aggr: How to aggregate messages onto nodes (``"sum"`` or ``"mean"``)."""

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
            norm_x=norm_x,
            norm_pos=norm_pos,
            norm_pos_scale_init=norm_pos_scale_init,
            dropout=dropout,
            pos_weights_clamp_value=pos_weights_clamp_value,
            tripp_num_layers=tripp_num_layers,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        box: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Message-passing update on a packed graph.

        :param x: Node features (N, dim).
        :param pos: Node positions (N, 3).
        :param edge_index: Edge connectivity (2, E) as ``[source/neighbor, target/center]``.
        :param edge_attr: Edge features (E, edge_dim), or None.
        :param box: Per-node box lengths (N, 3), or None.
        :return: Updated features (N, dim) and positions (N, 3)."""

        src, dst = edge_index[0], edge_index[1]
        n = pos.shape[0]

        rel_pos = minimum_image(
            pos[dst] - pos[src], box[dst] if box is not None else None
        )
        dist = squared_distance(rel_pos).clamp(min=1e-8).sqrt()

        m_ij = self.core.message(x[dst], x[src], dist, edge_attr)

        normed = self.core.normalize_rel(rel_pos)
        if self.core.tripp:
            abc = self.core.triple_abc(m_ij)  # (E, 3)
            v = torch.stack(
                [
                    scatter(abc[:, k : k + 1] * normed, dst, dim=0, dim_size=n)
                    for k in range(3)
                ],
                dim=1,
            )  # (N, 3=k, 3=xyz)
            chi = signed_volume(v[:, 0], v[:, 1], v[:, 2])  # (N, 1)
            weight = self.core.pos_weight(m_ij, chi[dst], chi[src])
        else:
            weight = self.core.pos_weight(m_ij)

        pos_out = pos + scatter(weight * normed, dst, dim=0, dim_size=n, reduce="sum")

        m_pooled = scatter(m_ij, dst, dim=0, dim_size=n, reduce=self.aggr)
        x_out = self.core.update_x(x, m_pooled)
        return x_out, pos_out


class GeometricEGNN(nn.Module):
    """E(3)-equivariant GNN on packed torch-geometric tensors.

    Expects node features already projected to ``dim`` (embed upstream, as with the dense
    backbone). The neighborhood is the union of static bonds (``edge_index`` / ``edge_attr``,
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
        aggr: Aggregation = "sum",
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
        :param aggr: Message aggregation onto nodes ("sum" or "mean").
        :param kwargs: Extra keyword arguments forwarded to every :class:`SparseEGNNLayer`."""

        super().__init__()
        self.edge_dim = edge_dim
        self.distance_cutoff = distance_cutoff
        self.num_nearest_neighbors = num_nearest_neighbors
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
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        box: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run all message-passing layers on a packed graph.

        :param x: Node features (N, dim).
        :param pos: Node positions (N, 3).
        :param edge_index: Static edge connectivity (2, E) as ``[source/neighbor, target/center]``,
            or None.
        :param edge_attr: Static edge features (E, edge_dim), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param box: Per-node box lengths (N, 3), or None.
        :return: Updated features (N, dim) and positions (N, 3)."""

        edge_index, edge_attr = self._build_graph(
            pos, edge_index, edge_attr, batch, box
        )

        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr, box=box)

        return x, pos

    def _build_graph(
        self,
        pos: Tensor,
        edge_index: Tensor | None,
        edge_attr: Tensor | None,
        batch: Tensor | None,
        box: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Union static bonds with internal distance-based (radius / kNN) edges.

        Dynamic edges get zero edge features; duplicates are coalesced (summing attributes, so
        static features survive). With no static edges and no distance graph the result is
        all-pairs within each graph.

        :return: The combined ``(edge_index, edge_attr)``."""

        n = pos.shape[0]
        dynamic: list[Tensor] = []
        if self.distance_cutoff > 0:
            dynamic.append(radius_edges(pos, self.distance_cutoff, batch, box))
        if self.num_nearest_neighbors > 0:
            dynamic.append(knn_edges(pos, self.num_nearest_neighbors, batch, box))
        if edge_index is None and not dynamic:
            dynamic.append(radius_graph_pbc(pos, float("inf"), batch, box))

        indices = ([edge_index] if edge_index is not None else []) + dynamic

        if self.edge_dim > 0:
            attrs = [edge_attr] if edge_index is not None else []
            attrs += [pos.new_zeros(extra.shape[1], self.edge_dim) for extra in dynamic]
            return coalesce(
                torch.cat(indices, dim=1),
                torch.cat(attrs, dim=0),
                num_nodes=n,
                reduce="sum",
            )
        return coalesce(torch.cat(indices, dim=1), num_nodes=n), None
