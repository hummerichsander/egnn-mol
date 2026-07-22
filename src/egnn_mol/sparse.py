from typing import Literal

import torch
from torch import Tensor, nn

try:
    from torch_geometric.utils import coalesce, scatter
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The sparse backbone requires torch-geometric. Install it with `pip install egnn-mol[pyg]`."
    ) from exc

from .encodings import Encoding
from .geometry import minimum_image, signed_volume, squared_distance
from .update import EquivariantUpdate

Aggregation = Literal["sum", "mean"]

POS_DIM = 3


def radius_graph_pbc(
    coors: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    box: Tensor | None = None,
    include_self: bool = False,
) -> Tensor:
    """Build an undirected within-cutoff graph under the minimum-image convention.

    Uses a dense ``(N, N)`` pairwise pass, so it is intended for small systems (the typical
    ``distance_cutoff`` use case); it is not a linear-scaling neighbor list.

    :param coors: Node coordinates (N, 3).
    :param cutoff: Distance cutoff.
    :param batch: Graph membership (N,), or None for a single graph.
    :param box: Per-node box lengths (N, 3), or None for open boundaries.
    :param include_self: Whether to keep self-loops.
    :return: Edge index (2, E) with the convention ``[source/neighbor, target/center]``."""

    n = coors.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=coors.device)

    rel = coors[:, None, :] - coors[None, :, :]
    rel = minimum_image(rel, box[:, None, :] if box is not None else None)
    dist_sq = (rel**2).sum(dim=-1)
    within = (batch[:, None] == batch[None, :]) & (dist_sq < cutoff**2)
    if not include_self:
        within = within & ~torch.eye(n, dtype=torch.bool, device=coors.device)

    center, neighbor = within.nonzero(as_tuple=True)
    return torch.stack([neighbor, center], dim=0)


def knn_graph_pbc(
    coors: Tensor,
    k: int,
    batch: Tensor | None = None,
    box: Tensor | None = None,
) -> Tensor:
    """Build a directed k-nearest-neighbor graph under the minimum-image convention.

    Dense ``(N, N)`` pairwise pass — intended for small systems.

    :param coors: Node coordinates (N, 3).
    :param k: Number of nearest neighbors per node.
    :param batch: Graph membership (N,), or None for a single graph.
    :param box: Per-node box lengths (N, 3), or None for open boundaries.
    :return: Edge index (2, E) with the convention ``[source/neighbor, target/center]``."""

    n = coors.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=coors.device)

    rel = coors[:, None, :] - coors[None, :, :]
    rel = minimum_image(rel, box[:, None, :] if box is not None else None)
    dist_sq = (rel**2).sum(dim=-1)
    same_graph = batch[:, None] == batch[None, :]
    dist_sq = dist_sq.masked_fill(~same_graph, float("inf"))
    dist_sq.fill_diagonal_(float("inf"))

    k = min(k, n)
    neighbor = dist_sq.topk(k, dim=-1, largest=False).indices  # (N, k)
    center = torch.arange(n, device=coors.device)[:, None].expand(-1, k)
    return torch.stack([neighbor.reshape(-1), center.reshape(-1)], dim=0)


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
        norm_feats: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1.0,
        dropout: float = 0.0,
        coor_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
    ) -> None:
        """See :class:`GeometricEGNN` for the shared arguments.

        :param aggr: How to aggregate messages onto nodes (``"sum"`` or ``"mean"``)."""

        super().__init__()
        self.aggr = "sum" if aggr == "add" else aggr
        self.core = EquivariantUpdate(
            dim=dim,
            encoding=encoding,
            encoding_features=encoding_features,
            cutoff=cutoff,
            m_dim=m_dim,
            edge_dim=edge_dim,
            soft_edges=soft_edges,
            norm_feats=norm_feats,
            norm_coors=norm_coors,
            norm_coors_scale_init=norm_coors_scale_init,
            dropout=dropout,
            coor_weights_clamp_value=coor_weights_clamp_value,
            tripp_num_layers=tripp_num_layers,
        )

    def forward(
        self,
        coors: Tensor,
        feats: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        box: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Message-passing update on a packed graph.

        :param coors: Node coordinates (N, 3).
        :param feats: Node features (N, dim).
        :param edge_index: Edge connectivity (2, E) as ``[source/neighbor, target/center]``.
        :param edge_attr: Edge features (E, edge_dim), or None.
        :param box: Per-node box lengths (N, 3), or None.
        :return: Updated coordinates (N, 3) and features (N, dim)."""

        src, dst = edge_index[0], edge_index[1]
        n = coors.shape[0]

        rel = minimum_image(
            coors[dst] - coors[src], box[dst] if box is not None else None
        )
        dist = squared_distance(rel).clamp(min=1e-8).sqrt()

        m_ij = self.core.message(feats[dst], feats[src], dist, edge_attr)

        normed = self.core.normalize_rel(rel)
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
            weight = self.core.coord_weight(m_ij, chi[dst], chi[src])
        else:
            weight = self.core.coord_weight(m_ij)

        coors_out = coors + scatter(
            weight * normed, dst, dim=0, dim_size=n, reduce="sum"
        )

        m_pooled = scatter(m_ij, dst, dim=0, dim_size=n, reduce=self.aggr)
        feats_out = self.core.update_feats(feats, m_pooled)
        return coors_out, feats_out


class GeometricEGNN(nn.Module):
    """E(3)-equivariant GNN on packed torch-geometric tensors.

    Expects node features already projected to ``dim`` (embed upstream, as with the dense
    backbone). The neighborhood is the union of static bonds (``edge_index`` / ``edge_attr``,
    given from outside) and internal distance-based edges (``distance_cutoff`` radius and/or
    ``num_nearest_neighbors`` kNN), with self-loops excluded; with none of these it is all-pairs
    within each graph."""

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
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        unitcell_lengths: Tensor | None = None,
    ) -> Tensor:
        """Run all message-passing layers on a packed graph.

        :param x: Packed node tensor (N, 3 + dim): coordinates followed by features.
        :param edge_index: Static edge connectivity (2, E) as ``[source/neighbor, target/center]``,
            or None.
        :param edge_attr: Static edge features (E, edge_dim), or None.
        :param batch: Graph membership (N,), or None for a single graph.
        :param unitcell_lengths: Per-node box lengths (N, 3), or None.
        :return: Updated packed node tensor (N, 3 + dim)."""

        coors, feats = x[:, :POS_DIM], x[:, POS_DIM:]
        edge_index, edge_attr = self._build_graph(
            coors, edge_index, edge_attr, batch, unitcell_lengths
        )

        for layer in self.layers:
            coors, feats = layer(
                coors, feats, edge_index, edge_attr, box=unitcell_lengths
            )

        return torch.cat([coors, feats], dim=-1)

    def _build_graph(
        self,
        coors: Tensor,
        edge_index: Tensor | None,
        edge_attr: Tensor | None,
        batch: Tensor | None,
        box: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Union static bonds with internal distance-based (radius / kNN) edges.

        Dynamic edges get zero edge features; duplicates are coalesced (summing attributes, so
        static features survive). With no static edges and no distance graph the result is
        all-pairs within each graph (radius with infinite cutoff).

        :return: The combined ``(edge_index, edge_attr)``."""

        n = coors.shape[0]
        dynamic: list[Tensor] = []
        if self.distance_cutoff > 0:
            dynamic.append(radius_graph_pbc(coors, self.distance_cutoff, batch, box))
        if self.num_nearest_neighbors > 0:
            dynamic.append(knn_graph_pbc(coors, self.num_nearest_neighbors, batch, box))
        if edge_index is None and not dynamic:
            dynamic.append(radius_graph_pbc(coors, float("inf"), batch, box))

        indices = ([edge_index] if edge_index is not None else []) + dynamic

        if self.edge_dim > 0:
            attrs = [edge_attr] if edge_index is not None else []
            attrs += [
                coors.new_zeros(extra.shape[1], self.edge_dim) for extra in dynamic
            ]
            return coalesce(
                torch.cat(indices, dim=1),
                torch.cat(attrs, dim=0),
                num_nodes=n,
                reduce="sum",
            )
        return coalesce(torch.cat(indices, dim=1), num_nodes=n), None
