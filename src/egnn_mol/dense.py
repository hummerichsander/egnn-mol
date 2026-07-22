from typing import Literal

import torch
from torch import Tensor, nn
from einops import rearrange, repeat

from .encodings import Encoding
from .geometry import minimum_image, signed_volume, squared_distance
from .update import EquivariantUpdate

Aggregation = Literal["sum", "mean"]


def batched_index_select(values: Tensor, indices: Tensor, dim: int) -> Tensor:
    """Select from ``values`` along ``dim`` using batched neighbor indices (B, N, K).

    :param values: Source tensor with a leading batch axis.
    :param indices: Batched neighbor indices (B, N, K).
    :param dim: Axis of ``values`` to index into.
    :return: Selected slices (B, N, K, *trailing)."""

    b, n, k = indices.shape
    trailing = values.shape[dim + 1 :]

    if dim == 1:
        # Gather from (B, M, F) directly; the expand-then-gather form makes gather's backward
        # allocate a dense (B, N, M, F) gradient — O(N^2) and independent of K.
        m = values.shape[1]
        flat_values = values.reshape(b, m, -1)
        f = flat_values.shape[-1]
        flat_idx = indices.reshape(b, n * k, 1).expand(b, n * k, f)
        out = torch.gather(flat_values, 1, flat_idx)
        return out.reshape(b, n, k, *trailing)

    idx = indices
    if trailing:
        idx = idx.view(b, n, k, *([1] * len(trailing))).expand(b, n, k, *trailing)
    return values.gather(dim, idx)


def _box_for_pairs(unitcell_lengths: Tensor | None) -> Tensor | None:
    """Reshape per-graph box lengths (B, 3) to (B, 1, 1, 3) for pairwise broadcasting."""
    if unitcell_lengths is None:
        return None
    return unitcell_lengths.view(unitcell_lengths.shape[0], 1, 1, 3)


def build_neighborhood(
    coors: Tensor,
    unitcell_lengths: Tensor | None,
    adj_mat: Tensor | None,
    mask: Tensor | None,
    distance_cutoff: float,
    num_nearest_neighbors: int,
) -> tuple[Tensor, Tensor]:
    """Build a padded neighborhood from static bonds and internal distance-based edges.

    The graph is the union of the static ``adj_mat`` edges, a radius graph (``distance_cutoff``),
    and a kNN graph (``num_nearest_neighbors``), all under the minimum-image convention and with
    self-loops removed. It is derived once from the input coordinates and shared across layers.

    :param coors: Node coordinates (B, N, 3).
    :param unitcell_lengths: Periodic box lengths (B, 3), or None.
    :param adj_mat: Static boolean adjacency (B, N, N) or (N, N), or None.
    :param mask: Node validity mask (B, N), or None.
    :param distance_cutoff: Radius cutoff for dynamic edges (0 disables).
    :param num_nearest_neighbors: kNN degree for dynamic edges (0 disables).
    :return: Neighbor indices (B, N, K) and a boolean edge-validity mask (B, N, K)."""

    b, n, _ = coors.shape
    device = coors.device
    eye = torch.eye(n, dtype=torch.bool, device=device)[None]

    with torch.no_grad():
        rel = rearrange(coors, "b i d -> b i () d") - rearrange(
            coors, "b j d -> b () j d"
        )
        dist_sq = (minimum_image(rel, _box_for_pairs(unitcell_lengths)) ** 2).sum(
            dim=-1
        )

        pair_valid = torch.ones(b, n, n, dtype=torch.bool, device=device)
        if mask is not None:
            pair_valid = mask[:, :, None] & mask[:, None, :]

        graph = torch.zeros(b, n, n, dtype=torch.bool, device=device)
        active = adj_mat is not None or distance_cutoff > 0 or num_nearest_neighbors > 0

        if adj_mat is not None:
            graph = graph | (
                repeat(adj_mat.bool(), "i j -> b i j", b=b)
                if adj_mat.dim() == 2
                else adj_mat.bool()
            )
        if distance_cutoff > 0:
            graph = graph | (dist_sq < distance_cutoff**2)
        if num_nearest_neighbors > 0:
            ranking = dist_sq.masked_fill(eye | ~pair_valid, float("inf"))
            k = min(num_nearest_neighbors, n - 1)
            knn = ranking.topk(k, dim=-1, largest=False).indices
            graph = graph.scatter(-1, knn, torch.ones_like(knn, dtype=torch.bool))
        if not active:
            graph = (~eye).expand(b, n, n).clone()

        graph = graph & ~eye & pair_valid

        k_max = int(graph.sum(dim=-1).max().clamp(min=1))
        # stable descending argsort puts real neighbors (1) first in ascending index order,
        # padding slots (0) after; gather recovers which of the K slots are real edges.
        order = graph.int().argsort(dim=-1, descending=True, stable=True)
        nbhd_indices = order[..., :k_max]
        nbhd_mask = torch.gather(graph, -1, nbhd_indices)

    return nbhd_indices, nbhd_mask


class DenseEGNNLayer(nn.Module):
    """One dense message-passing layer wrapping the shared :class:`EquivariantUpdate`."""

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
        dropout: float = 0.0,
        norm_feats: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        soft_edges: bool = False,
        coor_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
    ) -> None:
        """See :class:`E3GNN` for the meaning of the arguments."""

        super().__init__()
        assert aggr in {"sum", "mean"}
        self.aggr = aggr
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
        feats: Tensor,
        coors: Tensor,
        unitcell_lengths: Tensor | None = None,
        edges: Tensor | None = None,
        mask: Tensor | None = None,
        nbhd_indices: Tensor | None = None,
        nbhd_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Message-passing update over a neighborhood (or all-pairs, self-excluded, if none given).

        :param feats: Node features (B, N, D).
        :param coors: Node coordinates (B, N, 3).
        :param unitcell_lengths: Periodic box lengths (B, 3), or None.
        :param edges: Edge features — dense (B, N, N, E) or, with ``nbhd_indices``, (B, N, K, E).
        :param mask: Node validity mask (B, N), or None.
        :param nbhd_indices: Precomputed neighbor indices (B, N, K), or None for all-pairs.
        :param nbhd_mask: Precomputed edge-validity mask (B, N, K), or None.
        :return: Updated features (B, N, D) and coordinates (B, N, 3)."""

        box = _box_for_pairs(unitcell_lengths)
        b, n, _ = feats.shape

        if nbhd_indices is not None:
            coors_j = batched_index_select(coors, nbhd_indices, dim=1)
            rel_coors = minimum_image(
                rearrange(coors, "b i d -> b i () d") - coors_j, box
            )
            feats_j = batched_index_select(feats, nbhd_indices, dim=1)
        else:
            rel_coors = rearrange(coors, "b i d -> b i () d") - rearrange(
                coors, "b j d -> b () j d"
            )
            rel_coors = minimum_image(rel_coors, box)
            feats_j = rearrange(feats, "b j d -> b () j d")

        dist = squared_distance(rel_coors).clamp(min=1e-8).sqrt()
        feats_i = rearrange(feats, "b i d -> b i () d")
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        m_ij = self.core.message(feats_i, feats_j, dist, edges)

        edge_mask = self._edge_mask(mask, nbhd_indices, nbhd_mask, b, n, coors.device)

        normed = self.core.normalize_rel(rel_coors)

        if self.core.tripp:
            abc = self.core.triple_abc(m_ij)  # (B, N, K, 3)
            weighted = abc.unsqueeze(-1) * normed.unsqueeze(-2)  # (B, N, K, 3=k, 3=xyz)
            if edge_mask is not None:
                weighted = weighted.masked_fill(
                    ~rearrange(edge_mask, "... -> ... () ()"), 0.0
                )
            v = weighted.sum(dim=2)  # (B, N, 3=k, 3=xyz)
            chi = signed_volume(v[..., 0, :], v[..., 1, :], v[..., 2, :])  # (B, N, 1)
            chi_i = rearrange(chi, "b i one -> b i () one")
            if nbhd_indices is not None:
                chi_j = batched_index_select(chi, nbhd_indices, dim=1)
            else:
                chi_j = rearrange(chi, "b j one -> b () j one")
            chi_i, chi_j = torch.broadcast_tensors(chi_i, chi_j)
            weight = self.core.coord_weight(m_ij, chi_i, chi_j)
        else:
            weight = self.core.coord_weight(m_ij)

        delta = weight * normed
        if edge_mask is not None:
            delta = delta.masked_fill(~rearrange(edge_mask, "... -> ... ()"), 0.0)
        coors_out = coors + delta.sum(dim=2)

        if edge_mask is not None:
            m_ij = m_ij.masked_fill(~rearrange(edge_mask, "... -> ... ()"), 0.0)
        if self.aggr == "mean":
            if edge_mask is not None:
                count = rearrange(edge_mask, "... -> ... ()").sum(dim=2).clamp(min=1)
                m_pooled = m_ij.sum(dim=2) / count
            else:
                m_pooled = m_ij.mean(dim=2)
        else:
            m_pooled = m_ij.sum(dim=2)

        feats_out = self.core.update_feats(feats, m_pooled)
        return feats_out, coors_out

    def _edge_mask(
        self,
        mask: Tensor | None,
        nbhd_indices: Tensor | None,
        nbhd_mask: Tensor | None,
        b: int,
        n: int,
        device: torch.device,
    ) -> Tensor | None:
        """Combine node padding, neighborhood validity, and self-exclusion into one edge mask."""

        edge_mask: Tensor | None = None
        if mask is not None:
            mask_i = rearrange(mask, "b i -> b i ()")
            if nbhd_indices is not None:
                mask_j = batched_index_select(mask, nbhd_indices, dim=1)
            else:
                mask_j = rearrange(mask, "b j -> b () j")
            edge_mask = (mask_i * mask_j).bool()
        if nbhd_mask is not None:
            edge_mask = nbhd_mask if edge_mask is None else (edge_mask & nbhd_mask)
        if nbhd_indices is None:
            # All-pairs path: exclude self-loops on the diagonal.
            off_diag = ~torch.eye(n, dtype=torch.bool, device=device)[None].expand(
                b, n, n
            )
            edge_mask = off_diag if edge_mask is None else (edge_mask & off_diag)
        return edge_mask


class E3GNN(nn.Module):
    """E(3)-equivariant graph neural network on dense padded tensors.

    Handles periodic boxes (pass ``unitcell_lengths``) and open boundaries. The neighborhood is
    the union of static bonds (``adj_mat`` / ``edges``, given from outside) and internal
    distance-based edges (``distance_cutoff`` radius and/or ``num_nearest_neighbors`` kNN), with
    self-loops excluded; with none of these it is all-pairs. It is built once and shared across
    layers."""

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
        :param kwargs: Extra keyword arguments forwarded to every :class:`DenseEGNNLayer`."""

        super().__init__()
        self.distance_cutoff = distance_cutoff
        self.num_nearest_neighbors = num_nearest_neighbors

        self.layers = nn.ModuleList(
            [
                DenseEGNNLayer(
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
        feats: Tensor,
        coors: Tensor,
        unitcell_lengths: Tensor | None = None,
        adj_mat: Tensor | None = None,
        edges: Tensor | None = None,
        mask: Tensor | None = None,
        return_coor_changes: bool = False,
    ):
        """Run all message-passing layers.

        :param feats: Node features (B, N, D).
        :param coors: Node coordinates (B, N, 3).
        :param unitcell_lengths: Periodic box lengths (B, 3), or None.
        :param adj_mat: Static boolean adjacency (B, N, N) or (N, N), or None.
        :param edges: Static edge features (B, N, N, E), or None.
        :param mask: Node validity mask (B, N), or None.
        :param return_coor_changes: If True, also return the coordinate trajectory.
        :return: ``(feats, coors)`` or ``(feats, coors, coor_changes)``."""

        nbhd_indices: Tensor | None = None
        nbhd_mask: Tensor | None = None
        if (
            adj_mat is not None
            or self.distance_cutoff > 0
            or self.num_nearest_neighbors > 0
        ):
            nbhd_indices, nbhd_mask = build_neighborhood(
                coors,
                unitcell_lengths,
                adj_mat,
                mask,
                self.distance_cutoff,
                self.num_nearest_neighbors,
            )
            if edges is not None:
                edges = batched_index_select(edges, nbhd_indices, dim=2)

        coor_changes = [coors]
        for layer in self.layers:
            feats, coors = layer(
                feats,
                coors,
                unitcell_lengths,
                edges=edges,
                mask=mask,
                nbhd_indices=nbhd_indices,
                nbhd_mask=nbhd_mask,
            )
            coor_changes.append(coors)

        if return_coor_changes:
            return feats, coors, coor_changes
        return feats, coors
