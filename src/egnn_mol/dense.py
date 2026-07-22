"""Dense native-torch E(3)-equivariant backbone.

Operates on batched padded tensors ``(B, N, .)``. A neighborhood is selected once per forward
(kNN / sparse-adjacency / within-cutoff) and shared across layers, so no dense ``(B, N, N)``
tensor is materialized when a sparse neighborhood is used. Depends only on torch + einops."""

from typing import Literal

import torch
from torch import Tensor, nn
from einops import rearrange, repeat

from .encodings import Encoding
from .geometry import minimum_image, signed_volume, squared_distance
from .update import EquivariantUpdate


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


def select_neighbors(
    coors: Tensor,
    unitcell_lengths: Tensor | None,
    adj_mat: Tensor | None,
    mask: Tensor | None,
    num_nearest_neighbors: int,
    only_sparse_neighbors: bool,
    valid_radius: float,
    dist_sq: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Select a fixed neighborhood for every node from the input coordinates.

    Adjacency edges are forced to rank first; the remaining ``topk`` slots are padding and are
    flagged invalid in the returned mask. The neighborhood is derived once and reused by every
    layer, so the graph follows the input configuration rather than the per-layer coordinates.

    :param coors: Node coordinates (B, N, 3).
    :param unitcell_lengths: Periodic box lengths (B, 3), or None.
    :param adj_mat: Boolean adjacency (B, N, N) or (N, N) forcing fixed edges, or None.
    :param mask: Node validity mask (B, N), or None.
    :param num_nearest_neighbors: Number of nearest neighbors for a pure kNN graph.
    :param only_sparse_neighbors: If True, use exactly the adjacency edges (k = max degree).
    :param valid_radius: Squared-distance radius below which a slot counts as a real edge.
    :param dist_sq: Optional precomputed squared distances (B, N, N).
    :return: Neighbor indices (B, N, K) and a boolean edge-validity mask (B, N, K)."""

    b, n, _ = coors.shape
    device = coors.device

    with torch.no_grad():
        if dist_sq is None:
            rel = rearrange(coors, "b i d -> b i () d") - rearrange(
                coors, "b j d -> b () j d"
            )
            rel = minimum_image(rel, _box_for_pairs(unitcell_lengths))
            dist_sq = (rel**2).sum(dim=-1)

        ranking = dist_sq.clone()

        if mask is not None:
            rank_mask = mask[:, :, None] * mask[:, None, :]
            ranking.masked_fill_(~rank_mask, 1e5)

        num_nearest = num_nearest_neighbors
        if adj_mat is not None:
            adj_mat = (
                repeat(adj_mat, "i j -> b i j", b=b) if adj_mat.dim() == 2 else adj_mat
            )
            adj_mat = adj_mat.clone()

            if only_sparse_neighbors:
                num_nearest = int(adj_mat.float().sum(dim=-1).max().item())
                valid_radius = 0.0

            self_mask = rearrange(
                torch.eye(n, device=device, dtype=torch.bool), "i j -> () i j"
            )
            adj_mat = adj_mat.masked_fill(self_mask, False)
            ranking.masked_fill_(self_mask, -1.0)
            ranking.masked_fill_(adj_mat, 0.0)

        num_nearest = min(max(num_nearest, 1), n)
        nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim=-1, largest=False)
        nbhd_mask = nbhd_ranking <= valid_radius

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
        num_nearest_neighbors: int = 0,
        only_sparse_neighbors: bool = False,
        valid_radius: float = float("inf"),
        m_pool_method: Literal["sum", "mean"] = "sum",
        dropout: float = 0.0,
        norm_feats: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        soft_edges: bool = False,
        coor_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
    ) -> None:
        """See :class:`E3GNN` for the meaning of the shared arguments.

        :param num_nearest_neighbors: kNN degree used for self-selection when no neighborhood
            is supplied by the network.
        :param only_sparse_neighbors: Use exactly the adjacency edges (k = max degree).
        :param valid_radius: Squared-distance radius marking a slot as a real edge.
        :param m_pool_method: How to pool messages onto nodes."""

        super().__init__()
        assert m_pool_method in {"sum", "mean"}
        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius
        self.m_pool_method = m_pool_method
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
        adj_mat: Tensor | None = None,
        nbhd_indices: Tensor | None = None,
        nbhd_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Message-passing update over a fixed neighborhood (or dense all-pairs if none).

        :param feats: Node features (B, N, D).
        :param coors: Node coordinates (B, N, 3).
        :param unitcell_lengths: Periodic box lengths (B, 3), or None.
        :param edges: Edge features — dense (B, N, N, E) or, with ``nbhd_indices``, (B, N, K, E).
        :param mask: Node validity mask (B, N), or None.
        :param adj_mat: Boolean adjacency, used only for self-selection when ``nbhd_indices`` is None.
        :param nbhd_indices: Precomputed neighbor indices (B, N, K), or None.
        :param nbhd_mask: Precomputed edge-validity mask (B, N, K), or None.
        :return: Updated features (B, N, D) and coordinates (B, N, 3)."""

        box = _box_for_pairs(unitcell_lengths)
        use_nearest = self.num_nearest_neighbors > 0 or self.only_sparse_neighbors

        if nbhd_indices is None and use_nearest:
            nbhd_indices, nbhd_mask = select_neighbors(
                coors,
                unitcell_lengths,
                adj_mat,
                mask,
                self.num_nearest_neighbors,
                self.only_sparse_neighbors,
                self.valid_radius,
            )
            if edges is not None:
                edges = batched_index_select(edges, nbhd_indices, dim=2)

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
        if self.m_pool_method == "mean":
            if edge_mask is not None:
                count = rearrange(edge_mask, "... -> ... ()").sum(dim=2).clamp(min=1)
                m_pooled = m_ij.sum(dim=2) / count
            else:
                m_pooled = m_ij.mean(dim=2)
        else:
            m_pooled = m_ij.sum(dim=2)

        feats_out = self.core.update_feats(feats, m_pooled)
        return feats_out, coors_out


class E3GNN(nn.Module):
    """E(3)-equivariant graph neural network on dense padded tensors.

    Handles periodic boxes (pass ``unitcell_lengths``) and open boundaries
    (``unitcell_lengths=None``). Fixed edges (bonds, component connectivity) are supplied through
    ``adj_mat`` with features in ``edges``; when ``distance_cutoff > 0`` extra within-cutoff edges
    are added and flagged with a trailing edge-feature column. The neighborhood is selected once
    and shared across layers."""

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
        **kwargs,
    ) -> None:
        """Build the network.

        :param depth: Number of message-passing layers.
        :param dim: Node feature dimensionality (features must already be this wide).
        :param encoding: Radial distance encoding for every layer.
        :param encoding_features: Number of basis functions / frequency bands for the encoding.
        :param cutoff: Radial length scale of the encoding.
        :param edge_dim: Edge-feature dimensionality (0 if no edge features).
        :param distance_cutoff: If > 0, add within-cutoff edges to the graph.
        :param kwargs: Extra keyword arguments forwarded to every :class:`DenseEGNNLayer`."""

        super().__init__()
        self.distance_cutoff = distance_cutoff
        # +1 for the within-cutoff flag column appended to the edge features at forward time.
        layer_edge_dim = edge_dim + (1 if distance_cutoff > 0 else 0)

        self.layers = nn.ModuleList(
            [
                DenseEGNNLayer(
                    dim=dim,
                    encoding=encoding,
                    encoding_features=encoding_features,
                    cutoff=cutoff,
                    edge_dim=layer_edge_dim,
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
        :param adj_mat: Boolean adjacency (B, N, N) or (N, N), or None.
        :param edges: Edge features (B, N, N, E), or None.
        :param mask: Node validity mask (B, N), or None.
        :param return_coor_changes: If True, also return the coordinate trajectory.
        :return: ``(feats, coors)`` or ``(feats, coors, coor_changes)``."""

        # Build the neighborhood once and share it. The distance pass only selects edges, so it
        # runs under no_grad and is not part of the backward graph.
        use_nearest = (
            self.layers[0].num_nearest_neighbors > 0
            or self.layers[0].only_sparse_neighbors
        )
        nbhd_indices: Tensor | None = None
        nbhd_mask: Tensor | None = None
        dist_adj_mat: Tensor | None = None
        dist_sq: Tensor | None = None

        if self.distance_cutoff > 0:
            with torch.no_grad():
                rel = rearrange(coors, "b i d -> b i () d") - rearrange(
                    coors, "b j d -> b () j d"
                )
                rel = minimum_image(rel, _box_for_pairs(unitcell_lengths))
                dist_sq = (rel**2).sum(dim=-1)
                dist_adj_mat = dist_sq < self.distance_cutoff**2
            if adj_mat is None:
                adj_mat = dist_adj_mat
            else:
                adj_mat = adj_mat.bool()
                if adj_mat.dim() == 2:
                    adj_mat = adj_mat[None]
                adj_mat = adj_mat | dist_adj_mat

        if use_nearest:
            nbhd_indices, nbhd_mask = select_neighbors(
                coors,
                unitcell_lengths,
                adj_mat,
                mask,
                self.layers[0].num_nearest_neighbors,
                self.layers[0].only_sparse_neighbors,
                self.layers[0].valid_radius,
                dist_sq=dist_sq,
            )
            if edges is not None:
                edges = batched_index_select(edges, nbhd_indices, dim=2)
            if dist_adj_mat is not None:
                flag = batched_index_select(
                    dist_adj_mat.unsqueeze(-1), nbhd_indices, dim=2
                ).float()
                edges = flag if edges is None else torch.cat([edges, flag], dim=-1)
        elif dist_adj_mat is not None:
            flag = dist_adj_mat.float().unsqueeze(-1)
            edges = flag if edges is None else torch.cat([edges, flag], dim=-1)

        coor_changes = [coors]
        for layer in self.layers:
            feats, coors = layer(
                feats,
                coors,
                unitcell_lengths,
                edges=edges,
                mask=mask,
                adj_mat=adj_mat,
                nbhd_indices=nbhd_indices,
                nbhd_mask=nbhd_mask,
            )
            coor_changes.append(coors)

        if return_coor_changes:
            return feats, coors, coor_changes
        return feats, coors
