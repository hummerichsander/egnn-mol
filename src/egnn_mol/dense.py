from collections.abc import Sequence
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from .encodings import Encoding, polynomial_envelope
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


def _box_for_pairs(box: Tensor | None) -> Tensor | None:
    """Reshape per-graph box lengths (B, 3) to (B, 1, 1, 3) for pairwise broadcasting."""
    if box is None:
        return None
    return box.view(box.shape[0], 1, 1, 3)


def build_neighborhood(
    x: Tensor,
    box: Tensor | None,
    adj_mat: Tensor | None,
    mask: Tensor | None,
    distance_cutoff: float,
    num_nearest_neighbors: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a padded neighborhood from static bonds and internal distance-based edges.

    The graph is the union of the static ``adj_mat`` edges, a radius graph (``distance_cutoff``),
    and a kNN graph (``num_nearest_neighbors``), all under the minimum-image convention and with
    self-loops removed. It is derived once from the input positions and shared across layers.

    :param x: Node positions (B, N, 3).
    :param box: Periodic box lengths (B, 3), or None.
    :param adj_mat: Static boolean adjacency (B, N, N) or (N, N), or None.
    :param mask: Node validity mask (B, N), or None.
    :param distance_cutoff: Radius cutoff for dynamic edges (0 disables).
    :param num_nearest_neighbors: kNN degree for dynamic edges (0 disables).
    :return: Neighbor indices (B, N, K), a boolean edge-validity mask (B, N, K), and a mask
        (B, N, K) marking the edges that came from ``adj_mat`` and so need no cutoff taper."""

    b, n, _ = x.shape
    device = x.device
    eye = torch.eye(n, dtype=torch.bool, device=device)[None]

    with torch.no_grad():
        rel = rearrange(x, "b i d -> b i () d") - rearrange(x, "b j d -> b () j d")
        dist_sq = (minimum_image(rel, _box_for_pairs(box)) ** 2).sum(dim=-1)

        pair_valid = torch.ones(b, n, n, dtype=torch.bool, device=device)
        if mask is not None:
            pair_valid = mask[:, :, None] & mask[:, None, :]

        graph = torch.zeros(b, n, n, dtype=torch.bool, device=device)
        static = torch.zeros(b, n, n, dtype=torch.bool, device=device)
        active = adj_mat is not None or distance_cutoff > 0 or num_nearest_neighbors > 0

        if adj_mat is not None:
            static = (
                repeat(adj_mat.bool(), "i j -> b i j", b=b)
                if adj_mat.dim() == 2
                else adj_mat.bool()
            )
            graph = graph | static
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
        nbhd_static = torch.gather(static & ~eye & pair_valid, -1, nbhd_indices)

    return nbhd_indices, nbhd_mask, nbhd_static


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
        norm_h_node: bool = False,
        norm_displacement: bool = False,
        norm_displacement_scale_init: float = 1.0,
        soft_edges: bool = False,
        x_weights_clamp_value: float | None = None,
        tripp_num_layers: int = 0,
        mlp_depth: int = 1,
    ) -> None:
        """See :class:`EGNN` for the meaning of the arguments."""

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
            norm_h_node=norm_h_node,
            norm_displacement=norm_displacement,
            norm_displacement_scale_init=norm_displacement_scale_init,
            dropout=dropout,
            x_weights_clamp_value=x_weights_clamp_value,
            tripp_num_layers=tripp_num_layers,
            mlp_depth=mlp_depth,
        )

    def forward(
        self,
        h_node: Tensor,
        x: Tensor,
        h_edge: Tensor | None = None,
        mask: Tensor | None = None,
        box: Tensor | None = None,
        nbhd_indices: Tensor | None = None,
        nbhd_mask: Tensor | None = None,
        env: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Message-passing update over a neighborhood (or all-pairs, self-excluded, if none given).

        :param h_node: Node features (B, N, dim).
        :param x: Node positions (B, N, 3).
        :param h_edge: Edge features — dense (B, N, N, edge_dim) or, with ``nbhd_indices``,
            (B, N, K, edge_dim).
        :param mask: Node validity mask (B, N), or None.
        :param box: Periodic box lengths (B, 3), or None.
        :param nbhd_indices: Precomputed neighbor indices (B, N, K), or None for all-pairs.
        :param nbhd_mask: Precomputed edge-validity mask (B, N, K), or None.
        :param env: Per-edge cutoff envelope (B, N, K, 1), or None for an untapered update.
        :return: Updated features (B, N, dim) and positions (B, N, 3)."""

        box_pairs = _box_for_pairs(box)
        b, n, _ = h_node.shape

        if nbhd_indices is not None:
            x_j = batched_index_select(x, nbhd_indices, dim=1)
            rel_x = minimum_image(rearrange(x, "b i d -> b i () d") - x_j, box_pairs)
            h_node_j = batched_index_select(h_node, nbhd_indices, dim=1)
        else:
            rel_x = rearrange(x, "b i d -> b i () d") - rearrange(
                x, "b j d -> b () j d"
            )
            rel_x = minimum_image(rel_x, box_pairs)
            h_node_j = rearrange(h_node, "b j d -> b () j d")

        dist = squared_distance(rel_x).clamp(min=1e-8).sqrt()
        h_node_i = rearrange(h_node, "b i d -> b i () d")
        h_node_i, h_node_j = torch.broadcast_tensors(h_node_i, h_node_j)

        m_ij = self.core.message(h_node_i, h_node_j, dist, h_edge)

        edge_mask = self._edge_mask(mask, nbhd_indices, nbhd_mask, b, n, h_node.device)

        # applied to the displacement and the message, the two quantities every aggregation
        # here is built from, so an edge fades out of all three at once.
        normed = self.core.normalize_rel(rel_x)
        if env is not None:
            normed, m_ij = env * normed, env * m_ij

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
            weight = self.core.x_weight(m_ij, chi_i, chi_j)
        else:
            weight = self.core.x_weight(m_ij)

        delta = weight * normed
        if edge_mask is not None:
            delta = delta.masked_fill(~rearrange(edge_mask, "... -> ... ()"), 0.0)
        x_out = x + delta.sum(dim=2)

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

        h_node_out = self.core.update_h_node(h_node, m_pooled)
        return h_node_out, x_out

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


class EGNN(nn.Module):
    """E(3)-equivariant graph neural network on dense padded tensors.

    Handles periodic boxes (pass ``box``) and open boundaries. The neighborhood is the union of
    static bonds (``adj_mat`` / ``h_edge``, given from outside) and internal distance-based
    edges (``distance_cutoff`` radius and/or ``num_nearest_neighbors`` kNN), with self-loops
    excluded; with none of these it is all-pairs. It is built once and shared across layers."""

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
            static adjacency alone. None (the default) gives every layer the full neighborhood.
        :param aggr: Message aggregation onto nodes ("sum" or "mean").
        :param envelope: Taper every edge's contribution to zero at ``distance_cutoff``, so the
            field stays C^1 where an edge enters or leaves the radius graph. Off by default: a
            network trained without it computes a different function, so turning it on silently
            would change what an existing checkpoint evaluates.
        :param envelope_exponent: Polynomial exponent of that envelope.
        :param kwargs: Extra keyword arguments forwarded to every :class:`DenseEGNNLayer`."""

        super().__init__()
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
        h_node: Tensor,
        x: Tensor,
        adj_mat: Tensor | None = None,
        h_edge: Tensor | None = None,
        mask: Tensor | None = None,
        box: Tensor | None = None,
        return_x_changes: bool = False,
    ):
        """Run all message-passing layers.

        :param h_node: Node features (B, N, dim).
        :param x: Node positions (B, N, 3).
        :param adj_mat: Static boolean adjacency (B, N, N) or (N, N), or None.
        :param h_edge: Static edge features (B, N, N, edge_dim), or None.
        :param mask: Node validity mask (B, N), or None.
        :param box: Periodic box lengths (B, 3), or None.
        :param return_x_changes: If True, also return the position trajectory.
        :return: ``(h_node, x)`` or ``(h_node, x, x_changes)``."""

        nbhd_indices: Tensor | None = None
        nbhd_mask: Tensor | None = None
        nbhd_static: Tensor | None = None
        if (
            adj_mat is not None
            or self.distance_cutoff > 0
            or self.num_nearest_neighbors > 0
        ):
            nbhd_indices, nbhd_mask, nbhd_static = build_neighborhood(
                x,
                box,
                adj_mat,
                mask,
                self.distance_cutoff,
                self.num_nearest_neighbors,
            )
            if h_edge is not None:
                h_edge = batched_index_select(h_edge, nbhd_indices, dim=2)

        env = self.edge_envelope(x, box, nbhd_indices, nbhd_static)

        # a static-only layer drops the dynamic entries by masking them out of the aggregation;
        # the envelope is already one on the static ones, so it needs no slice of its own.
        static_mask = (
            nbhd_mask & nbhd_static
            if nbhd_mask is not None and nbhd_static is not None
            else nbhd_mask
        )

        x_changes = [x]
        for index, layer in enumerate(self.layers):
            h_node, x = layer(
                h_node,
                x,
                h_edge=h_edge,
                mask=mask,
                box=box,
                nbhd_indices=nbhd_indices,
                nbhd_mask=nbhd_mask if self.reads_dynamic(index) else static_mask,
                env=env,
            )
            x_changes.append(x)

        if return_x_changes:
            return h_node, x, x_changes
        return h_node, x

    def reads_dynamic(self, layer: int) -> bool:
        """Whether a layer sees the dynamic edges on top of the static adjacency.

        :param layer: Index of the layer.
        :return: True if it does."""

        return self.dynamic_layers is None or layer in self.dynamic_layers

    def edge_envelope(
        self,
        x: Tensor,
        box: Tensor | None,
        nbhd_indices: Tensor | None,
        nbhd_static: Tensor | None = None,
    ) -> Tensor | None:
        """The cutoff envelope of every edge, evaluated on the positions the edges were built from.

        Evaluated **once**, on the input positions, and shared by every layer -- exactly as the
        neighborhood is. Letting each layer taper by its own distances instead would break the
        very continuity the envelope exists for: an edge appears at ``distance_cutoff`` with
        zero weight, but by the second layer the pair has moved, so it would re-enter with a
        finite weight and the field would jump as the neighborhood changed.

        :param x: Node positions (B, N, 3), the ones the neighborhood was built from.
        :param box: Periodic box lengths (B, 3), or None.
        :param nbhd_indices: Neighbor indices (B, N, K), or None for all-pairs.
        :param nbhd_static: Mask (B, N, K) of static edges to exempt from the taper, or None.
        :return: Envelope weights (B, N, K, 1) or (B, N, N, 1), or None when none is configured."""

        if self.envelope_cutoff is None:
            return None

        box_pairs = _box_for_pairs(box)
        if nbhd_indices is not None:
            x_j = batched_index_select(x, nbhd_indices, dim=1)
            rel_x = minimum_image(rearrange(x, "b i d -> b i () d") - x_j, box_pairs)
        else:
            rel_x = minimum_image(
                rearrange(x, "b i d -> b i () d") - rearrange(x, "b j d -> b () j d"),
                box_pairs,
            )
        dist = squared_distance(rel_x).clamp(min=1e-8).sqrt()
        env = polynomial_envelope(dist, self.envelope_cutoff, self.envelope_exponent)

        # a static edge set is position-independent, so no edge enters or leaves at the cutoff and
        # there is nothing to taper; see the sparse backbone's `edge_envelope`.
        if nbhd_static is not None:
            env = torch.where(nbhd_static[..., None], torch.ones_like(env), env)

        return env
