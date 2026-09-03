import torch
from torch import Tensor


def _binary_sparse(edge_index: Tensor, num_nodes: int) -> Tensor:
    """A coalesced sparse matrix carrying a one at every given index.

    :param edge_index: Indices (2, E) of the entries to set.
    :param num_nodes: Side length N of the square matrix.
    :return: A coalesced sparse (N, N) tensor."""

    values = torch.ones(edge_index.shape[1], device=edge_index.device)
    return torch.sparse_coo_tensor(
        edge_index, values, (num_nodes, num_nodes)
    ).coalesce()


def _adjacency(edge_index: Tensor, num_nodes: int) -> Tensor:
    """The symmetrized adjacency of a graph, carrying every self-loop.

    :param edge_index: Edge connectivity (2, E); direction is ignored.
    :param num_nodes: Number of nodes N.
    :return: A coalesced sparse (N, N) tensor of ones."""

    loops = torch.arange(num_nodes, device=edge_index.device).expand(2, num_nodes)

    return _binary_sparse(
        torch.cat([edge_index, edge_index.flip(0), loops], dim=1), num_nodes
    )


def composed_closure(edge_indices: list[Tensor], num_nodes: int) -> Tensor:
    """Reachability through a sequence of graphs, one per hop, self-loops included.

    This is the sparsity pattern of a stack whose layers do not all read the same edges: hop
    ``l`` carries information one hop of ``edge_indices[l]``, so the update at node ``i`` can
    only depend on nodes reached by walking one hop through each graph in turn. A composition
    rather than a power, which is what lets a layer restricted to a small graph cost a small
    ball while a layer on the full neighborhood still reaches all of its own neighbors.

    Symmetrized at the end, and only there. A product of symmetric matrices is not itself
    symmetric -- ``support(AB) = support(BA)^T`` -- so which end of the stack the walk starts
    from decides the direction of the entries. Closing over both is what keeps the result a
    valid colouring input (a colour class must be independent in *either* direction, since a
    seeded neighbor contaminates the read-out whichever way the dependence runs), and it makes
    the order of the sequence immaterial.

    :param edge_indices: One edge index (2, E) per hop; direction is ignored.
    :param num_nodes: Number of nodes N.
    :return: Edge index (2, E') of the closure, symmetric and carrying every self-loop."""

    if not edge_indices:
        raise ValueError("at least one hop is needed, got none.")

    # the same graph repeats across most of a stack, and building its adjacency is the expensive
    # part; `hop_closure` is the extreme case of one graph repeated `hops` times.
    closure = _adjacency(edge_indices[0], num_nodes)
    cache = {id(edge_indices[0]): closure}

    for edge_index in edge_indices[1:]:
        if (key := id(edge_index)) not in cache:
            cache[key] = _adjacency(edge_index, num_nodes)

        # re-binarize each step: the products count paths, which overflows float32 quickly.
        product = torch.sparse.mm(closure, cache[key]).coalesce()
        closure = _binary_sparse(product.indices(), num_nodes)

    indices = closure.indices()

    return _binary_sparse(
        torch.cat([indices, indices.flip(0)], dim=1), num_nodes
    ).indices()


def hop_closure(edge_index: Tensor, num_nodes: int, hops: int) -> Tensor:
    """Reachability within ``hops`` hops of one graph, self-loops included.

    This is the sparsity pattern of a stacked backbone whose layers all read the same edges. One
    layer carries information one hop, so after ``hops`` layers the update at node ``i`` can only
    depend on nodes at graph distance ``<= hops`` from it, and every other block of the Jacobian
    is structurally zero. See :func:`composed_closure` for a stack that reads different edges in
    different layers.

    Kept sparse throughout: a packed batch has ``num_nodes = B * N``, whose dense adjacency is
    quadratic in the batch size.

    :param edge_index: Edge connectivity (2, E); direction is ignored.
    :param num_nodes: Number of nodes N.
    :param hops: Number of hops to close over; at least one.
    :return: Edge index (2, E') of the closure, carrying every self-loop."""

    if hops < 1:
        raise ValueError(f"hops must be at least one, got {hops}.")

    return composed_closure([edge_index] * hops, num_nodes)


def greedy_colouring(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Colour a graph largest-degree-first so that no edge joins two nodes of one colour.

    Each colour class is an independent set, which is what lets one derivative pass read a whole
    group of diagonal blocks at once: with no two seeded nodes adjacent in the Jacobian's
    sparsity pattern, the off-diagonal couplings that share a row are structurally zero and
    cannot contaminate the read-out.

    The greedy count is not the chromatic number, but it is the number of passes a compressed
    Jacobian actually costs, which is the quantity of interest.

    :param edge_index: Edge connectivity (2, E) of the pattern to colour; direction is ignored.
    :param num_nodes: Number of nodes N.
    :return: Colours (N,), numbered contiguously from zero."""

    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]

    order = torch.argsort(src, stable=True)
    dst = dst[order]
    counts = torch.bincount(src, minlength=num_nodes)
    starts = torch.cumsum(counts, dim=0) - counts

    # the sweep is inherently sequential, so it runs on python lists; going back to the device
    # per node would cost far more than the colouring itself.
    neighbors, offsets, degrees = dst.tolist(), starts.tolist(), counts.tolist()
    colours = [-1] * num_nodes

    for node in torch.argsort(counts, descending=True, stable=True).tolist():
        start, degree = offsets[node], degrees[node]
        used = {colours[j] for j in neighbors[start : start + degree]}
        colour = 0
        while colour in used:
            colour += 1
        colours[node] = colour

    return torch.tensor(colours, dtype=torch.long, device=edge_index.device)
