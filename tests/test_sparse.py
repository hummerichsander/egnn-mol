import math

import pytest
import torch

from egnn_mol import EGNN, GeometricEGNN
from conftest import rotation_z


def full_edge_index(nodes: torch.Tensor, include_self: bool) -> torch.Tensor:
    """All ordered pairs among ``nodes`` as edge_index [source/neighbor, target/center]."""
    src = nodes.repeat(len(nodes))
    dst = nodes.repeat_interleave(len(nodes))
    edge_index = torch.stack([src, dst], dim=0)
    if not include_self:
        edge_index = edge_index[:, src != dst]
    return edge_index


def test_radius_and_knn_graph():
    from egnn_mol import knn_graph_pbc, radius_graph_pbc

    torch.manual_seed(0)
    x = torch.randn(10, 3)
    edges = radius_graph_pbc(x, cutoff=1.5)
    d = (x[edges[0]] - x[edges[1]]).norm(dim=-1)
    assert (d < 1.5).all()
    assert (edges[0] != edges[1]).all()  # no self-loops by default

    knn = knn_graph_pbc(x, k=3)
    assert knn.shape[1] == 10 * 3


def test_sparse_rotation_and_translation(compact_system):
    h_node, x, _ = compact_system
    h_node, x = h_node[0], x[0]  # single graph, drop batch axis
    n = x.shape[0]
    edge_index = full_edge_index(torch.arange(n), include_self=False)
    net = GeometricEGNN(depth=2, dim=8, m_dim=8).eval()

    def run(p):
        return net(h_node, p, edge_index=edge_index)

    R = rotation_z(math.pi / 4)
    centroid = x.mean(0, keepdim=True)
    x_rot = (x - centroid) @ R.T + centroid
    delta = torch.tensor([2.0, -1.0, 0.5])
    with torch.no_grad():
        h_node_out, x_out = run(x)
        h_node_rot, x_rot_out = run(x_rot)
        h_node_tr, x_tr = run(x + delta)

    assert torch.allclose(h_node_out, h_node_rot, atol=1e-5)                              # feature rotation invariance
    assert torch.allclose(x_rot_out - x_rot, (x_out - x) @ R.T, atol=1e-5)  # velocity equivariance
    assert torch.allclose(h_node_out, h_node_tr, atol=1e-5)                              # translation invariance
    assert torch.allclose(x_out - x, x_tr - (x + delta), atol=1e-5)


def test_internal_graph_pbc():
    """The sparse backbone builds its own periodic graph from distance_cutoff / num_nearest."""
    torch.manual_seed(3)
    n = 12
    h_node, x = torch.randn(n, 8), torch.rand(n, 3) * 3.0
    box = torch.full((n, 3), 3.0)
    for kwargs in (dict(distance_cutoff=1.5), dict(num_nearest_neighbors=4)):
        net = GeometricEGNN(depth=2, dim=8, m_dim=8, **kwargs).eval()
        with torch.no_grad():
            h_node_out, x_out = net(h_node, x, box=box)  # edge_index=None -> built internally
        assert h_node_out.shape == h_node.shape and torch.isfinite(x_out).all()


def test_internal_graph_open():
    """Open-boundary dynamic graph uses torch_cluster."""
    torch.manual_seed(3)
    n = 12
    h_node, x = torch.randn(n, 8), torch.rand(n, 3) * 3.0
    for kwargs in (dict(distance_cutoff=1.5), dict(num_nearest_neighbors=4)):
        net = GeometricEGNN(depth=2, dim=8, m_dim=8, **kwargs).eval()
        with torch.no_grad():
            h_node_out, x_out = net(h_node, x)  # box=None -> torch_cluster
        assert h_node_out.shape == h_node.shape and torch.isfinite(x_out).all()


def test_static_union_dynamic():
    """Providing bonds AND a distance_cutoff unions the two edge sets (periodic path)."""
    torch.manual_seed(4)
    n = 10
    h_node, x = torch.randn(n, 8), torch.rand(n, 3) * 3.0
    box = torch.full((n, 3), 3.0)
    bonds = full_edge_index(torch.arange(n), include_self=False)[:, :6]  # a few static edges
    net = GeometricEGNN(depth=2, dim=8, m_dim=8, distance_cutoff=1.5).eval()
    with torch.no_grad():
        h_node_out, x_out = net(h_node, x, edge_index=bonds, box=box)
    assert h_node_out.shape == h_node.shape and torch.isfinite(x_out).all()


def test_ragged_batch_no_leakage():
    """A batch of two different-size graphs equals running each graph alone."""
    torch.manual_seed(1)
    na, nb = 5, 8
    h_node_a, xa = torch.randn(na, 8), torch.randn(na, 3)
    h_node_b, xb = torch.randn(nb, 8), torch.randn(nb, 3)
    net = GeometricEGNN(depth=2, dim=8, m_dim=8, norm_h_node=False).eval()

    ea = full_edge_index(torch.arange(na), include_self=False)
    eb = full_edge_index(torch.arange(nb), include_self=False)

    with torch.no_grad():
        h_node_a_out, xa_out = net(h_node_a, xa, edge_index=ea)
        h_node_b_out, xb_out = net(h_node_b, xb, edge_index=eb)

        h_node = torch.cat([h_node_a, h_node_b], 0)
        x = torch.cat([xa, xb], 0)
        batch = torch.cat([torch.zeros(na, dtype=torch.long), torch.ones(nb, dtype=torch.long)])
        edge_index = torch.cat([ea, eb + na], dim=1)
        h_node_out, x_out = net(h_node, x, edge_index=edge_index, batch=batch)

    assert torch.allclose(h_node_out[:na], h_node_a_out, atol=1e-5)
    assert torch.allclose(x_out[:na], xa_out, atol=1e-5)
    assert torch.allclose(h_node_out[na:], h_node_b_out, atol=1e-5)
    assert torch.allclose(x_out[na:], xb_out, atol=1e-5)


def _sparse_edges_from_adj(adj: torch.Tensor, dense_h_edge: torch.Tensor | None):
    """Convert a symmetric (N, N) adjacency into sparse [neighbor, center] edges + attrs.

    Matches the dense convention: for center i and neighbor j, the edge feature is
    ``dense_h_edge[0, i, j]``."""
    center, neighbor = adj.nonzero(as_tuple=True)  # adj[i, j] -> center i, neighbor j
    edge_index = torch.stack([neighbor, center], dim=0)
    h_edge = None if dense_h_edge is None else dense_h_edge[0, center, neighbor]
    return edge_index, h_edge


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("edge_dim", [0, 3])
@pytest.mark.parametrize("tripp", [0, 2])
@pytest.mark.parametrize("graph", ["bonds", "radius", "bonds+radius"])
@pytest.mark.parametrize("envelope", [False, True])
def test_cross_backbone_agreement(periodic, edge_dim, tripp, graph, envelope):
    """Dense and sparse agree exactly with shared weights and the same unified graph.

    Swept over static-bond vs internal-radius graphs, open/periodic boundaries, edge features, and
    the E(3)/SE(3) term — the definitive proof that the two backbones implement one function."""
    if graph == "radius" and edge_dim:
        pytest.skip("dynamic edges carry no features; edge_dim only applies to static bonds")
    if envelope and graph == "bonds":
        pytest.skip("the envelope tapers at distance_cutoff, which only a radius graph has")

    torch.manual_seed(2)
    n, dim, depth = 7, 8, 2
    x = torch.rand(n, 3) * 4.0
    h_node = torch.randn(n, dim)
    box_row = torch.tensor([4.0, 4.5, 3.5])
    distance_cutoff = 0.0 if graph == "bonds" else 2.5

    common = dict(
        depth=depth,
        dim=dim,
        m_dim=8,
        edge_dim=edge_dim,
        tripp_num_layers=tripp,
        distance_cutoff=distance_cutoff,
        envelope=envelope,
    )
    dense = EGNN(**common).eval()
    sparse = GeometricEGNN(**common).eval()
    for dl, sl in zip(dense.layers, sparse.layers):
        sl.core.load_state_dict(dl.core.state_dict())

    adj_mat = dense_h_edge = edge_index = h_edge = None
    if graph != "radius":
        adj = torch.rand(n, n) > 0.4
        adj = (adj | adj.T) & ~torch.eye(n, dtype=torch.bool)
        adj_mat = adj
        if edge_dim:
            dense_h_edge = torch.randn(1, n, n, edge_dim)
        edge_index, h_edge = _sparse_edges_from_adj(adj, dense_h_edge)

    dense_box = box_row[None] if periodic else None
    sparse_box = box_row.expand(n, 3) if periodic else None

    with torch.no_grad():
        h_node_d, x_d = dense(h_node[None], x[None], adj_mat=adj_mat, h_edge=dense_h_edge, box=dense_box)
        h_node_s, x_s = sparse(h_node, x, edge_index=edge_index, h_edge=h_edge, box=sparse_box)

    assert torch.allclose(x_d[0], x_s, atol=1e-5)
    assert torch.allclose(h_node_d[0], h_node_s, atol=1e-5)


def test_cross_backbone_agreement_with_deep_mlps():
    """Both layer classes must thread ``mlp_depth``, or their state dicts stop being swappable."""
    torch.manual_seed(2)
    n, dim = 7, 8
    x = torch.rand(n, 3) * 4.0
    h_node = torch.randn(n, dim)

    common = dict(depth=2, dim=dim, m_dim=8, distance_cutoff=2.5, mlp_depth=3)
    dense = EGNN(**common).eval()
    sparse = GeometricEGNN(**common).eval()
    for dl, sl in zip(dense.layers, sparse.layers):
        sl.core.load_state_dict(dl.core.state_dict())

    with torch.no_grad():
        h_node_d, x_d = dense(h_node[None], x[None])
        h_node_s, x_s = sparse(h_node, x)

    assert torch.allclose(x_d[0], x_s, atol=1e-5)
    assert torch.allclose(h_node_d[0], h_node_s, atol=1e-5)


def randomized(net, seed: int = 0):
    """Overwrite the near-identity init, which otherwise leaves every variant equal to ``x``.

    :param net: The module to overwrite in place.
    :param seed: Seed of the weight draw.
    :return: The same module, in eval mode."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return net.eval()


def test_per_call_cutoff_matches_a_net_built_at_that_radius():
    """A per-call radius must reproduce the net that was constructed with it, weights aside.

    The definitive statement of the feature: the override changes the neighborhood and nothing
    else, so it cannot be distinguished from having built the net that way."""
    torch.manual_seed(4)
    n, dim = 9, 8
    x = torch.rand(n, 3) * 4.0
    h_node = torch.randn(n, dim)

    common = dict(depth=2, dim=dim, m_dim=8, envelope=True)
    built_at_1 = randomized(GeometricEGNN(**common, distance_cutoff=1.0))
    built_at_3 = GeometricEGNN(**common, distance_cutoff=3.0).eval()
    built_at_3.load_state_dict(built_at_1.state_dict())

    with torch.no_grad():
        overridden = built_at_1(h_node, x, distance_cutoff=3.0)
        constructed = built_at_3(h_node, x)
        unchanged = built_at_1(h_node, x)

    assert torch.allclose(overridden[0], constructed[0])
    assert torch.allclose(overridden[1], constructed[1])
    # the two radii must actually disagree, or the test above passes vacuously.
    assert not torch.allclose(overridden[1], unchanged[1])


def test_per_call_cutoff_matches_a_field_built_at_that_radius():
    """The same claim for ``RadialField``, whose envelope derivative must move with the radius too."""
    from egnn_mol import RadialField

    torch.manual_seed(5)
    n, dim = 9, 8
    x = torch.rand(n, 3) * 4.0
    h_node = torch.randn(n, dim)

    common = dict(dim=dim, encoding="gaussian", encoding_features=6, cutoff=3.0, m_dim=8)
    built_at_1 = randomized(RadialField(**common, distance_cutoff=1.0))
    built_at_3 = RadialField(**common, distance_cutoff=3.0).eval()
    built_at_3.load_state_dict(built_at_1.state_dict())

    with torch.no_grad():
        v, div = built_at_1(h_node, x, distance_cutoff=3.0)
        v_ref, div_ref = built_at_3(h_node, x)
        v_unchanged, _ = built_at_1(h_node, x)

    assert torch.allclose(v, v_ref)
    assert torch.allclose(div, div_ref)
    assert not torch.allclose(v, v_unchanged)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("tripp", [0, 2])
def test_cross_backbone_agreement_under_a_per_call_cutoff(periodic, tripp):
    """Dense and sparse must agree on what an overridden radius *means*, not just on the default.

    They build their neighborhoods by entirely different code paths -- a padded (B, N, K) gather
    against ``torch_cluster`` -- so agreement here is the sharpest check that the override lands
    in the same place on both."""
    torch.manual_seed(2)
    n, dim = 7, 8
    x = torch.rand(n, 3) * 4.0
    h_node = torch.randn(n, dim)
    box_row = torch.tensor([4.0, 4.5, 3.5])

    common = dict(
        depth=2, dim=dim, m_dim=8, tripp_num_layers=tripp, distance_cutoff=1.0, envelope=True
    )
    dense = randomized(EGNN(**common))
    sparse = GeometricEGNN(**common).eval()
    for dl, sl in zip(dense.layers, sparse.layers):
        sl.core.load_state_dict(dl.core.state_dict())

    dense_box = box_row[None] if periodic else None
    sparse_box = box_row.expand(n, 3) if periodic else None

    with torch.no_grad():
        h_d, x_d = dense(h_node[None], x[None], box=dense_box, distance_cutoff=2.5)
        h_s, x_s = sparse(h_node, x, box=sparse_box, distance_cutoff=2.5)
        _, x_default = sparse(h_node, x, box=sparse_box)

    assert torch.allclose(x_d[0], x_s, atol=1e-5)
    assert torch.allclose(h_d[0], h_s, atol=1e-5)
    assert not torch.allclose(x_s, x_default, atol=1e-5)
