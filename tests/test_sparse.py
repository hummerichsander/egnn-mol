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
@pytest.mark.parametrize("graph", ["bonds", "radius"])
@pytest.mark.parametrize("envelope", [False, True])
def test_cross_backbone_agreement(periodic, edge_dim, tripp, graph, envelope):
    """Dense and sparse agree exactly with shared weights and the same unified graph.

    Swept over static-bond vs internal-radius graphs, open/periodic boundaries, edge features, and
    the E(3)/SE(3) term — the definitive proof that the two backbones implement one function."""
    if graph == "radius" and edge_dim:
        pytest.skip("dynamic edges carry no features; edge_dim only applies to static bonds")
    if envelope and graph != "radius":
        pytest.skip("the envelope tapers at distance_cutoff, which only the radius graph has")

    torch.manual_seed(2)
    n, dim, depth = 7, 8, 2
    x = torch.rand(n, 3) * 4.0
    h_node = torch.randn(n, dim)
    box_row = torch.tensor([4.0, 4.5, 3.5])
    distance_cutoff = 2.5 if graph == "radius" else 0.0

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
    if graph == "bonds":
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
