import importlib.util
import math

import pytest
import torch

from egnn_mol import E3GNN, has_pyg
from conftest import rotation_z

pytestmark = pytest.mark.skipif(not has_pyg(), reason="requires torch-geometric ([pyg] extra)")

_HAS_TC = importlib.util.find_spec("torch_cluster") is not None
needs_tc = pytest.mark.skipif(not _HAS_TC, reason="open-boundary dynamic graphs need torch-cluster")


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
    pos = torch.randn(10, 3)
    edges = radius_graph_pbc(pos, cutoff=1.5)
    d = (pos[edges[0]] - pos[edges[1]]).norm(dim=-1)
    assert (d < 1.5).all()
    assert (edges[0] != edges[1]).all()  # no self-loops by default

    knn = knn_graph_pbc(pos, k=3)
    assert knn.shape[1] == 10 * 3


def test_sparse_rotation_and_translation(compact_system):
    from egnn_mol import GeometricEGNN

    x, pos, _ = compact_system
    x, pos = x[0], pos[0]  # single graph, drop batch axis
    n = pos.shape[0]
    edge_index = full_edge_index(torch.arange(n), include_self=False)
    net = GeometricEGNN(depth=2, dim=8, m_dim=8).eval()

    def run(p):
        return net(x, p, edge_index=edge_index)

    R = rotation_z(math.pi / 4)
    centroid = pos.mean(0, keepdim=True)
    pos_rot = (pos - centroid) @ R.T + centroid
    delta = torch.tensor([2.0, -1.0, 0.5])
    with torch.no_grad():
        x_out, pos_out = run(pos)
        x_rot, pos_rot_out = run(pos_rot)
        x_tr, pos_tr = run(pos + delta)

    assert torch.allclose(x_out, x_rot, atol=1e-5)                              # feature rotation invariance
    assert torch.allclose(pos_rot_out - pos_rot, (pos_out - pos) @ R.T, atol=1e-5)  # velocity equivariance
    assert torch.allclose(x_out, x_tr, atol=1e-5)                              # translation invariance
    assert torch.allclose(pos_out - pos, pos_tr - (pos + delta), atol=1e-5)


def test_internal_graph_pbc():
    """The sparse backbone builds its own periodic graph from distance_cutoff / num_nearest."""
    from egnn_mol import GeometricEGNN

    torch.manual_seed(3)
    n = 12
    x, pos = torch.randn(n, 8), torch.rand(n, 3) * 3.0
    box = torch.full((n, 3), 3.0)
    for kwargs in (dict(distance_cutoff=1.5), dict(num_nearest_neighbors=4)):
        net = GeometricEGNN(depth=2, dim=8, m_dim=8, **kwargs).eval()
        with torch.no_grad():
            x_out, pos_out = net(x, pos, box=box)  # edge_index=None -> built internally
        assert x_out.shape == x.shape and torch.isfinite(pos_out).all()


@needs_tc
def test_internal_graph_open():
    """Open-boundary dynamic graph uses torch_cluster."""
    from egnn_mol import GeometricEGNN

    torch.manual_seed(3)
    n = 12
    x, pos = torch.randn(n, 8), torch.rand(n, 3) * 3.0
    for kwargs in (dict(distance_cutoff=1.5), dict(num_nearest_neighbors=4)):
        net = GeometricEGNN(depth=2, dim=8, m_dim=8, **kwargs).eval()
        with torch.no_grad():
            x_out, pos_out = net(x, pos)  # box=None -> torch_cluster
        assert x_out.shape == x.shape and torch.isfinite(pos_out).all()


def test_static_union_dynamic():
    """Providing bonds AND a distance_cutoff unions the two edge sets (periodic path)."""
    from egnn_mol import GeometricEGNN

    torch.manual_seed(4)
    n = 10
    x, pos = torch.randn(n, 8), torch.rand(n, 3) * 3.0
    box = torch.full((n, 3), 3.0)
    bonds = full_edge_index(torch.arange(n), include_self=False)[:, :6]  # a few static edges
    net = GeometricEGNN(depth=2, dim=8, m_dim=8, distance_cutoff=1.5).eval()
    with torch.no_grad():
        x_out, pos_out = net(x, pos, edge_index=bonds, box=box)
    assert x_out.shape == x.shape and torch.isfinite(pos_out).all()


def test_ragged_batch_no_leakage():
    """A batch of two different-size graphs equals running each graph alone."""
    from egnn_mol import GeometricEGNN

    torch.manual_seed(1)
    na, nb = 5, 8
    xa, pa = torch.randn(na, 8), torch.randn(na, 3)
    xb, pb = torch.randn(nb, 8), torch.randn(nb, 3)
    net = GeometricEGNN(depth=2, dim=8, m_dim=8, norm_x=False).eval()

    ea = full_edge_index(torch.arange(na), include_self=False)
    eb = full_edge_index(torch.arange(nb), include_self=False)

    with torch.no_grad():
        xa_out, pa_out = net(xa, pa, edge_index=ea)
        xb_out, pb_out = net(xb, pb, edge_index=eb)

        x = torch.cat([xa, xb], 0)
        pos = torch.cat([pa, pb], 0)
        batch = torch.cat([torch.zeros(na, dtype=torch.long), torch.ones(nb, dtype=torch.long)])
        edge_index = torch.cat([ea, eb + na], dim=1)
        x_out, pos_out = net(x, pos, edge_index=edge_index, batch=batch)

    assert torch.allclose(x_out[:na], xa_out, atol=1e-5)
    assert torch.allclose(pos_out[:na], pa_out, atol=1e-5)
    assert torch.allclose(x_out[na:], xb_out, atol=1e-5)
    assert torch.allclose(pos_out[na:], pb_out, atol=1e-5)


def _sparse_edges_from_adj(adj: torch.Tensor, dense_edge_attr: torch.Tensor | None):
    """Convert a symmetric (N, N) adjacency into sparse [neighbor, center] edges + attrs.

    Matches the dense convention: for center i and neighbor j, the edge feature is
    ``dense_edge_attr[0, i, j]``."""
    center, neighbor = adj.nonzero(as_tuple=True)  # adj[i, j] -> center i, neighbor j
    edge_index = torch.stack([neighbor, center], dim=0)
    edge_attr = None if dense_edge_attr is None else dense_edge_attr[0, center, neighbor]
    return edge_index, edge_attr


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("edge_dim", [0, 3])
@pytest.mark.parametrize("tripp", [0, 2])
@pytest.mark.parametrize("graph", ["bonds", "radius"])
def test_cross_backbone_agreement(periodic, edge_dim, tripp, graph):
    """Dense and sparse agree exactly with shared weights and the same unified graph.

    Swept over static-bond vs internal-radius graphs, open/periodic boundaries, edge features, and
    the E(3)/SE(3) term — the definitive proof that the two backbones implement one function."""
    from egnn_mol import GeometricEGNN

    if graph == "radius" and edge_dim:
        pytest.skip("dynamic edges carry no features; edge_dim only applies to static bonds")
    if graph == "radius" and not periodic and not _HAS_TC:
        pytest.skip("open-boundary radius graph needs torch-cluster")

    torch.manual_seed(2)
    n, dim, depth = 7, 8, 2
    pos = torch.rand(n, 3) * 4.0
    x = torch.randn(n, dim)
    box_row = torch.tensor([4.0, 4.5, 3.5])
    distance_cutoff = 2.5 if graph == "radius" else 0.0

    common = dict(
        depth=depth,
        dim=dim,
        m_dim=8,
        edge_dim=edge_dim,
        tripp_num_layers=tripp,
        distance_cutoff=distance_cutoff,
    )
    dense = E3GNN(**common).eval()
    sparse = GeometricEGNN(**common).eval()
    for dl, sl in zip(dense.layers, sparse.layers):
        sl.core.load_state_dict(dl.core.state_dict())

    adj_mat = dense_edge_attr = edge_index = edge_attr = None
    if graph == "bonds":
        adj = torch.rand(n, n) > 0.4
        adj = (adj | adj.T) & ~torch.eye(n, dtype=torch.bool)
        adj_mat = adj
        if edge_dim:
            dense_edge_attr = torch.randn(1, n, n, edge_dim)
        edge_index, edge_attr = _sparse_edges_from_adj(adj, dense_edge_attr)

    dense_box = box_row[None] if periodic else None
    sparse_box = box_row.expand(n, 3) if periodic else None

    with torch.no_grad():
        x_d, pos_d = dense(x[None], pos[None], adj_mat=adj_mat, edge_attr=dense_edge_attr, box=dense_box)
        x_s, pos_s = sparse(x, pos, edge_index=edge_index, edge_attr=edge_attr, box=sparse_box)

    assert torch.allclose(pos_d[0], pos_s, atol=1e-5)
    assert torch.allclose(x_d[0], x_s, atol=1e-5)
