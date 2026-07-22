import math

import pytest
import torch

from egnn_mol import E3GNN, has_pyg
from conftest import rotation_z

pytestmark = pytest.mark.skipif(not has_pyg(), reason="requires torch-geometric ([pyg] extra)")


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
    coors = torch.randn(10, 3)
    edges = radius_graph_pbc(coors, cutoff=1.5)
    d = (coors[edges[0]] - coors[edges[1]]).norm(dim=-1)
    assert (d < 1.5).all()
    assert (edges[0] != edges[1]).all()  # no self-loops by default

    knn = knn_graph_pbc(coors, k=3)
    assert knn.shape[1] == 10 * 3


def test_sparse_rotation_and_translation(compact_system):
    from egnn_mol import GeometricEGNN

    feats, coors, _ = compact_system
    feats, coors = feats[0], coors[0]  # single graph, drop batch axis
    n = coors.shape[0]
    edge_index = full_edge_index(torch.arange(n), include_self=False)
    net = GeometricEGNN(depth=2, dim=8, m_dim=8).eval()

    def run(c):
        out = net(torch.cat([c, feats], dim=-1), edge_index)
        return out[:, :3], out[:, 3:]

    R = rotation_z(math.pi / 4)
    centroid = coors.mean(0, keepdim=True)
    coors_rot = (coors - centroid) @ R.T + centroid
    delta = torch.tensor([2.0, -1.0, 0.5])
    with torch.no_grad():
        c_out, f_out = run(coors)
        c_rot, f_rot = run(coors_rot)
        c_tr, f_tr = run(coors + delta)

    assert torch.allclose(f_out, f_rot, atol=1e-5)                       # feature rotation invariance
    assert torch.allclose(c_rot - coors_rot, (c_out - coors) @ R.T, atol=1e-5)  # velocity equivariance
    assert torch.allclose(f_out, f_tr, atol=1e-5)                       # translation invariance
    assert torch.allclose(c_out - coors, c_tr - (coors + delta), atol=1e-5)


def test_ragged_batch_no_leakage():
    """A batch of two different-size graphs equals running each graph alone."""
    from torch_geometric.utils import scatter  # noqa: F401  (ensures PyG import path)

    from egnn_mol import GeometricEGNN

    torch.manual_seed(1)
    na, nb = 5, 8
    ca, fa = torch.randn(na, 3), torch.randn(na, 8)
    cb, fb = torch.randn(nb, 3), torch.randn(nb, 8)
    net = GeometricEGNN(depth=2, dim=8, m_dim=8, norm_feats=False).eval()

    ea = full_edge_index(torch.arange(na), include_self=False)
    eb = full_edge_index(torch.arange(nb), include_self=False)

    with torch.no_grad():
        out_a = net(torch.cat([ca, fa], -1), ea)
        out_b = net(torch.cat([cb, fb], -1), eb)

        coors = torch.cat([ca, cb], 0)
        feats = torch.cat([fa, fb], 0)
        batch = torch.cat([torch.zeros(na, dtype=torch.long), torch.ones(nb, dtype=torch.long)])
        edge_index = torch.cat([ea, eb + na], dim=1)
        out = net(torch.cat([coors, feats], -1), edge_index, batch=batch)

    assert torch.allclose(out[:na], out_a, atol=1e-5)
    assert torch.allclose(out[na:], out_b, atol=1e-5)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("edge_dim", [0, 3])
@pytest.mark.parametrize("tripp", [0, 2])
def test_cross_backbone_agreement(periodic, edge_dim, tripp):
    """Dense (all-pairs incl. self) and sparse (same graph) agree exactly with shared weights.

    Swept over open/periodic boundaries, presence of edge features, and the E(3)/SE(3) term —
    the definitive proof that the two backbones implement the same function."""
    from egnn_mol import GeometricEGNN

    torch.manual_seed(2)
    n, dim, depth = 7, 8, 2
    coors = torch.rand(n, 3) * 4.0
    feats = torch.randn(n, dim)
    box = torch.tensor([4.0, 4.5, 3.5]) if periodic else None

    dense = E3GNN(
        depth=depth, dim=dim, m_dim=8, edge_dim=edge_dim, tripp_num_layers=tripp
    ).eval()
    sparse = GeometricEGNN(
        depth=depth, dim=dim, m_dim=8, edge_dim=edge_dim, tripp_num_layers=tripp
    ).eval()
    for dl, sl in zip(dense.layers, sparse.layers):
        sl.core.load_state_dict(dl.core.state_dict())

    edge_index = full_edge_index(torch.arange(n), include_self=True)
    dst, src = edge_index[1], edge_index[0]

    dense_edges = sparse_edge_attr = None
    if edge_dim:
        dense_edges = torch.randn(1, n, n, edge_dim)
        sparse_edge_attr = dense_edges[0, dst, src]  # (E, edge_dim): dense[b, i=center, j=neighbor]

    dense_box = box[None] if periodic else None
    sparse_box = box.expand(n, 3) if periodic else None

    with torch.no_grad():
        f_d, c_d = dense(feats[None], coors[None], dense_box, edges=dense_edges)
        out_s = sparse(
            torch.cat([coors, feats], -1),
            edge_index,
            edge_attr=sparse_edge_attr,
            unitcell_lengths=sparse_box,
        )

    assert torch.allclose(c_d[0], out_s[:, :3], atol=1e-5)
    assert torch.allclose(f_d[0], out_s[:, 3:], atol=1e-5)
