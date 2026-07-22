import torch

from egnn_mol import E3GNN


def test_forward_shapes_open_boundary(system):
    feats, coors, _ = system
    net = E3GNN(depth=2, dim=8, m_dim=8).eval()
    with torch.no_grad():
        f_out, c_out = net(feats, coors)
    assert f_out.shape == feats.shape
    assert c_out.shape == coors.shape


def test_return_coor_changes(system):
    feats, coors, L = system
    net = E3GNN(depth=3, dim=8, m_dim=8).eval()
    with torch.no_grad():
        _, _, changes = net(feats, coors, L, return_coor_changes=True)
    assert len(changes) == 4  # input + one per layer
    assert torch.allclose(changes[0], coors)


def test_radius_graph_runs(system):
    feats, coors, L = system
    net = E3GNN(depth=2, dim=8, m_dim=8, distance_cutoff=4.0).eval()
    with torch.no_grad():
        f_out, c_out = net(feats, coors, L)
    assert torch.isfinite(f_out).all() and torch.isfinite(c_out).all()


def test_knn_graph_runs(system):
    feats, coors, L = system
    net = E3GNN(depth=2, dim=8, m_dim=8, num_nearest_neighbors=3).eval()
    with torch.no_grad():
        f_out, c_out = net(feats, coors, L)
    assert f_out.shape == feats.shape and c_out.shape == coors.shape


def test_static_bonds_only(system):
    feats, coors, L = system
    n = coors.shape[1]
    adj = torch.rand(n, n) > 0.5
    adj = adj | adj.T
    net = E3GNN(depth=2, dim=8, m_dim=8).eval()  # no distance_cutoff, no kNN
    with torch.no_grad():
        f_out, c_out = net(feats, coors, L, adj_mat=adj)
    assert f_out.shape == feats.shape and c_out.shape == coors.shape


def test_edge_features(system):
    feats, coors, L = system
    n = coors.shape[1]
    b = coors.shape[0]
    edges = torch.randn(b, n, n, 3)
    net = E3GNN(depth=2, dim=8, m_dim=8, edge_dim=3).eval()
    with torch.no_grad():
        f_out, c_out = net(feats, coors, L, edges=edges)
    assert f_out.shape == feats.shape and c_out.shape == coors.shape


def test_mask_ignores_padding(system):
    feats, coors, L = system
    b, n, _ = coors.shape
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, -1] = False  # last node is padding
    net = E3GNN(depth=2, dim=8, m_dim=8).eval()
    with torch.no_grad():
        _, c_ref = net(feats, coors, L, mask=mask)
        perturbed = coors.clone()
        perturbed[:, -1] += 5.0  # move the padded node
        _, c_pert = net(feats, perturbed, L, mask=mask)
    # Real nodes are unaffected by where the padded node sits.
    assert torch.allclose(c_ref[:, :-1], c_pert[:, :-1], atol=1e-5)
