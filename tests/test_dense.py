import torch

from egnn_mol import E3GNN


def test_forward_shapes_open_boundary(system):
    x, pos, _ = system
    net = E3GNN(depth=2, dim=8, m_dim=8).eval()
    with torch.no_grad():
        x_out, pos_out = net(x, pos)
    assert x_out.shape == x.shape
    assert pos_out.shape == pos.shape


def test_return_pos_changes(system):
    x, pos, box = system
    net = E3GNN(depth=3, dim=8, m_dim=8).eval()
    with torch.no_grad():
        _, _, changes = net(x, pos, box=box, return_pos_changes=True)
    assert len(changes) == 4  # input + one per layer
    assert torch.allclose(changes[0], pos)


def test_radius_graph_runs(system):
    x, pos, box = system
    net = E3GNN(depth=2, dim=8, m_dim=8, distance_cutoff=4.0).eval()
    with torch.no_grad():
        x_out, pos_out = net(x, pos, box=box)
    assert torch.isfinite(x_out).all() and torch.isfinite(pos_out).all()


def test_knn_graph_runs(system):
    x, pos, box = system
    net = E3GNN(depth=2, dim=8, m_dim=8, num_nearest_neighbors=3).eval()
    with torch.no_grad():
        x_out, pos_out = net(x, pos, box=box)
    assert x_out.shape == x.shape and pos_out.shape == pos.shape


def test_static_bonds_only(system):
    x, pos, box = system
    n = pos.shape[1]
    adj = torch.rand(n, n) > 0.5
    adj = adj | adj.T
    net = E3GNN(depth=2, dim=8, m_dim=8).eval()  # no distance_cutoff, no kNN
    with torch.no_grad():
        x_out, pos_out = net(x, pos, adj_mat=adj, box=box)
    assert x_out.shape == x.shape and pos_out.shape == pos.shape


def test_edge_features(system):
    x, pos, box = system
    b, n, _ = pos.shape
    edge_attr = torch.randn(b, n, n, 3)
    net = E3GNN(depth=2, dim=8, m_dim=8, edge_dim=3).eval()
    with torch.no_grad():
        x_out, pos_out = net(x, pos, edge_attr=edge_attr, box=box)
    assert x_out.shape == x.shape and pos_out.shape == pos.shape


def test_mask_ignores_padding(system):
    x, pos, box = system
    b, n, _ = pos.shape
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, -1] = False  # last node is padding
    net = E3GNN(depth=2, dim=8, m_dim=8).eval()
    with torch.no_grad():
        _, pos_ref = net(x, pos, mask=mask, box=box)
        perturbed = pos.clone()
        perturbed[:, -1] += 5.0  # move the padded node
        _, pos_pert = net(x, perturbed, mask=mask, box=box)
    # Real nodes are unaffected by where the padded node sits.
    assert torch.allclose(pos_ref[:, :-1], pos_pert[:, :-1], atol=1e-5)
