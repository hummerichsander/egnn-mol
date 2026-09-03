import pytest
import torch

from egnn_mol import EGNN


def test_forward_shapes_open_boundary(system):
    h_node, x, _ = system
    net = EGNN(depth=2, dim=8, m_dim=8).eval()
    with torch.no_grad():
        h_node_out, x_out = net(h_node, x)
    assert h_node_out.shape == h_node.shape
    assert x_out.shape == x.shape


def test_return_pos_changes(system):
    h_node, x, box = system
    net = EGNN(depth=3, dim=8, m_dim=8).eval()
    with torch.no_grad():
        _, _, changes = net(h_node, x, box=box, return_x_changes=True)
    assert len(changes) == 4  # input + one per layer
    assert torch.allclose(changes[0], x)


def test_radius_graph_runs(system):
    h_node, x, box = system
    net = EGNN(depth=2, dim=8, m_dim=8, distance_cutoff=4.0).eval()
    with torch.no_grad():
        h_node_out, x_out = net(h_node, x, box=box)
    assert torch.isfinite(h_node_out).all() and torch.isfinite(x_out).all()


def test_knn_graph_runs(system):
    h_node, x, box = system
    net = EGNN(depth=2, dim=8, m_dim=8, num_nearest_neighbors=3).eval()
    with torch.no_grad():
        h_node_out, x_out = net(h_node, x, box=box)
    assert h_node_out.shape == h_node.shape and x_out.shape == x.shape


def test_static_bonds_only(system):
    h_node, x, box = system
    n = x.shape[1]
    adj = torch.rand(n, n) > 0.5
    adj = adj | adj.T
    net = EGNN(depth=2, dim=8, m_dim=8).eval()  # no distance_cutoff, no kNN
    with torch.no_grad():
        h_node_out, x_out = net(h_node, x, adj_mat=adj, box=box)
    assert h_node_out.shape == h_node.shape and x_out.shape == x.shape


def test_edge_features(system):
    h_node, x, box = system
    b, n, _ = x.shape
    h_edge = torch.randn(b, n, n, 3)
    net = EGNN(depth=2, dim=8, m_dim=8, edge_dim=3).eval()
    with torch.no_grad():
        h_node_out, x_out = net(h_node, x, h_edge=h_edge, box=box)
    assert h_node_out.shape == h_node.shape and x_out.shape == x.shape


def test_mask_ignores_padding(system):
    h_node, x, box = system
    b, n, _ = x.shape
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, -1] = False  # last node is padding
    net = EGNN(depth=2, dim=8, m_dim=8).eval()
    with torch.no_grad():
        _, x_ref = net(h_node, x, mask=mask, box=box)
        perturbed = x.clone()
        perturbed[:, -1] += 5.0  # move the padded node
        _, x_pert = net(h_node, perturbed, mask=mask, box=box)
    # Real nodes are unaffected by where the padded node sits.
    assert torch.allclose(x_ref[:, :-1], x_pert[:, :-1], atol=1e-5)


def test_empty_schedule_drops_the_radius_graph(system):
    h_node, x, box = system
    n = x.shape[1]
    ring = torch.eye(n, dtype=torch.bool).roll(1, 0)
    adj = ring | ring.T

    torch.manual_seed(0)
    scheduled = EGNN(
        depth=2, dim=8, m_dim=8, distance_cutoff=4.0, dynamic_layers=()
    ).eval()
    torch.manual_seed(0)
    static_only = EGNN(depth=2, dim=8, m_dim=8).eval()  # no distance_cutoff at all

    with torch.no_grad():
        _, with_schedule = scheduled(h_node, x, adj_mat=adj, box=box)
        _, without = static_only(h_node, x, adj_mat=adj, box=box)

    assert torch.allclose(with_schedule, without, atol=1e-6)


def test_schedule_rejects_a_layer_that_does_not_exist():
    with pytest.raises(ValueError, match="dynamic_layers"):
        EGNN(depth=2, dim=8, m_dim=8, dynamic_layers=(2,))
