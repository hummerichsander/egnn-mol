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


def randomized(net: EGNN, seed: int = 0) -> EGNN:
    """Overwrite the near-identity init, which otherwise leaves every variant equal to ``x``.

    :param net: The net to overwrite in place.
    :param seed: Seed of the weight draw.
    :return: The same net, in eval mode."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return net.eval()


def test_per_call_cutoff_matches_a_net_built_at_that_radius(system):
    """A per-call radius must reproduce the net that was constructed with it, weights aside.

    The definitive statement of the feature: the override changes the neighborhood and nothing
    else, so it cannot be distinguished from having built the net that way."""
    h_node, x, box = system
    common = dict(depth=2, dim=8, m_dim=8, envelope=True)
    built_at_2 = randomized(EGNN(**common, distance_cutoff=2.0))
    built_at_5 = EGNN(**common, distance_cutoff=5.0).eval()
    built_at_5.load_state_dict(built_at_2.state_dict())

    with torch.no_grad():
        overridden = built_at_2(h_node, x, box=box, distance_cutoff=5.0)
        constructed = built_at_5(h_node, x, box=box)
        unchanged = built_at_2(h_node, x, box=box)

    assert torch.allclose(overridden[0], constructed[0])
    assert torch.allclose(overridden[1], constructed[1])
    # the two radii must actually disagree, or the test above passes vacuously.
    assert not torch.allclose(overridden[1], unchanged[1])


def test_per_call_cutoff_builds_a_graph_where_there_was_none(system):
    """A net with no dynamic graph at all falls back to all-pairs, so the override must be read
    by the guard that decides whether to build a neighborhood -- not only by the builder."""
    h_node, x, box = system
    all_pairs = randomized(EGNN(depth=2, dim=8, m_dim=8))
    radius = EGNN(depth=2, dim=8, m_dim=8, distance_cutoff=2.0).eval()
    radius.load_state_dict(all_pairs.state_dict())

    with torch.no_grad():
        overridden = all_pairs(h_node, x, box=box, distance_cutoff=2.0)[1]
        assert torch.allclose(overridden, radius(h_node, x, box=box)[1])
        assert not torch.allclose(overridden, all_pairs(h_node, x, box=box)[1])


def test_per_call_cutoff_moves_the_envelope_with_it(system):
    """The taper has to reach zero where the graph ends, or an edge re-enters with finite weight.

    Read off ``edge_envelope`` directly, which is weight-independent and so states the claim
    without the network in the way."""
    _, x, box = system
    common = dict(depth=2, dim=8, m_dim=8, envelope=True)
    net = EGNN(**common, distance_cutoff=4.0).eval()
    reference = EGNN(**common, distance_cutoff=5.0).eval()

    tapered_at_5 = net.edge_envelope(x, box, None, None, 5.0)
    assert torch.allclose(tapered_at_5, reference.edge_envelope(x, box, None, None))
    assert not torch.allclose(tapered_at_5, net.edge_envelope(x, box, None, None))


def test_a_per_call_zero_radius_still_needs_a_radius_to_taper_at(system):
    """The envelope is meaningless at a zero radius, exactly as it is at construction."""
    h_node, x, box = system
    net = EGNN(depth=2, dim=8, m_dim=8, distance_cutoff=4.0, envelope=True).eval()
    with pytest.raises(ValueError, match="distance_cutoff"):
        net(h_node, x, box=box, distance_cutoff=0.0)
