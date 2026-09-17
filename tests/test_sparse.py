import math

import pytest
import torch

from egnn_mol import EGNN, GeometricEGNN, SparseEGNNLayer
from conftest import reflection_z, rel_err, rotation_z


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


def randomized(net, seed: int = 0, scale: float = 0.1):
    """Overwrite the near-identity init, which otherwise leaves every variant equal to ``x``.

    :param net: The module to overwrite in place.
    :param seed: Seed of the weight draw.
    :param scale: Standard deviation of the draw. A symmetry test needs a larger one than the
        default: at 0.1 the heads barely vary across edges, which leaves the chirality term
        degenerate (its three vectors come out parallel) and would certify nothing.
    :return: The same module, in eval mode."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
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


def test_vp_aggregation_divides_by_the_square_root_of_degree():
    """`vp` pools the sum, scaled by 1 / sqrt(1 + degree), so the scale is degree-independent."""
    layer = SparseEGNNLayer(dim=4, m_dim=3, aggr="vp")

    dst = torch.tensor([0, 0, 0, 1])  # node 0 gets three messages, node 1 gets one
    m_ij = torch.randn(4, 3)

    pooled = layer.pool(m_ij, dst, 2, env=None)

    total = torch.zeros(2, 3).index_add_(0, dst, m_ij)
    expected = total / torch.tensor([[math.sqrt(4.0)], [math.sqrt(2.0)]])

    assert torch.allclose(pooled, expected, atol=1e-6)


def test_vp_aggregation_counts_the_envelope_not_the_edges():
    """An edge crossing the cutoff must not jump the field.

    The envelope already takes the arriving edge's own contribution to zero, so the only thing
    that can jump is the degree it is divided by. Counting edges would step 3 -> 4 at the
    crossing and rescale every other message with it; counting sum(env^2) arrives at zero.

    Two layers, because ``aggr`` pools into the feature channel and only the next layer's
    messages carry that into a position update -- at depth 1 the velocity cannot see it.
    """
    cutoff = 1.5
    net = GeometricEGNN(
        depth=2, dim=4, m_dim=4, distance_cutoff=cutoff, envelope=True, aggr="vp"
    ).eval()

    torch.manual_seed(0)
    h_node = torch.randn(5, 4)
    cluster = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    )

    def run(distance: float) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([cluster, torch.tensor([[distance, 0.0, 0.0]])], dim=0)
        with torch.no_grad():
            h_out, x_out = net(h_node, x)
        return h_out, x_out - x

    eps = 1e-6
    h_inside, v_inside = run(cutoff - eps)
    h_outside, v_outside = run(cutoff + eps)

    assert torch.allclose(h_inside, h_outside, atol=1e-4)
    assert torch.allclose(v_inside, v_outside, atol=1e-4)


def test_vector_channels_stay_e3_equivariant(compact_system):
    """The vector channel contracts only through dot products, so parity survives.

    Rotations, reflections and translations are all symmetries. A cross or triple product in the
    contraction would produce a pseudoscalar and break the reflection case -- that is what
    `tripp_num_layers` does, and `test_vector_channels_do_not_hide_a_broken_parity` checks this
    setup is strong enough to see it.
    """
    h_node, x, _ = compact_system
    h_node, x = h_node[0], x[0]
    n = x.shape[0]
    edge_index = full_edge_index(torch.arange(n), include_self=False)
    net = randomized(
        GeometricEGNN(depth=2, dim=8, m_dim=8, norm_displacement=True, vector_channels=4),
        scale=0.5,
    )

    def run(positions):
        with torch.no_grad():
            h_out, x_out = net(h_node, positions, edge_index=edge_index)
        return h_out, x_out - positions

    centroid = x.mean(0, keepdim=True)
    h_ref, v_ref = run(x)

    for M in (rotation_z(math.pi / 5), reflection_z()):
        h_m, v_m = run((x - centroid) @ M.T + centroid)
        assert rel_err(h_m, h_ref) < 1e-5
        assert rel_err(v_m, v_ref @ M.T) < 1e-5

    delta = torch.tensor([2.0, -1.0, 0.5])
    h_tr, v_tr = run(x + delta)
    assert rel_err(h_tr, h_ref) < 1e-5
    assert rel_err(v_tr, v_ref) < 1e-5


def test_vector_channels_do_not_hide_a_broken_parity(compact_system):
    """The reflection check above has to be able to fail, or it certifies nothing.

    Same geometry and weight scale, with the chirality term switched on instead: its pseudoscalar
    is parity-odd, so the reflected field must not match.
    """
    h_node, x, _ = compact_system
    h_node, x = h_node[0], x[0]
    edge_index = full_edge_index(torch.arange(x.shape[0]), include_self=False)
    net = randomized(
        GeometricEGNN(depth=2, dim=8, m_dim=8, norm_displacement=True, tripp_num_layers=2),
        scale=0.5,
    )

    centroid = x.mean(0, keepdim=True)
    M = reflection_z()
    x_ref = (x - centroid) @ M.T + centroid
    with torch.no_grad():
        _, x_out = net(h_node, x, edge_index=edge_index)
        _, x_ref_out = net(h_node, x_ref, edge_index=edge_index)

    assert rel_err(x_ref_out - x_ref, (x_out - x) @ M.T) > 1e-2


def test_vector_channels_change_the_field(compact_system):
    """Turning the channel on has to do something, or the equivariance test proves nothing."""
    h_node, x, _ = compact_system
    h_node, x = h_node[0], x[0]
    edge_index = full_edge_index(torch.arange(x.shape[0]), include_self=False)

    common = dict(depth=2, dim=8, m_dim=8)
    with torch.no_grad():
        _, plain = randomized(GeometricEGNN(**common))(h_node, x, edge_index=edge_index)
        _, vector = randomized(GeometricEGNN(**common, vector_channels=4))(
            h_node, x, edge_index=edge_index
        )

    assert not torch.allclose(plain - x, vector - x, atol=1e-5)


def test_vector_chirality_is_parity_odd_but_costs_no_hop(compact_system):
    """The pseudoscalar off the vector channels breaks reflection, and only reflection.

    That is the point of it: chirality awareness with the receptive field of the plain E(3)
    stack, because it contracts vectors the layer was handed rather than ones it aggregates.
    Rotations stay symmetries; `receptive_hops` stays at `depth`, where `tripp_num_layers`
    doubles it.
    """
    h_node, x, _ = compact_system
    h_node, x = h_node[0], x[0]
    edge_index = full_edge_index(torch.arange(x.shape[0]), include_self=False)
    net = randomized(
        GeometricEGNN(
            depth=2,
            dim=8,
            m_dim=8,
            norm_displacement=True,
            vector_channels=4,
            vector_chirality=True,
        ),
        scale=0.5,
    )

    def run(positions):
        with torch.no_grad():
            _, x_out = net(h_node, positions, edge_index=edge_index)
        return x_out - positions

    centroid = x.mean(0, keepdim=True)
    v_ref = run(x)

    R = rotation_z(math.pi / 5)
    assert rel_err(run((x - centroid) @ R.T + centroid), v_ref @ R.T) < 1e-5

    # the break measures 8.5e-3 to 2.7 over the first five seeds, against a rotation residual of
    # ~1e-6, so the floor sits an order under the weakest draw rather than at the dense SE(3)
    # test's 1e-2, which seed 0 would slip past.
    M = reflection_z()
    assert rel_err(run((x - centroid) @ M.T + centroid), v_ref @ M.T) > 1e-3

    assert net.receptive_hops == 2  # depth, not 2 * depth


def test_vector_chirality_needs_vector_channels():
    """Contracting channels that do not exist is a configuration error, not a silent no-op."""
    with pytest.raises(ValueError, match="vector_channels"):
        GeometricEGNN(depth=1, dim=8, m_dim=8, vector_chirality=True)


def test_vector_norm_bounds_the_channels_without_flattening_the_pseudoscalar(compact_system):
    """`norm_vec` has to bound the channels without costing the chirality its signal.

    The signed volume is cubic in the vectors, so unnormalized it grows faster than they do and
    ends up dominating the field -- a reflection residual four orders over the rotation one is a
    symptom of that, not of healthy chirality. Normalized it is a proportionate term: still
    unambiguously above the rotation floor, no longer running away. The floor is a ratio rather
    than an absolute, because normalizing changes the scale of the whole field.
    """
    h_node, x, _ = compact_system
    h_node, x = h_node[0], x[0]
    edge_index = full_edge_index(torch.arange(x.shape[0]), include_self=False)
    net = randomized(
        GeometricEGNN(
            depth=3,
            dim=8,
            m_dim=8,
            norm_displacement=True,
            vector_channels=8,
            vector_chirality=True,
            norm_vec=True,
        ),
        scale=0.3,
    )

    def run(positions):
        with torch.no_grad():
            _, x_out = net(h_node, positions, edge_index=edge_index)
        return x_out - positions

    centroid = x.mean(0, keepdim=True)
    v_ref = run(x)

    R, M = rotation_z(math.pi / 5), reflection_z()
    rotation = rel_err(run((x - centroid) @ R.T + centroid), v_ref @ R.T)
    reflection = rel_err(run((x - centroid) @ M.T + centroid), v_ref @ M.T)

    assert torch.isfinite(v_ref).all()
    assert reflection > 10 * rotation
