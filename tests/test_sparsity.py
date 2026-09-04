import math

import pytest
import torch
from torch import Tensor, nn

from egnn_mol import (
    GeometricEGNN,
    build_edges,
    composed_closure,
    greedy_colouring,
    hop_closure,
)

CUTOFF = 1.5


def make_net(seed: int = 0, **kwargs) -> GeometricEGNN:
    """A double-precision backbone with non-trivial weights (the near-identity init hides everything).

    :param seed: Seed of the weight draw.
    :param kwargs: Overrides of the constructor defaults.
    :return: The backbone, in eval mode."""

    defaults = dict(
        depth=2,
        dim=8,
        encoding="bessel",
        encoding_features=6,
        cutoff=CUTOFF,
        m_dim=16,
        distance_cutoff=CUTOFF,
        tripp_num_layers=0,
    )
    defaults.update(kwargs)
    net = GeometricEGNN(**defaults).double().eval()

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.1 * torch.randn(p.shape, generator=g, dtype=torch.float64))

    return net


@pytest.fixture
def chain() -> tuple[Tensor, Tensor]:
    """Ten nodes on a unit-pitch helix, so the radius graph is exactly a path.

    A path has diameter ``n - 1``, which is what makes the receptive field measurable: the
    ``hops``-hop closure is a band of known width, where a compact cloud would be one hop wide.
    The helix rather than a straight line because a collinear chain is degenerate for the SE(3)
    term -- every displacement is parallel, so ``chi`` vanishes identically and the chirality
    term silently stops widening the receptive field.

    :return: Features h_node (10, 8) and positions x (10, 3), in double precision."""

    torch.manual_seed(3)
    n = 10
    turn = [
        [i * 1.0, 0.3 * math.cos(i * 2.0), 0.3 * math.sin(i * 2.0)] for i in range(n)
    ]

    return torch.randn(n, 8, dtype=torch.float64), torch.tensor(
        turn, dtype=torch.float64
    )


def autograd_trace(v: Tensor, x: Tensor) -> Tensor:
    """Exact divergence by one backward pass per coordinate.

    :param v: Velocity (N, 3), differentiable w.r.t. ``x``.
    :param x: Positions (N, 3).
    :return: The scalar trace of dv/dx."""

    return sum(
        torch.autograd.grad(v.flatten()[i], x, retain_graph=True)[0].flatten()[i]
        for i in range(v.numel())
    )


def block_support(v: Tensor, x: Tensor) -> Tensor:
    """Which node blocks of the Jacobian of ``v`` w.r.t. ``x`` are not structurally zero.

    :param v: Velocity (N, 3), differentiable w.r.t. ``x``.
    :param x: Positions (N, 3).
    :return: Boolean (N, N); entry (i, j) is True where dv_i/dx_j has any nonzero entry."""

    rows = [
        torch.autograd.grad(v.flatten()[i], x, retain_graph=True)[0]
        for i in range(v.numel())
    ]
    jacobian = torch.stack(rows).reshape(x.shape[0], 3, x.shape[0], 3)

    return jacobian.abs().amax(dim=(1, 3)) > 0


def dense_pattern(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Boolean (N, N) view of a sparse pattern.

    :param edge_index: Edge connectivity (2, E).
    :param num_nodes: Number of nodes N.
    :return: The dense boolean adjacency."""

    out = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    out[edge_index[0], edge_index[1]] = True

    return out


class TestHopClosure:
    """The closure of a path graph, where the answer is a band matrix of known width."""

    @pytest.mark.parametrize("hops", [1, 2, 3])
    def test_is_the_band_of_width_hops(self, hops):
        """On a path, node i reaches node j within `hops` iff ``|i - j| <= hops``.

        :param hops: Number of hops under test."""
        n = 8
        idx = torch.arange(n - 1)
        edge_index = torch.stack([idx, idx + 1])

        closure = dense_pattern(hop_closure(edge_index, n, hops), n)
        expected = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs() <= hops

        assert torch.equal(closure, expected)

    def test_carries_self_loops_and_is_symmetric(self):
        """Reachability is symmetric and reflexive whatever direction the edges were given in."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        closure = dense_pattern(hop_closure(edge_index, 5, 2), 5)

        assert torch.equal(closure, closure.T)
        assert closure.diagonal().all()

    def test_rejects_zero_hops(self):
        """Zero hops is not a pattern; the caller has confused it with an empty graph."""
        with pytest.raises(ValueError, match="at least one"):
            hop_closure(torch.tensor([[0], [1]]), 2, 0)


class TestComposedClosure:
    """Reachability through a sequence of different graphs, one per hop."""

    def test_walks_one_hop_of_each_graph_in_turn(self):
        """Composition, not union: (0, 2) needs one hop of each graph, and (0, 3) is out of reach."""
        n = 4
        first = torch.tensor([[0], [1]])
        second = torch.tensor([[1], [2]])

        closure = dense_pattern(composed_closure([first, second], n), n)

        assert closure[0, 2] and closure[2, 0]
        assert not closure[0, 3]

    def test_does_not_depend_on_the_order(self):
        """The closure is symmetrized, which is what makes the two orders agree."""
        n = 5
        path = torch.tensor([[0, 1, 2], [1, 2, 3]])
        single = torch.tensor([[3], [4]])

        forwards = dense_pattern(composed_closure([path, single], n), n)
        backwards = dense_pattern(composed_closure([single, path], n), n)

        assert torch.equal(forwards, backwards)

    def test_is_symmetric(self):
        """A colour class must be independent whichever way the dependence runs."""
        n = 5
        first = torch.tensor([[0], [1]])
        second = torch.tensor([[1, 2], [2, 3]])

        closure = dense_pattern(composed_closure([first, second], n), n)

        assert torch.equal(closure, closure.T)

    def test_repeating_one_graph_is_its_hop_closure(self):
        """`hop_closure` is the homogeneous special case, and must stay exactly that."""
        n = 8
        idx = torch.arange(n - 1)
        edge_index = torch.stack([idx, idx + 1])

        composed = dense_pattern(composed_closure([edge_index] * 3, n), n)
        homogeneous = dense_pattern(hop_closure(edge_index, n, 3), n)

        assert torch.equal(composed, homogeneous)

    def test_rejects_an_empty_sequence(self):
        """No hops is not a pattern; the caller has confused it with an empty graph."""
        with pytest.raises(ValueError, match="at least one"):
            composed_closure([], 3)


class TestGreedyColouring:
    """The property the compressed Jacobian rests on: colour classes are independent sets."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_classes_are_independent_sets(self, seed):
        """No edge of the pattern may join two nodes of one colour.

        :param seed: Seed of the random graph."""
        torch.manual_seed(seed)
        n = 30
        adj = torch.rand(n, n) > 0.75
        adj = (adj | adj.T) & ~torch.eye(n, dtype=torch.bool)
        edge_index = adj.nonzero().t()

        colours = greedy_colouring(edge_index, n)

        assert not (adj & (colours[:, None] == colours[None, :])).any()

    def test_a_complete_graph_needs_a_colour_per_node(self):
        """The one case where colouring cannot help, and the compressed trace costs the dense one."""
        n = 6
        adj = ~torch.eye(n, dtype=torch.bool)

        colours = greedy_colouring(adj.nonzero().t(), n)

        assert int(colours.max()) + 1 == n


class TestReceptiveField:
    """The claimed hop count against the Jacobian the backbone actually has."""

    @pytest.mark.parametrize(
        "tripp,depth,hops", [(0, 1, 1), (0, 3, 3), (2, 1, 2), (2, 2, 4)]
    )
    def test_hops_double_with_the_chirality_term(self, tripp, depth, hops):
        """`chi` is itself a one-hop aggregate, so it pushes the position update a hop further.

        :param tripp: Depth of the triple-product MLP; 0 disables the SE(3) term.
        :param depth: Number of message-passing layers.
        :param hops: Expected receptive radius."""
        assert make_net(depth=depth, tripp_num_layers=tripp).receptive_hops == hops

    @pytest.mark.parametrize("tripp", [0, 2])
    def test_the_jacobian_lives_inside_the_pattern(self, chain, tripp):
        """Nothing outside the closure may be nonzero, or a colouring of it would be invalid.

        :param chain: Path-graph fixture.
        :param tripp: Depth of the triple-product MLP."""
        h_node, x = chain
        net = make_net(depth=2, tripp_num_layers=tripp)

        x = x.clone().requires_grad_(True)
        support = block_support(net(h_node, x)[1] - x, x)
        pattern = dense_pattern(net.sparsity_pattern(x.detach()), x.shape[0])

        assert not (support & ~pattern).any()

    @pytest.mark.parametrize("tripp", [0, 2])
    def test_the_pattern_is_tight(self, chain, tripp):
        """One hop fewer must *not* contain it, or `receptive_hops` is overcounting and the
        compressed trace is paying for passes it does not need.

        :param chain: Path-graph fixture.
        :param tripp: Depth of the triple-product MLP."""
        h_node, x = chain
        net = make_net(depth=2, tripp_num_layers=tripp)

        x = x.clone().requires_grad_(True)
        support = block_support(net(h_node, x)[1] - x, x)

        edges, _, _ = net.build_neighborhood(x.detach())
        narrower = dense_pattern(
            hop_closure(edges, x.shape[0], net.receptive_hops - 1), x.shape[0]
        )

        assert (support & ~narrower).any()


class TestMlpDepth:
    """Depth inside a layer's own MLPs, which must buy capacity without widening anything."""

    @pytest.mark.parametrize("mlp_depth", [1, 2, 3])
    def test_deepens_every_layer_mlp(self, mlp_depth):
        """A hidden block is one hidden layer, so a depth-d MLP holds d + 1 linears.

        :param mlp_depth: Number of hidden blocks in the layer MLPs."""
        core = make_net(mlp_depth=mlp_depth).layers[0].core

        for mlp in (core.edge_mlp, core.node_mlp, core.x_mlp):
            assert sum(isinstance(m, nn.Linear) for m in mlp.modules()) == mlp_depth + 1

    def test_leaves_the_triple_product_depth_alone(self):
        """``tripp_num_layers`` doubles as the SE(3) switch, so it keeps a depth of its own."""
        core = make_net(mlp_depth=3, tripp_num_layers=1).layers[0].core

        assert sum(isinstance(m, nn.Linear) for m in core.triple_mlp.modules()) == 2

    @pytest.mark.parametrize("tripp", [0, 2])
    def test_leaves_the_colouring_unchanged(self, chain, tripp):
        """The whole point of the knob: no extra hop, hence no extra colour and no extra pass.

        :param chain: Path-graph fixture.
        :param tripp: Depth of the triple-product MLP."""
        _, x = chain
        shallow = make_net(mlp_depth=1, tripp_num_layers=tripp)
        deep = make_net(mlp_depth=3, tripp_num_layers=tripp)

        assert shallow.receptive_hops == deep.receptive_hops
        assert int(shallow.jacobian_colouring(x).max()) == int(
            deep.jacobian_colouring(x).max()
        )


class TestColouredDivergence:
    """The compressed divergence against a coordinate-at-a-time trace of the same field."""

    @pytest.mark.parametrize("envelope", [False, True])
    @pytest.mark.parametrize("mlp_depth", [1, 2])
    @pytest.mark.parametrize("tripp", [0, 2])
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_matches_the_dense_trace(self, chain, depth, tripp, mlp_depth, envelope):
        """Exact, not estimated: this must agree to machine precision, with no variance.

        :param chain: Path-graph fixture.
        :param depth: Number of message-passing layers.
        :param tripp: Depth of the triple-product MLP.
        :param mlp_depth: Number of hidden blocks in the layer MLPs.
        :param envelope: Whether the cutoff envelope is on."""
        h_node, x = chain
        net = make_net(
            depth=depth,
            tripp_num_layers=tripp,
            mlp_depth=mlp_depth,
            envelope=envelope,
        )

        x_grad = x.clone().requires_grad_(True)
        reference = autograd_trace(net(h_node, x_grad)[1] - x_grad, x_grad)

        _, _, div = net.forward_and_divergence(h_node, x.clone())

        assert torch.allclose(div.squeeze(), reference, rtol=1e-9, atol=1e-9)

    def test_splits_over_a_packed_batch(self, chain):
        """Two graphs in one packed batch each get their own divergence."""
        h_node, x = chain
        n = x.shape[0]
        net = make_net(depth=2)

        batch = torch.cat(
            [torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)]
        )
        packed_h = torch.cat([h_node, h_node])
        packed_x = torch.cat([x, x + 20.0])

        _, _, div = net.forward_and_divergence(packed_h, packed_x, batch=batch)
        _, _, single = net.forward_and_divergence(h_node, x.clone())

        assert div.shape == (2,)
        assert torch.allclose(div, single.expand(2), rtol=1e-9, atol=1e-9)

    def test_costs_fewer_passes_than_the_dense_trace(self, chain):
        """The point of the exercise: colours, not coordinates, set the number of passes."""
        _, x = chain
        net = make_net(depth=2)

        colours = int(net.jacobian_colouring(x).max()) + 1

        assert colours < x.shape[0]


class TestEnvelope:
    """The condition that makes a divergence over the current edge set a divergence at all."""

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_the_field_is_smooth_where_an_edge_appears(self, depth):
        """Halving the step must halve the difference; a jump would not shrink at all.

        This is the whole reason the envelope is evaluated once on the input positions: taper
        each layer by its own distances instead and depth 2 onwards jumps, because an edge that
        entered with zero weight has moved by the time the second layer reads it.

        :param depth: Number of message-passing layers."""
        torch.manual_seed(0)
        h_node = torch.randn(3, 8, dtype=torch.float64)
        net = make_net(depth=depth, envelope=True)

        def displacement(separation: float) -> Tensor:
            x = torch.tensor(
                [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [separation, 0.0, 0.0]],
                dtype=torch.float64,
            )
            return net(h_node, x)[1] - x

        jumps = [
            float(
                (displacement(CUTOFF + eps) - displacement(CUTOFF - eps))
                .abs()
                .max()
                .detach()
            )
            for eps in (1e-4, 1e-5)
        ]

        assert jumps[0] / jumps[1] == pytest.approx(10.0, rel=0.05)

    def test_needs_a_radius_to_taper_at(self):
        """Without a distance cutoff there is no radius at which edges appear, so none to taper."""
        with pytest.raises(ValueError, match="distance_cutoff"):
            GeometricEGNN(depth=2, dim=8, encoding_features=6, envelope=True)

    def test_a_long_static_edge_still_contributes(self):
        """A static edge beyond the cutoff must survive the envelope, or a topological graph is lost.

        The envelope exists to keep the field smooth where a *dynamic* edge appears. A static edge
        set does not depend on the positions, so nothing appears or disappears and there is nothing
        to taper -- tapering it would delete every bond longer than ``distance_cutoff``.

        :return: None."""
        h_node = torch.randn(3, 8, dtype=torch.float64)
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [2.5 * CUTOFF, 0.0, 0.0]],
            dtype=torch.float64,
        )
        long_edge = torch.tensor([[0, 2], [2, 0]])
        net = make_net(depth=1, envelope=True)

        without = net(h_node, x)[1]
        with_edge = net(h_node, x, long_edge)[1]

        assert not torch.allclose(without, with_edge)

    def test_a_purely_static_graph_is_unaffected_by_the_envelope(self):
        """With no pair inside the cutoff, enveloping is a no-op -- the sharpest form of the claim.

        :return: None."""
        h_node = torch.randn(3, 8, dtype=torch.float64)
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0 * CUTOFF, 0.0, 0.0], [4.0 * CUTOFF, 0.0, 0.0]],
            dtype=torch.float64,
        )
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

        enveloped = make_net(depth=2, envelope=True)(h_node, x, edge_index)[1]
        plain = make_net(depth=2, envelope=False)(h_node, x, edge_index)[1]

        assert torch.allclose(enveloped, plain)

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_the_field_is_smooth_where_an_edge_appears_beside_a_static_edge(
        self, depth
    ):
        """Exempting static edges must not reintroduce a jump where a dynamic edge enters.

        Continuity scaling, not a single jump measurement, is what catches this: halving the probe
        step must halve the difference.

        :param depth: Number of message-passing layers.
        :return: None."""
        torch.manual_seed(0)
        h_node = torch.randn(3, 8, dtype=torch.float64)
        static = torch.tensor([[0, 1], [1, 0]])
        net = make_net(depth=depth, envelope=True)

        def displacement(separation: float) -> Tensor:
            x = torch.tensor(
                [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [separation, 0.0, 0.0]],
                dtype=torch.float64,
            )
            return net(h_node, x, static)[1] - x

        jumps = [
            float(
                (displacement(CUTOFF + eps) - displacement(CUTOFF - eps))
                .abs()
                .max()
                .detach()
            )
            for eps in (1e-4, 1e-5)
        ]

        assert jumps[0] / jumps[1] == pytest.approx(10.0, rel=0.05)

    def test_the_static_mask_survives_coalesce(self):
        """``coalesce`` re-sorts the edge index, so the mask must be built through it, not before.

        A pair that is both static and inside the cutoff coalesces to one edge that is still
        static, and its features are the static row plus the dynamic zeros.

        :return: None."""
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5 * CUTOFF, 0.0, 0.0]], dtype=torch.float64
        )
        edge_index = torch.tensor([[0, 1], [1, 0]])
        h_edge = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)

        edges, attrs, static = build_edges(
            x, edge_index, h_edge, None, None, edge_dim=2, distance_cutoff=CUTOFF
        )

        assert edges.shape[1] == 2
        assert static.all()
        order = {(int(s), int(d)): k for k, (s, d) in enumerate(edges.t().tolist())}
        assert torch.allclose(attrs[order[(0, 1)]], h_edge[0])
        assert torch.allclose(attrs[order[(1, 0)]], h_edge[1])

    def test_a_dynamic_edge_is_not_marked_static(self):
        """The mask must distinguish the two sources, or the exemption would cover everything.

        :return: None."""
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5 * CUTOFF, 0.0, 0.0]], dtype=torch.float64
        )

        _, _, static = build_edges(x, None, None, None, None, distance_cutoff=CUTOFF)

        assert not static.any()


class TestLayerSchedule:
    """`dynamic_layers`: only some layers see the radius graph, so depth stops setting the ball."""

    @pytest.fixture
    def bonds(self) -> Tensor:
        """Two isolated static bonds, far sparser than the radius graph on the same chain.

        Sparser on purpose: a static graph that matched the radius graph would make a scheduled
        stack indistinguishable from an unscheduled one.

        :return: Edge index (2, 4)."""

        return torch.tensor([[0, 1, 5, 6], [1, 0, 6, 5]])

    @pytest.mark.parametrize("schedule", [(0,), (2,)])
    @pytest.mark.parametrize("tripp", [0, 2])
    def test_matches_the_dense_trace(self, chain, bonds, tripp, schedule):
        """The composed pattern must still contain the whole Jacobian, or the trace is wrong.

        :param chain: Path-graph fixture.
        :param bonds: Static edge fixture.
        :param tripp: Depth of the triple-product MLP.
        :param schedule: Which layers read the dynamic edges."""
        h_node, x = chain
        net = make_net(depth=3, tripp_num_layers=tripp, dynamic_layers=schedule)

        x_grad = x.clone().requires_grad_(True)
        reference = autograd_trace(net(h_node, x_grad, bonds)[1] - x_grad, x_grad)

        _, _, div = net.forward_and_divergence(h_node, x.clone(), bonds)

        assert torch.allclose(div.squeeze(), reference, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("tripp", [0, 2])
    def test_the_jacobian_lives_inside_the_pattern(self, chain, bonds, tripp):
        """Nothing outside the composed closure may be nonzero, or a colouring of it is invalid.

        :param chain: Path-graph fixture.
        :param bonds: Static edge fixture.
        :param tripp: Depth of the triple-product MLP."""
        h_node, x = chain
        net = make_net(depth=3, tripp_num_layers=tripp, dynamic_layers=(1,))

        x = x.clone().requires_grad_(True)
        support = block_support(net(h_node, x, bonds)[1] - x, x)
        pattern = dense_pattern(net.sparsity_pattern(x.detach(), bonds), x.shape[0])

        assert not (support & ~pattern).any()

    def test_the_dynamic_hop_is_still_needed(self, chain, bonds):
        """Tightness in the direction that matters: dropping it must lose part of the support.

        :param chain: Path-graph fixture.
        :param bonds: Static edge fixture."""
        h_node, x = chain
        net = make_net(depth=3, dynamic_layers=(1,))

        x = x.clone().requires_grad_(True)
        support = block_support(net(h_node, x, bonds)[1] - x, x)
        static_only = dense_pattern(
            hop_closure(bonds, x.shape[0], net.receptive_hops), x.shape[0]
        )

        assert (support & ~static_only).any()

    def test_costs_fewer_colours_than_the_unscheduled_stack(self, chain, bonds):
        """The whole point: the same depth, over a smaller ball.

        :param chain: Path-graph fixture.
        :param bonds: Static edge fixture."""
        _, x = chain
        scheduled = make_net(depth=3, dynamic_layers=(0,))
        every_layer = make_net(depth=3)

        assert int(scheduled.jacobian_colouring(x, bonds).max()) < int(
            every_layer.jacobian_colouring(x, bonds).max()
        )

    def test_an_empty_schedule_drops_the_radius_graph(self, chain, bonds):
        """With no layer reading them, the dynamic edges must not reach the field at all.

        The sharpest form of the claim: identical to a backbone built without a radius graph.

        :param chain: Path-graph fixture.
        :param bonds: Static edge fixture."""
        h_node, x = chain

        scheduled = make_net(depth=2, dynamic_layers=())(h_node, x, bonds)[1]
        static_only = make_net(depth=2, distance_cutoff=0.0)(h_node, x, bonds)[1]

        assert torch.allclose(scheduled, static_only, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("depth", [2, 3])
    def test_the_field_is_smooth_where_an_edge_appears(self, depth, bonds):
        """A schedule must not reintroduce the jump the envelope exists to remove.

        Continuity scaling again: halving the probe step must halve the difference. A layer
        outside the schedule sees no dynamic edges at all, so the edge entering at the cutoff
        reaches it through the enveloped layer alone.

        :param depth: Number of message-passing layers.
        :param bonds: Static edge fixture (unused pairs are ignored on three nodes)."""
        torch.manual_seed(0)
        h_node = torch.randn(3, 8, dtype=torch.float64)
        static = torch.tensor([[0, 1], [1, 0]])
        net = make_net(depth=depth, envelope=True, dynamic_layers=(0,))

        def displacement(separation: float) -> Tensor:
            x = torch.tensor(
                [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [separation, 0.0, 0.0]],
                dtype=torch.float64,
            )
            return net(h_node, x, static)[1] - x

        jumps = [
            float(
                (displacement(CUTOFF + eps) - displacement(CUTOFF - eps))
                .abs()
                .max()
                .detach()
            )
            for eps in (1e-4, 1e-5)
        ]

        assert jumps[0] / jumps[1] == pytest.approx(10.0, rel=0.05)

    def test_rejects_a_layer_that_does_not_exist(self):
        """An index past the stack is a config error, not a silently ignored one."""
        with pytest.raises(ValueError, match="dynamic_layers"):
            GeometricEGNN(depth=2, dim=8, encoding_features=6, dynamic_layers=(2,))


class TestPerCallCutoff:
    """The radius may be given per call; everything derived from the edge set must follow it."""

    @pytest.mark.parametrize("tripp", [0, 2])
    def test_the_divergence_matches_the_dense_trace(self, chain, tripp):
        """The colouring must describe the graph the layers ran on, not the constructed one.

        A colouring taken at the smaller radius would leave two same-coloured nodes within reach
        at the larger one, and the trace would be silently wrong rather than obviously broken.

        :param chain: Path-graph fixture.
        :param tripp: Depth of the triple-product MLP."""
        h_node, x = chain
        net = make_net(depth=2, tripp_num_layers=tripp, envelope=True)
        wide = 2.0 * CUTOFF

        x_grad = x.clone().requires_grad_(True)
        reference = autograd_trace(
            net(h_node, x_grad, distance_cutoff=wide)[1] - x_grad, x_grad
        )

        _, _, div = net.forward_and_divergence(
            h_node, x.clone(), distance_cutoff=wide
        )

        assert torch.allclose(div.squeeze(), reference, rtol=1e-9, atol=1e-9)
        # the wider graph must actually change the field, or a backbone that dropped the
        # override entirely would agree with itself here.
        narrow = net.forward_and_divergence(h_node, x.clone())[2]
        assert not torch.allclose(div, narrow)

    def test_the_pattern_matches_a_net_built_at_that_radius(self, chain):
        """``sparsity_pattern`` has its own ``build_edges`` call, so it needs the override too.

        :param chain: Path-graph fixture."""
        _, x = chain
        n = x.shape[0]
        wide = 2.0 * CUTOFF
        reference = make_net(distance_cutoff=wide)

        pattern = make_net().sparsity_pattern(x, distance_cutoff=wide)
        expected = reference.sparsity_pattern(x)

        assert torch.equal(pattern, expected)
        assert int(greedy_colouring(pattern, n).max()) > int(
            greedy_colouring(make_net().sparsity_pattern(x), n).max()
        )

    def test_the_field_is_smooth_where_an_edge_appears(self, chain):
        """The taper must move with the radius, or an edge re-enters with a finite weight.

        Halving the step halves the difference only if the envelope reaches zero exactly at the
        radius the graph was built at; a taper left at the constructed radius jumps instead.

        :param chain: Path-graph fixture."""
        torch.manual_seed(0)
        h_node = torch.randn(3, 8, dtype=torch.float64)
        net = make_net(depth=2, distance_cutoff=0.5 * CUTOFF, envelope=True)

        def displacement(separation: float) -> Tensor:
            x = torch.tensor(
                [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [separation, 0.0, 0.0]],
                dtype=torch.float64,
            )
            return net(h_node, x, distance_cutoff=CUTOFF)[1] - x

        jumps = [
            float(
                (displacement(CUTOFF + eps) - displacement(CUTOFF - eps))
                .abs()
                .max()
                .detach()
            )
            for eps in (1e-4, 1e-5)
        ]

        assert jumps[0] / jumps[1] == pytest.approx(10.0, rel=0.05)

    def test_still_needs_a_radius_to_taper_at(self, chain):
        """A per-call zero radius is as meaningless for the envelope as a constructed one.

        :param chain: Path-graph fixture."""
        h_node, x = chain
        net = make_net(envelope=True)

        with pytest.raises(ValueError, match="distance_cutoff"):
            net(h_node, x, distance_cutoff=0.0)
