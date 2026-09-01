import math

import pytest
import torch
from torch import Tensor

from egnn_mol import GeometricEGNN, greedy_colouring, hop_closure

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

    return torch.randn(n, 8, dtype=torch.float64), torch.tensor(turn, dtype=torch.float64)


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

    @pytest.mark.parametrize("tripp,depth,hops", [(0, 1, 1), (0, 3, 3), (2, 1, 2), (2, 2, 4)])
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

        edges, _ = net.build_neighborhood(x.detach())
        narrower = dense_pattern(
            hop_closure(edges, x.shape[0], net.receptive_hops - 1), x.shape[0]
        )

        assert (support & ~narrower).any()


class TestColouredDivergence:
    """The compressed divergence against a coordinate-at-a-time trace of the same field."""

    @pytest.mark.parametrize("envelope", [False, True])
    @pytest.mark.parametrize("tripp", [0, 2])
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_matches_the_dense_trace(self, chain, depth, tripp, envelope):
        """Exact, not estimated: this must agree to machine precision, with no variance.

        :param chain: Path-graph fixture.
        :param depth: Number of message-passing layers.
        :param tripp: Depth of the triple-product MLP.
        :param envelope: Whether the cutoff envelope is on."""
        h_node, x = chain
        net = make_net(depth=depth, tripp_num_layers=tripp, envelope=envelope)

        x_grad = x.clone().requires_grad_(True)
        reference = autograd_trace(net(h_node, x_grad)[1] - x_grad, x_grad)

        _, _, div = net.forward_and_divergence(h_node, x.clone())

        assert torch.allclose(div.squeeze(), reference, rtol=1e-9, atol=1e-9)

    def test_splits_over_a_packed_batch(self, chain):
        """Two graphs in one packed batch each get their own divergence."""
        h_node, x = chain
        n = x.shape[0]
        net = make_net(depth=2)

        batch = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
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
            float((displacement(CUTOFF + eps) - displacement(CUTOFF - eps)).abs().max().detach())
            for eps in (1e-4, 1e-5)
        ]

        assert jumps[0] / jumps[1] == pytest.approx(10.0, rel=0.05)

    def test_needs_a_radius_to_taper_at(self):
        """Without a distance cutoff there is no radius at which edges appear, so none to taper."""
        with pytest.raises(ValueError, match="distance_cutoff"):
            GeometricEGNN(depth=2, dim=8, encoding_features=6, envelope=True)
