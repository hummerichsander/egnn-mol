import math

import pytest
import torch
from torch import Tensor

from egnn_mol import RadialField

ENCODINGS = ("bessel", "fourier", "gaussian")


def rotation_z(theta: float) -> Tensor:
    """Rotation matrix about the z-axis by ``theta`` radians."""
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def reflection_z() -> Tensor:
    """Improper (det = -1) reflection through the xy-plane."""
    return torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))


def full_edge_index(n: int) -> Tensor:
    """All ordered pairs among ``n`` nodes as edge_index [source/neighbor, target/center]."""
    idx = torch.arange(n)
    src, dst = idx.repeat(n), idx.repeat_interleave(n)
    return torch.stack([src, dst])[:, src != dst]


def make_field(seed: int = 0, **kwargs) -> RadialField:
    """A double-precision field with non-trivial weights (the zero-init hides everything).

    :param seed: Seed of the weight draw.
    :param kwargs: Overrides of the constructor defaults.
    :return: The field, in eval mode."""

    defaults = dict(
        dim=8, encoding="gaussian", encoding_features=6, cutoff=2.0, m_dim=16, head_depth=2,
    )
    defaults.update(kwargs)
    net = RadialField(**defaults).double().eval()

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))

    return net


@pytest.fixture
def system() -> tuple[Tensor, Tensor]:
    """Seven nodes in a 2 nm cube, in double precision.

    :return: Features h_node (N, 8) and positions x (N, 3)."""
    torch.manual_seed(3)
    n = 7
    x = torch.rand(n, 3, dtype=torch.float64) * 2.0
    return torch.randn(n, 8, dtype=torch.float64), x


def autograd_trace(v: Tensor, x: Tensor) -> Tensor:
    """Exact divergence by one backward pass per coordinate.

    :param v: Velocity (N, 3), differentiable w.r.t. ``x``.
    :param x: Positions (N, 3).
    :return: The scalar trace of dv/dx."""

    return sum(
        torch.autograd.grad(v.flatten()[i], x, retain_graph=True)[0].flatten()[i]
        for i in range(v.numel())
    )


class TestClosedFormDivergence:
    """The closed form against the autograd trace of the same velocity field."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_implicit_all_pairs(self, system, encoding):
        """No edges and no radius is all-pairs, which needs no envelope: it is smooth everywhere.

        The graph has to come from the distance side rather than a static ``edge_index``, since
        static edges are excluded from this field entirely.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x = system
        x = x.clone().requires_grad_(True)
        net = make_field(encoding=encoding)

        v, div = net(h_node, x)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_radius_graph(self, system, encoding):
        """With a radius graph the envelope is what keeps the closed form exact.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x = system
        x = x.clone().requires_grad_(True)
        net = make_field(encoding=encoding, distance_cutoff=1.0)

        v, div = net(h_node, x)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_radius_only_with_edge_features(self, system, encoding):
        """No static edges, and the coefficient head reading edge features.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x = system
        x = x.clone().requires_grad_(True)
        net = make_field(encoding=encoding, edge_dim=4, distance_cutoff=1.0)

        v, div = net(h_node, x)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_periodic_box(self, system, encoding):
        """Minimum-image wrapping is a locally constant shift, so d(r_ij)/d(x_i) = I holds.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x = system
        x = x.clone().requires_grad_(True)
        box = torch.full((x.shape[0], 3), 1.5, dtype=torch.float64)
        net = make_field(encoding=encoding)

        v, div = net(h_node, x, box=box)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    def test_continuous_across_the_radius_boundary(self):
        """The envelope makes an edge entering or leaving the radius graph a no-op.

        Without it, phi and its derivative would jump when a pair crosses the cutoff, and the
        integrated log-det would pick up an error every time an edge appeared."""
        cutoff = 1.0
        net = make_field(distance_cutoff=cutoff, cutoff=cutoff)
        h_node = torch.randn(2, 8, dtype=torch.float64)

        eps = 1e-7
        out = []
        for d in (cutoff - eps, cutoff + eps):
            x = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
            out.append(net(h_node, x))

        (v_in, div_in), (v_out, div_out) = out
        assert torch.allclose(v_in, v_out, atol=1e-10)
        assert torch.allclose(div_in, div_out, atol=1e-10)
        # and the pair contributes nothing at all at the boundary.
        assert torch.allclose(v_out, torch.zeros_like(v_out), atol=1e-10)

    def test_a_static_edge_contributes_nothing(self):
        """A static edge is the local backbone's to correct, so this field must not see it at all.

        Supplying one may not change the field by so much as a rounding error -- this is the
        exclusion list, and it has to hold *while* genuine dynamic edges are present, which is what
        separates it from the field being trivially zero.

        :return: None."""
        cutoff = 1.0
        h_node = torch.randn(3, 8, dtype=torch.float64)
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [2.5 * cutoff, 0.0, 0.0]], dtype=torch.float64
        )
        long_edge = torch.tensor([[0, 2], [2, 0]])
        net = make_field(distance_cutoff=cutoff, cutoff=4.0)

        x = pos.clone().requires_grad_(True)
        v, div = net(h_node, x, edge_index=long_edge)
        v_without = net(h_node, pos.clone().requires_grad_(True))[0]

        assert torch.allclose(v, v_without, atol=1e-12)
        # and the dynamic pair 0-1 is inside the radius, so this is not the all-zero field.
        assert not torch.allclose(v, torch.zeros_like(v))
        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    def test_a_static_edge_is_excluded_at_bonded_range_too(self):
        """The exclusion is by topology, not by distance: a short bond is dropped like any other.

        This is the pathology the exclusion exists for -- a bonded pair sits well inside every
        radius, so before the exclusion it picked up a full-strength, untapered ``phi(d)`` on top
        of whatever the local backbone already did with it.

        :return: None."""
        h_node = torch.randn(2, 8, dtype=torch.float64)
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float64)
        bond = torch.tensor([[0, 1], [1, 0]])
        net = make_field(distance_cutoff=1.0, cutoff=2.0)

        # 0.1 is far inside the radius, so the envelope is ~1 there: exempting this edge from the
        # taper (what the field used to do) leaves it contributing in full. Excluding it does not.
        v_dynamic = net(h_node, pos.clone())[0]
        v_declared_bond = net(h_node, pos.clone(), edge_index=bond)[0]

        assert not torch.allclose(v_dynamic, torch.zeros_like(v_dynamic))
        assert torch.allclose(v_declared_bond, torch.zeros_like(v_declared_bond))


class TestEquivariance:
    """E(3) equivariance of the velocity and invariance of the divergence."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_rotation_and_translation(self, system, encoding):
        """Rotating and translating the input rotates the velocity and leaves the divergence.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x = system
        net = make_field(encoding=encoding, distance_cutoff=1.0)

        R = rotation_z(0.7)
        shift = torch.tensor([1.3, -0.4, 2.0], dtype=torch.float64)

        v, div = net(h_node, x)
        v_t, div_t = net(h_node, x @ R.T + shift)

        assert torch.allclose(v_t, v @ R.T, atol=1e-10)
        assert torch.allclose(div_t, div, atol=1e-10)

    def test_reflection(self, system):
        """The field carries no chirality term, so it is E(3)- not merely SE(3)-equivariant.

        :param system: System fixture."""
        h_node, x = system
        net = make_field(distance_cutoff=1.0)

        M = reflection_z()
        v, div = net(h_node, x)
        v_m, div_m = net(h_node, x @ M.T)

        assert torch.allclose(v_m, v @ M.T, atol=1e-10)
        assert torch.allclose(div_m, div, atol=1e-10)


class TestBatching:
    """Per-graph reduction of the divergence."""

    def test_divergence_is_per_graph(self):
        """One divergence per graph, and a graph is unaffected by the other's nodes."""
        net = make_field(distance_cutoff=1.0)
        torch.manual_seed(11)
        n = 5
        x0 = torch.rand(n, 3, dtype=torch.float64)
        x1 = torch.rand(n, 3, dtype=torch.float64)
        h0 = torch.randn(n, 8, dtype=torch.float64)
        h1 = torch.randn(n, 8, dtype=torch.float64)
        batch = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

        v, div = net(torch.cat([h0, h1]), torch.cat([x0, x1]), batch=batch)
        assert div.shape == (2,)

        # perturbing graph 1 must leave graph 0 alone.
        v_p, div_p = net(
            torch.cat([h0, h1]), torch.cat([x0, x1 + 0.3]), batch=batch
        )
        assert torch.allclose(div_p[0], div[0], atol=1e-12)
        assert torch.allclose(v_p[:n], v[:n], atol=1e-12)

    def test_single_graph_returns_one_divergence(self, system):
        """With no batch vector the whole input is one graph.

        :param system: System fixture."""
        h_node, x = system
        v, div = make_field()(h_node, x)

        assert v.shape == x.shape
        assert div.shape == (1,)


class TestStaticEdgesAreExcluded:
    """The exclusion list: static edges belong to the local backbone, never to this field."""

    def test_no_dynamic_mechanism_is_identically_zero(self, system):
        """Static edges and no way to discover any others leaves nothing to score.

        Documented rather than rejected: the same state is reachable through the per-call radius
        override, so there is no construction-time configuration to validate.

        :param system: System fixture."""
        h_node, x = system
        net = make_field(distance_cutoff=0.0, num_nearest_neighbors=0)

        v, div = net(h_node, x, edge_index=full_edge_index(x.shape[0]))

        assert torch.allclose(v, torch.zeros_like(v))
        assert torch.allclose(div, torch.zeros_like(div))

    def test_weights_cannot_move_a_static_only_field(self, system):
        """No draw of the coefficient head may put velocity on an all-static graph.

        :param system: System fixture."""
        h_node, x = system
        edge_index = full_edge_index(x.shape[0])

        for seed in (0, 1, 2):
            v, div = make_field(seed=seed, distance_cutoff=0.0)(h_node, x, edge_index=edge_index)

            assert torch.allclose(v, torch.zeros_like(v))
            assert torch.allclose(div, torch.zeros_like(div))


class TestInitialisation:
    """The zero-init contract the flow relies on."""

    def test_zero_init_is_the_identity_flow(self, system):
        """A fresh field has zero velocity and zero divergence, i.e. unit Jacobian determinant.

        :param system: System fixture."""
        h_node, x = system
        net = RadialField(dim=8, encoding_features=6, cutoff=2.0).double()

        v, div = net(h_node, x)

        assert torch.allclose(v, torch.zeros_like(v))
        assert torch.allclose(div, torch.zeros_like(div))


class TestTimeAsANodeFeature:
    """Time reaches the field through a node-feature channel, not an argument of its own."""

    def test_a_time_channel_changes_the_field(self, system):
        """The coefficient head reads every node feature, so a time channel conditions the field.

        This is the whole of the time contract after the signature was aligned with
        ``GeometricEGNN.forward``: the caller widens ``dim`` by one and writes ``t`` into the extra
        channel. Nothing in the closed form notices, because a node feature does not depend on
        positions.

        :param system: System fixture."""
        h_node, x = system
        net = make_field(dim=h_node.shape[-1] + 1)

        def at(time: float) -> Tensor:
            t = torch.full((h_node.shape[0], 1), time, dtype=torch.float64)
            return net(torch.cat([h_node, t], dim=-1), x)[0]

        assert not torch.allclose(at(0.1), at(0.9), atol=1e-6)

    def test_a_time_channel_keeps_the_closed_form_exact(self, system):
        """Conditioning on time must not disturb the divergence it reports.

        :param system: System fixture."""
        h_node, x = system
        net = make_field(dim=h_node.shape[-1] + 1)
        t = torch.full((h_node.shape[0], 1), 0.37, dtype=torch.float64)

        x = x.clone().requires_grad_(True)
        v, div = net(torch.cat([h_node, t], dim=-1), x)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)


class TestPerCallCutoff:
    """The radius may be given per call; the envelope and its derivative must follow it."""

    def test_matches_a_field_built_at_that_radius(self):
        """The override changes the neighborhood and nothing else."""
        common = dict(cutoff=3.0)
        built_at_1 = make_field(distance_cutoff=1.0, **common)
        built_at_3 = make_field(distance_cutoff=3.0, **common)
        built_at_3.load_state_dict(built_at_1.state_dict())
        h_node = torch.randn(6, 8, dtype=torch.float64)
        x = torch.rand(6, 3, dtype=torch.float64) * 2.5

        v, div = built_at_1(h_node, x, distance_cutoff=3.0)
        v_ref, div_ref = built_at_3(h_node, x)
        v_narrow, _ = built_at_1(h_node, x)

        assert torch.allclose(v, v_ref)
        assert torch.allclose(div, div_ref)
        assert not torch.allclose(v, v_narrow)

    def test_the_closed_form_stays_exact(self):
        """``env`` and ``d_env`` must taper at the *same* radius, or the closed form drifts.

        This is the test that catches a per-call radius reaching ``polynomial_envelope`` but not
        ``polynomial_envelope_derivative``: the field would still look plausible while its
        reported divergence quietly stopped being the divergence of it."""
        net = make_field(distance_cutoff=1.0, cutoff=3.0)
        h_node = torch.randn(6, 8, dtype=torch.float64)
        x = (torch.rand(6, 3, dtype=torch.float64) * 2.5).requires_grad_(True)

        v, div = net(h_node, x, distance_cutoff=2.5)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    def test_continuous_across_the_overridden_boundary(self):
        """An edge crossing the *called* radius must still be a no-op, not the built one."""
        net = make_field(distance_cutoff=0.4, cutoff=1.0)
        h_node = torch.randn(2, 8, dtype=torch.float64)
        radius, eps = 1.0, 1e-7

        out = []
        for d in (radius - eps, radius + eps):
            x = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
            out.append(net(h_node, x, distance_cutoff=radius))

        (v_in, div_in), (v_out, div_out) = out
        assert torch.allclose(v_in, v_out, atol=1e-10)
        assert torch.allclose(div_in, div_out, atol=1e-10)

    def test_per_call_zero_radius_with_only_static_edges_is_exact_zero(self):
        """A zero radius drops the dynamic graph, and static edges were never in the field.

        This is the case no constructor-time guard could catch: the field was *built* with a
        radius, and only the call walks it into having nothing to score."""
        net = make_field(distance_cutoff=1.0, cutoff=3.0)
        h_node = torch.randn(3, 8, dtype=torch.float64)
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=torch.float64
        )
        bond = torch.tensor([[0, 1], [1, 0]])

        v, div = net(h_node, pos.clone(), edge_index=bond, distance_cutoff=0.0)

        assert torch.allclose(v, torch.zeros_like(v))
        assert torch.allclose(div, torch.zeros_like(div))
        # the same field does have something to say when its own radius graph is allowed.
        v_with_radius = net(h_node, pos.clone(), edge_index=bond)[0]
        assert not torch.allclose(v_with_radius, torch.zeros_like(v_with_radius))
