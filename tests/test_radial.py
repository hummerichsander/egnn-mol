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
        dim=8, encoding="gaussian", encoding_features=6, cutoff=2.0, time_features=5,
        m_dim=16, head_depth=2,
    )
    defaults.update(kwargs)
    net = RadialField(**defaults).double().eval()

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))

    return net


@pytest.fixture
def system() -> tuple[Tensor, Tensor, Tensor]:
    """Seven nodes in a 2 nm cube, in double precision.

    :return: Features h_node (N, 8), positions x (N, 3), per-node time t (N, 1)."""
    torch.manual_seed(3)
    n = 7
    x = torch.rand(n, 3, dtype=torch.float64) * 2.0
    h_node = torch.randn(n, 8, dtype=torch.float64)
    return h_node, x, torch.full((n, 1), 0.37, dtype=torch.float64)


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
    def test_static_all_pairs(self, system, encoding):
        """A static edge set needs no envelope: the field is smooth everywhere.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x, t = system
        x = x.clone().requires_grad_(True)
        net = make_field(encoding=encoding)

        v, div = net(h_node, x, t, edge_index=full_edge_index(x.shape[0]))

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_static_union_radius(self, system, encoding):
        """With a radius graph the envelope is what keeps the closed form exact.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x, t = system
        x = x.clone().requires_grad_(True)
        net = make_field(encoding=encoding, distance_cutoff=1.0)

        v, div = net(h_node, x, t, edge_index=full_edge_index(x.shape[0]))

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_radius_only_with_edge_features(self, system, encoding):
        """No static edges, and the coefficient head reading edge features.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x, t = system
        x = x.clone().requires_grad_(True)
        net = make_field(encoding=encoding, edge_dim=4, distance_cutoff=1.0)

        v, div = net(h_node, x, t)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_periodic_box(self, system, encoding):
        """Minimum-image wrapping is a locally constant shift, so d(r_ij)/d(x_i) = I holds.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x, t = system
        x = x.clone().requires_grad_(True)
        box = torch.full((x.shape[0], 3), 1.5, dtype=torch.float64)
        net = make_field(encoding=encoding)

        v, div = net(h_node, x, t, edge_index=full_edge_index(x.shape[0]), box=box)

        assert torch.allclose(div.squeeze(), autograd_trace(v, x), rtol=1e-9, atol=1e-9)

    def test_continuous_across_the_radius_boundary(self):
        """The envelope makes an edge entering or leaving the radius graph a no-op.

        Without it, phi and its derivative would jump when a pair crosses the cutoff, and the
        integrated log-det would pick up an error every time an edge appeared."""
        cutoff = 1.0
        net = make_field(distance_cutoff=cutoff, cutoff=cutoff)
        h_node = torch.randn(2, 8, dtype=torch.float64)
        t = torch.full((2, 1), 0.5, dtype=torch.float64)

        eps = 1e-7
        out = []
        for d in (cutoff - eps, cutoff + eps):
            x = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
            out.append(net(h_node, x, t))

        (v_in, div_in), (v_out, div_out) = out
        assert torch.allclose(v_in, v_out, atol=1e-10)
        assert torch.allclose(div_in, div_out, atol=1e-10)
        # and the pair contributes nothing at all at the boundary.
        assert torch.allclose(v_out, torch.zeros_like(v_out), atol=1e-10)


class TestEquivariance:
    """E(3) equivariance of the velocity and invariance of the divergence."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_rotation_and_translation(self, system, encoding):
        """Rotating and translating the input rotates the velocity and leaves the divergence.

        :param system: System fixture.
        :param encoding: Radial basis under test."""
        h_node, x, t = system
        net = make_field(encoding=encoding, distance_cutoff=1.0)
        edge_index = full_edge_index(x.shape[0])

        R = rotation_z(0.7)
        shift = torch.tensor([1.3, -0.4, 2.0], dtype=torch.float64)

        v, div = net(h_node, x, t, edge_index=edge_index)
        v_t, div_t = net(h_node, x @ R.T + shift, t, edge_index=edge_index)

        assert torch.allclose(v_t, v @ R.T, atol=1e-10)
        assert torch.allclose(div_t, div, atol=1e-10)

    def test_reflection(self, system):
        """The field carries no chirality term, so it is E(3)- not merely SE(3)-equivariant.

        :param system: System fixture."""
        h_node, x, t = system
        net = make_field(distance_cutoff=1.0)
        edge_index = full_edge_index(x.shape[0])

        M = reflection_z()
        v, div = net(h_node, x, t, edge_index=edge_index)
        v_m, div_m = net(h_node, x @ M.T, t, edge_index=edge_index)

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
        t = torch.full((2 * n, 1), 0.4, dtype=torch.float64)
        batch = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

        v, div = net(torch.cat([h0, h1]), torch.cat([x0, x1]), t, batch=batch)
        assert div.shape == (2,)

        # perturbing graph 1 must leave graph 0 alone.
        v_p, div_p = net(
            torch.cat([h0, h1]), torch.cat([x0, x1 + 0.3]), t, batch=batch
        )
        assert torch.allclose(div_p[0], div[0], atol=1e-12)
        assert torch.allclose(v_p[:n], v[:n], atol=1e-12)

    def test_single_graph_returns_one_divergence(self, system):
        """With no batch vector the whole input is one graph.

        :param system: System fixture."""
        h_node, x, t = system
        v, div = make_field()(h_node, x, t, edge_index=full_edge_index(x.shape[0]))

        assert v.shape == x.shape
        assert div.shape == (1,)


class TestInitialisation:
    """The zero-init contract the flow relies on."""

    def test_zero_init_is_the_identity_flow(self, system):
        """A fresh field has zero velocity and zero divergence, i.e. unit Jacobian determinant.

        :param system: System fixture."""
        h_node, x, t = system
        net = RadialField(dim=8, encoding_features=6, cutoff=2.0, time_features=5).double()

        v, div = net(h_node, x, t, edge_index=full_edge_index(x.shape[0]))

        assert torch.allclose(v, torch.zeros_like(v))
        assert torch.allclose(div, torch.zeros_like(div))


class TestTimeInput:
    """Shape tolerance of the per-node time input."""

    def test_flat_and_column_time_agree(self, system):
        """``t`` may arrive as (N,) or (N, 1).

        :param system: System fixture."""
        h_node, x, t = system
        net = make_field()
        edge_index = full_edge_index(x.shape[0])

        v_col, div_col = net(h_node, x, t, edge_index=edge_index)
        v_flat, div_flat = net(h_node, x, t.squeeze(-1), edge_index=edge_index)

        assert torch.allclose(v_col, v_flat)
        assert torch.allclose(div_col, div_flat)

    def test_time_changes_the_field(self, system):
        """The coefficient head reads time, so the field is genuinely time-dependent.

        :param system: System fixture."""
        h_node, x, t = system
        net = make_field()
        edge_index = full_edge_index(x.shape[0])

        v_early, _ = net(h_node, x, torch.full_like(t, 0.1), edge_index=edge_index)
        v_late, _ = net(h_node, x, torch.full_like(t, 0.9), edge_index=edge_index)

        assert not torch.allclose(v_early, v_late, atol=1e-6)
