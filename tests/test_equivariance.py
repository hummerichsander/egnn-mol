import math

import torch
from torch import Tensor

from egnn_mol import EGNN
from egnn_mol.dense import DenseEGNNLayer
from conftest import rotation_z


def reflection_z() -> Tensor:
    """Improper (det = -1) reflection through the xy-plane."""
    return torch.diag(torch.tensor([1.0, 1.0, -1.0]))


def make_layer(**kwargs) -> DenseEGNNLayer:
    defaults = dict(dim=8, m_dim=8)
    defaults.update(kwargs)
    return DenseEGNNLayer(**defaults).eval()


def make_network(**kwargs) -> EGNN:
    defaults = dict(depth=2, dim=8, m_dim=8)
    defaults.update(kwargs)
    return EGNN(**defaults).eval()


def randomize(net: EGNN, seed: int = 0) -> EGNN:
    """Replace parameters with non-trivial random values (near-identity init hides the SE(3) term)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
    return net


def rel_err(a: Tensor, b: Tensor) -> float:
    """Relative L2 error, robust to the overall magnitude of the velocity field."""
    return (a - b).norm().item() / b.norm().clamp(min=1e-8).item()


def _run_layer(layer, x, pos, box):
    return layer(x, pos, box=box)


def _run_net(net, x, pos, box):
    return net(x, pos, box=box)


class TestPeriodicity:
    """Features invariant and velocity (displacement) equivariant under lattice / arbitrary shifts."""

    def test_layer_features_periodic(self, system):
        x, pos, box = system
        layer = make_layer()
        with torch.no_grad():
            x_out, _ = _run_layer(layer, x, pos, box)
            x_shift, _ = _run_layer(layer, x, pos + box[:, None, :], box)
        assert torch.allclose(x_out, x_shift, atol=1e-5)

    def test_layer_velocity_periodic(self, system):
        x, pos, box = system
        layer = make_layer()
        shift = box[:, None, :]
        with torch.no_grad():
            _, pos_out = _run_layer(layer, x, pos, box)
            _, pos_shift = _run_layer(layer, x, pos + shift, box)
        assert torch.allclose(pos_out - pos, pos_shift - (pos + shift), atol=1e-5)

    def test_net_features_invariant_to_translation(self, system):
        x, pos, box = system
        net = make_network()
        delta = torch.tensor([[[3.1, -1.7, 4.2]], [[-2.3, 5.0, -0.8]]])
        with torch.no_grad():
            x_out, _ = _run_net(net, x, pos, box)
            x_shift, _ = _run_net(net, x, pos + delta, box)
        assert torch.allclose(x_out, x_shift, atol=1e-5)

    def test_net_velocity_invariant_to_translation(self, system):
        x, pos, box = system
        net = make_network()
        delta = torch.tensor([[[3.1, -1.7, 4.2]], [[-2.3, 5.0, -0.8]]])
        with torch.no_grad():
            _, pos_out = _run_net(net, x, pos, box)
            _, pos_shift = _run_net(net, x, pos + delta, box)
        assert torch.allclose(pos_out - pos, pos_shift - (pos + delta), atol=1e-5)

    def test_raw_pos_not_periodic(self, system):
        """pos_out(pos + L) == pos_out(pos) + L, so only the displacement is periodic."""
        x, pos, box = system
        net = make_network()
        shift = box[:, None, :]
        with torch.no_grad():
            _, pos_out = _run_net(net, x, pos, box)
            _, pos_shift = _run_net(net, x, pos + shift, box)
        assert not torch.allclose(pos_out, pos_shift, atol=1e-3)


class TestEquivariance:
    """Rotation invariance of features, rotation equivariance of velocity, permutation equivariance."""

    def test_features_invariant_to_rotation(self, compact_system):
        x, pos, box = compact_system
        net = make_network()
        R = rotation_z(math.pi / 5)
        centroid = pos.mean(dim=1, keepdim=True)
        pos_rot = (pos - centroid) @ R.T + centroid
        with torch.no_grad():
            x_out, _ = _run_net(net, x, pos, box)
            x_rot, _ = _run_net(net, x, pos_rot, box)
        assert torch.allclose(x_out, x_rot, atol=1e-5)

    def test_velocity_equivariant_to_rotation(self, compact_system):
        x, pos, box = compact_system
        net = make_network()
        R = rotation_z(math.pi / 5)
        centroid = pos.mean(dim=1, keepdim=True)
        pos_rot = (pos - centroid) @ R.T + centroid
        with torch.no_grad():
            _, pos_out = _run_net(net, x, pos, box)
            _, pos_rot_out = _run_net(net, x, pos_rot, box)
        assert torch.allclose(pos_rot_out - pos_rot, (pos_out - pos) @ R.T, atol=1e-5)

    def test_permutation_equivariance(self, system):
        x, pos, box = system
        net = make_network()
        perm = torch.randperm(pos.shape[1])
        with torch.no_grad():
            x_out, pos_out = _run_net(net, x, pos, box)
            x_perm, pos_perm = _run_net(net, x[:, perm], pos[:, perm], box)
        assert torch.allclose(x_out[:, perm], x_perm, atol=1e-5)
        assert torch.allclose((pos_out - pos)[:, perm], pos_perm - pos[:, perm], atol=1e-5)


class TestSE3TripleProduct:
    """The triple-product term keeps rotation equivariance but breaks reflection equivariance.

    That is exactly the SE(3) (chirality-aware) property: proper rotations are still symmetries,
    improper ones (reflections) are not, whereas the plain E(3) model is symmetric under both.
    """

    # norm_pos + a single layer keep position magnitudes O(1) so exact symmetries certify at a
    # tight relative tolerance while randomized weights make the chirality term measurable.
    NET_KW = dict(depth=1, norm_pos=True, norm_pos_scale_init=1.0)

    def _transform(self, pos, M):
        centroid = pos.mean(dim=1, keepdim=True)
        return (pos - centroid) @ M.T + centroid

    def test_rotation_equivariance_holds_with_triple_product(self, compact_system):
        x, pos, box = compact_system
        net = randomize(make_network(tripp_num_layers=2, **self.NET_KW))
        R = rotation_z(math.pi / 5)
        pos_rot = self._transform(pos, R)
        with torch.no_grad():
            _, pos_out = _run_net(net, x, pos, box)
            _, pos_rot_out = _run_net(net, x, pos_rot, box)
        assert rel_err(pos_rot_out - pos_rot, (pos_out - pos) @ R.T) < 1e-4

    def test_reflection_equivariance_only_without_triple_product(self, compact_system):
        x, pos, box = compact_system
        M = reflection_z()
        pos_ref = self._transform(pos, M)

        e3 = randomize(make_network(tripp_num_layers=0, **self.NET_KW))
        with torch.no_grad():
            _, pos_out = _run_net(e3, x, pos, box)
            _, pos_ref_out = _run_net(e3, x, pos_ref, box)
        assert rel_err(pos_ref_out - pos_ref, (pos_out - pos) @ M.T) < 1e-4  # E(3): symmetric

        se3 = randomize(make_network(tripp_num_layers=2, **self.NET_KW))
        with torch.no_grad():
            _, pos_out = _run_net(se3, x, pos, box)
            _, pos_ref_out = _run_net(se3, x, pos_ref, box)
        assert rel_err(pos_ref_out - pos_ref, (pos_out - pos) @ M.T) > 1e-2  # SE(3): broken
