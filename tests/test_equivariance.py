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


def _run_layer(layer, h_node, x, box):
    return layer(h_node, x, box=box)


def _run_net(net, h_node, x, box):
    return net(h_node, x, box=box)


class TestPeriodicity:
    """Features invariant and velocity (displacement) equivariant under lattice / arbitrary shifts."""

    def test_layer_features_periodic(self, system):
        h_node, x, box = system
        layer = make_layer()
        with torch.no_grad():
            h_node_out, _ = _run_layer(layer, h_node, x, box)
            h_node_shift, _ = _run_layer(layer, h_node, x + box[:, None, :], box)
        assert torch.allclose(h_node_out, h_node_shift, atol=1e-5)

    def test_layer_velocity_periodic(self, system):
        h_node, x, box = system
        layer = make_layer()
        shift = box[:, None, :]
        with torch.no_grad():
            _, x_out = _run_layer(layer, h_node, x, box)
            _, x_shift = _run_layer(layer, h_node, x + shift, box)
        assert torch.allclose(x_out - x, x_shift - (x + shift), atol=1e-5)

    def test_net_features_invariant_to_translation(self, system):
        h_node, x, box = system
        net = make_network()
        delta = torch.tensor([[[3.1, -1.7, 4.2]], [[-2.3, 5.0, -0.8]]])
        with torch.no_grad():
            h_node_out, _ = _run_net(net, h_node, x, box)
            h_node_shift, _ = _run_net(net, h_node, x + delta, box)
        assert torch.allclose(h_node_out, h_node_shift, atol=1e-5)

    def test_net_velocity_invariant_to_translation(self, system):
        h_node, x, box = system
        net = make_network()
        delta = torch.tensor([[[3.1, -1.7, 4.2]], [[-2.3, 5.0, -0.8]]])
        with torch.no_grad():
            _, x_out = _run_net(net, h_node, x, box)
            _, x_shift = _run_net(net, h_node, x + delta, box)
        assert torch.allclose(x_out - x, x_shift - (x + delta), atol=1e-5)

    def test_raw_pos_not_periodic(self, system):
        """x_out(x + L) == x_out(x) + L, so only the displacement is periodic."""
        h_node, x, box = system
        net = make_network()
        shift = box[:, None, :]
        with torch.no_grad():
            _, x_out = _run_net(net, h_node, x, box)
            _, x_shift = _run_net(net, h_node, x + shift, box)
        assert not torch.allclose(x_out, x_shift, atol=1e-3)


class TestEquivariance:
    """Rotation invariance of features, rotation equivariance of velocity, permutation equivariance."""

    def test_features_invariant_to_rotation(self, compact_system):
        h_node, x, box = compact_system
        net = make_network()
        R = rotation_z(math.pi / 5)
        centroid = x.mean(dim=1, keepdim=True)
        x_rot = (x - centroid) @ R.T + centroid
        with torch.no_grad():
            h_node_out, _ = _run_net(net, h_node, x, box)
            h_node_rot, _ = _run_net(net, h_node, x_rot, box)
        assert torch.allclose(h_node_out, h_node_rot, atol=1e-5)

    def test_velocity_equivariant_to_rotation(self, compact_system):
        h_node, x, box = compact_system
        net = make_network()
        R = rotation_z(math.pi / 5)
        centroid = x.mean(dim=1, keepdim=True)
        x_rot = (x - centroid) @ R.T + centroid
        with torch.no_grad():
            _, x_out = _run_net(net, h_node, x, box)
            _, x_rot_out = _run_net(net, h_node, x_rot, box)
        assert torch.allclose(x_rot_out - x_rot, (x_out - x) @ R.T, atol=1e-5)

    def test_permutation_equivariance(self, system):
        h_node, x, box = system
        net = make_network()
        perm = torch.randperm(x.shape[1])
        with torch.no_grad():
            h_node_out, x_out = _run_net(net, h_node, x, box)
            h_node_perm, x_perm = _run_net(net, h_node[:, perm], x[:, perm], box)
        assert torch.allclose(h_node_out[:, perm], h_node_perm, atol=1e-5)
        assert torch.allclose((x_out - x)[:, perm], x_perm - x[:, perm], atol=1e-5)


class TestSE3TripleProduct:
    """The triple-product term keeps rotation equivariance but breaks reflection equivariance.

    That is exactly the SE(3) (chirality-aware) property: proper rotations are still symmetries,
    improper ones (reflections) are not, whereas the plain E(3) model is symmetric under both.
    """

    # norm_displacement + a single layer keep position magnitudes O(1) so exact symmetries certify at a
    # tight relative tolerance while randomized weights make the chirality term measurable.
    NET_KW = dict(depth=1, norm_displacement=True, norm_displacement_scale_init=1.0)

    def _transform(self, x, M):
        centroid = x.mean(dim=1, keepdim=True)
        return (x - centroid) @ M.T + centroid

    def test_rotation_equivariance_holds_with_triple_product(self, compact_system):
        h_node, x, box = compact_system
        net = randomize(make_network(tripp_num_layers=2, **self.NET_KW))
        R = rotation_z(math.pi / 5)
        x_rot = self._transform(x, R)
        with torch.no_grad():
            _, x_out = _run_net(net, h_node, x, box)
            _, x_rot_out = _run_net(net, h_node, x_rot, box)
        assert rel_err(x_rot_out - x_rot, (x_out - x) @ R.T) < 1e-4

    def test_reflection_equivariance_only_without_triple_product(self, compact_system):
        h_node, x, box = compact_system
        M = reflection_z()
        x_ref = self._transform(x, M)

        e3 = randomize(make_network(tripp_num_layers=0, **self.NET_KW))
        with torch.no_grad():
            _, x_out = _run_net(e3, h_node, x, box)
            _, x_ref_out = _run_net(e3, h_node, x_ref, box)
        assert rel_err(x_ref_out - x_ref, (x_out - x) @ M.T) < 1e-4  # E(3): symmetric

        se3 = randomize(make_network(tripp_num_layers=2, **self.NET_KW))
        with torch.no_grad():
            _, x_out = _run_net(se3, h_node, x, box)
            _, x_ref_out = _run_net(se3, h_node, x_ref, box)
        assert rel_err(x_ref_out - x_ref, (x_out - x) @ M.T) > 1e-2  # SE(3): broken
