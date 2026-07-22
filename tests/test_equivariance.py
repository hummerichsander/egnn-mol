import math

import torch
from torch import Tensor

from egnn_mol import E3GNN
from egnn_mol.dense import DenseEGNNLayer
from conftest import rotation_z


def reflection_z() -> Tensor:
    """Improper (det = -1) reflection through the xy-plane."""
    return torch.diag(torch.tensor([1.0, 1.0, -1.0]))


def make_layer(**kwargs) -> DenseEGNNLayer:
    defaults = dict(dim=8, m_dim=8)
    defaults.update(kwargs)
    return DenseEGNNLayer(**defaults).eval()


def make_network(**kwargs) -> E3GNN:
    defaults = dict(depth=2, dim=8, m_dim=8)
    defaults.update(kwargs)
    return E3GNN(**defaults).eval()


def randomize(net: E3GNN, seed: int = 0) -> E3GNN:
    """Replace parameters with non-trivial random values (near-identity init hides the SE(3) term)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g))
    return net


def rel_err(a: Tensor, b: Tensor) -> float:
    """Relative L2 error, robust to the overall magnitude of the velocity field."""
    return (a - b).norm().item() / b.norm().clamp(min=1e-8).item()


def _run_layer(layer, feats, coors, L):
    return layer(feats, coors, L)


def _run_net(net, feats, coors, L):
    return net(feats, coors, L)


class TestPeriodicity:
    """Features invariant and velocity (displacement) equivariant under lattice / arbitrary shifts."""

    def test_layer_features_periodic(self, system):
        feats, coors, L = system
        layer = make_layer()
        with torch.no_grad():
            f_out, _ = _run_layer(layer, feats, coors, L)
            f_shift, _ = _run_layer(layer, feats, coors + L[:, None, :], L)
        assert torch.allclose(f_out, f_shift, atol=1e-5)

    def test_layer_velocity_periodic(self, system):
        feats, coors, L = system
        layer = make_layer()
        shift = L[:, None, :]
        with torch.no_grad():
            _, c_out = _run_layer(layer, feats, coors, L)
            _, c_shift = _run_layer(layer, feats, coors + shift, L)
        assert torch.allclose(c_out - coors, c_shift - (coors + shift), atol=1e-5)

    def test_net_features_invariant_to_translation(self, system):
        feats, coors, L = system
        net = make_network()
        delta = torch.tensor([[[3.1, -1.7, 4.2]], [[-2.3, 5.0, -0.8]]])
        with torch.no_grad():
            f_out, _ = _run_net(net, feats, coors, L)
            f_shift, _ = _run_net(net, feats, coors + delta, L)
        assert torch.allclose(f_out, f_shift, atol=1e-5)

    def test_net_velocity_invariant_to_translation(self, system):
        feats, coors, L = system
        net = make_network()
        delta = torch.tensor([[[3.1, -1.7, 4.2]], [[-2.3, 5.0, -0.8]]])
        with torch.no_grad():
            _, c_out = _run_net(net, feats, coors, L)
            _, c_shift = _run_net(net, feats, coors + delta, L)
        assert torch.allclose(c_out - coors, c_shift - (coors + delta), atol=1e-5)

    def test_raw_coors_not_periodic(self, system):
        """coors_out(x + L) == coors_out(x) + L, so only the displacement is periodic."""
        feats, coors, L = system
        net = make_network()
        shift = L[:, None, :]
        with torch.no_grad():
            _, c_out = _run_net(net, feats, coors, L)
            _, c_shift = _run_net(net, feats, coors + shift, L)
        assert not torch.allclose(c_out, c_shift, atol=1e-3)


class TestEquivariance:
    """Rotation invariance of features, rotation equivariance of velocity, permutation equivariance."""

    def test_features_invariant_to_rotation(self, compact_system):
        feats, coors, L = compact_system
        net = make_network()
        R = rotation_z(math.pi / 5)
        centroid = coors.mean(dim=1, keepdim=True)
        coors_rot = (coors - centroid) @ R.T + centroid
        with torch.no_grad():
            f_out, _ = _run_net(net, feats, coors, L)
            f_rot, _ = _run_net(net, feats, coors_rot, L)
        assert torch.allclose(f_out, f_rot, atol=1e-5)

    def test_velocity_equivariant_to_rotation(self, compact_system):
        feats, coors, L = compact_system
        net = make_network()
        R = rotation_z(math.pi / 5)
        centroid = coors.mean(dim=1, keepdim=True)
        coors_rot = (coors - centroid) @ R.T + centroid
        with torch.no_grad():
            _, c_out = _run_net(net, feats, coors, L)
            _, c_rot = _run_net(net, feats, coors_rot, L)
        assert torch.allclose(c_rot - coors_rot, (c_out - coors) @ R.T, atol=1e-5)

    def test_permutation_equivariance(self, system):
        feats, coors, L = system
        net = make_network()
        perm = torch.randperm(coors.shape[1])
        with torch.no_grad():
            f_out, c_out = _run_net(net, feats, coors, L)
            f_perm, c_perm = _run_net(net, feats[:, perm], coors[:, perm], L)
        assert torch.allclose(f_out[:, perm], f_perm, atol=1e-5)
        assert torch.allclose((c_out - coors)[:, perm], c_perm - coors[:, perm], atol=1e-5)


class TestSE3TripleProduct:
    """The triple-product term keeps rotation equivariance but breaks reflection equivariance.

    That is exactly the SE(3) (chirality-aware) property: proper rotations are still symmetries,
    improper ones (reflections) are not, whereas the plain E(3) model is symmetric under both.
    """

    # norm_coors + a single layer keep coordinate magnitudes O(1) so exact symmetries certify
    # at a tight relative tolerance while randomized weights make the chirality term measurable.
    NET_KW = dict(depth=1, norm_coors=True, norm_coors_scale_init=1.0)

    def _transform(self, coors, M):
        centroid = coors.mean(dim=1, keepdim=True)
        return (coors - centroid) @ M.T + centroid

    def test_rotation_equivariance_holds_with_triple_product(self, compact_system):
        feats, coors, L = compact_system
        net = randomize(make_network(tripp_num_layers=2, **self.NET_KW))
        R = rotation_z(math.pi / 5)
        coors_rot = self._transform(coors, R)
        with torch.no_grad():
            _, c_out = _run_net(net, feats, coors, L)
            _, c_rot = _run_net(net, feats, coors_rot, L)
        assert rel_err(c_rot - coors_rot, (c_out - coors) @ R.T) < 1e-4

    def test_reflection_equivariance_only_without_triple_product(self, compact_system):
        feats, coors, L = compact_system
        M = reflection_z()
        coors_ref = self._transform(coors, M)

        e3 = randomize(make_network(tripp_num_layers=0, **self.NET_KW))
        with torch.no_grad():
            _, c_out = _run_net(e3, feats, coors, L)
            _, c_ref = _run_net(e3, feats, coors_ref, L)
        assert rel_err(c_ref - coors_ref, (c_out - coors) @ M.T) < 1e-4  # E(3): symmetric

        se3 = randomize(make_network(tripp_num_layers=2, **self.NET_KW))
        with torch.no_grad():
            _, c_out = _run_net(se3, feats, coors, L)
            _, c_ref = _run_net(se3, feats, coors_ref, L)
        assert rel_err(c_ref - coors_ref, (c_out - coors) @ M.T) > 1e-2  # SE(3): broken
