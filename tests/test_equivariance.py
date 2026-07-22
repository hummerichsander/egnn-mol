"""E(3)-equivariance and periodicity of the dense backbone (layer and full network)."""

import math

import torch

from egnn_mol import E3GNN
from egnn_mol.dense import DenseEGNNLayer
from conftest import rotation_z


def make_layer(**kwargs) -> DenseEGNNLayer:
    defaults = dict(dim=8, m_dim=8)
    defaults.update(kwargs)
    return DenseEGNNLayer(**defaults).eval()


def make_network(**kwargs) -> E3GNN:
    defaults = dict(depth=2, dim=8, m_dim=8)
    defaults.update(kwargs)
    return E3GNN(**defaults).eval()


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
