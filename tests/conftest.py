import math

import pytest
import torch
from torch import Tensor


def rotation_z(theta: float) -> Tensor:
    """Rotation matrix about the z-axis by ``theta`` radians."""
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def system() -> tuple[Tensor, Tensor, Tensor]:
    """Two batches with non-cubic cells and atoms placed in the interior.

    :return: Features (B, N, 8), coordinates (B, N, 3), box lengths (B, 3)."""
    torch.manual_seed(42)
    b, n = 2, 6
    L = torch.tensor([[10.0, 10.0, 10.0], [12.0, 10.0, 8.0]])
    coors = torch.rand(b, n, 3) * L[:, None, :] * 0.4 + L[:, None, :] * 0.3
    feats = torch.randn(b, n, 8)
    return feats, coors, L


@pytest.fixture
def compact_system() -> tuple[Tensor, Tensor, Tensor]:
    """Atoms clustered near the box center so minimum-image wrapping never fires.

    This is the condition under which rotation equivariance is exact with an axis-aligned
    minimum-image convention.

    :return: Features (B, N, 8), coordinates (B, N, 3), box lengths (B, 3)."""
    torch.manual_seed(7)
    b, n = 1, 6
    L_val = 20.0
    L = torch.full((b, 3), L_val)
    coors = L_val / 2.0 + 0.5 * torch.randn(b, n, 3)
    feats = torch.randn(b, n, 8)
    return feats, coors, L
