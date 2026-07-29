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

    :return: Features h_node (B, N, 8), positions x (B, N, 3), box lengths (B, 3)."""
    torch.manual_seed(42)
    b, n = 2, 6
    box = torch.tensor([[10.0, 10.0, 10.0], [12.0, 10.0, 8.0]])
    x = torch.rand(b, n, 3) * box[:, None, :] * 0.4 + box[:, None, :] * 0.3
    h_node = torch.randn(b, n, 8)
    return h_node, x, box


@pytest.fixture
def compact_system() -> tuple[Tensor, Tensor, Tensor]:
    """Atoms clustered near the box center so minimum-image wrapping never fires.

    This is the condition under which rotation equivariance is exact with an axis-aligned
    minimum-image convention.

    :return: Features h_node (B, N, 8), positions x (B, N, 3), box lengths (B, 3)."""
    torch.manual_seed(7)
    b, n = 1, 6
    box_len = 20.0
    box = torch.full((b, 3), box_len)
    x = box_len / 2.0 + 0.5 * torch.randn(b, n, 3)
    h_node = torch.randn(b, n, 8)
    return h_node, x, box
