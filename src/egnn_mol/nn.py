import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Feed-forward MLP: ``num_layers`` hidden blocks with a shared activation.

    A single module covers every small network the backbones need. ``final_activation``
    appends the activation after the output layer, which the edge network uses so its
    messages pass through a nonlinearity."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        final_activation: bool = False,
    ) -> None:
        """Build the MLP.

        :param in_dim: Input dimensionality.
        :param hidden_dim: Hidden dimensionality.
        :param out_dim: Output dimensionality.
        :param num_layers: Number of hidden blocks (a single block is one hidden layer).
        :param activation: Activation module class.
        :param dropout: Dropout probability applied before each linear beyond the first.
        :param final_activation: Whether to apply the activation after the output layer."""

        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(num_layers - 1):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        if final_activation:
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """:param x: Input (..., in_dim). :return: Output (..., out_dim)."""
        return self.net(x)


class PosNorm(nn.Module):
    """Normalize displacement vectors to unit length, then rescale by a learnable factor.

    Normalizing keeps position-update magnitudes independent of the box / bond lengths,
    which matters under periodic boundary conditions."""

    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0) -> None:
        """:param eps: Denominator clamp to avoid division by zero.
        :param scale_init: Initial value of the learnable output scale."""
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(1).fill_(scale_init))

    def forward(self, pos: Tensor) -> Tensor:
        """:param pos: Displacement vectors (..., 3). :return: Rescaled unit vectors (..., 3)."""
        norm = pos.norm(dim=-1, keepdim=True)
        return pos / norm.clamp(min=self.eps) * self.scale
