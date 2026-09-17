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


class DisplacementNorm(nn.Module):
    """Normalize displacement vectors to unit length, then rescale by a learnable factor.

    Normalizing keeps position-update magnitudes independent of the box / bond lengths,
    which matters under periodic boundary conditions."""

    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0) -> None:
        """:param eps: Denominator clamp to avoid division by zero.
        :param scale_init: Initial value of the learnable output scale."""
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(1).fill_(scale_init))

    def forward(self, x: Tensor) -> Tensor:
        """:param x: Displacement vectors (..., 3). :return: Rescaled unit vectors (..., 3)."""
        norm = x.norm(dim=-1, keepdim=True)
        return x / norm.clamp(min=self.eps) * self.scale


class VectorNorm(nn.Module):
    """RMS-normalize equivariant vector features, with a learnable gain per channel.

    The scalar counterpart is a ``LayerNorm``, which this deliberately is not: subtracting a mean
    over the spatial axis, or adding a bias, would add something that does not rotate with the
    system and the features would stop being vectors. Only multiplication by an invariant scalar
    survives, so the scale is an RMS over the channels' own norms and the gain is a scalar per
    channel."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        """:param channels: Number of vector channels.
        :param eps: Added inside the square root, so the scale stays differentiable at zero."""
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(channels))

    def forward(self, vec: Tensor) -> Tensor:
        """:param vec: Vector features (..., channels, 3). :return: Normalized features."""
        # inside the root, not clamped outside it: the channels start at exactly zero, where a
        # norm has no derivative, and the divergence needs this to stay C^1.
        scale = vec.pow(2).sum(-1).mean(-1, keepdim=True).add(self.eps).sqrt()

        return vec / scale[..., None] * self.gain[..., None]
