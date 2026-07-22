"""Radial distance encodings and smooth cutoff envelopes.

Each encoding is a plain function ``(dist, num_features, cutoff) -> (..., width)`` taking the
true L2 distance. New bases are added by writing one function and one ``case`` arm in
:func:`encode_distance` — no registry, no base class."""

from typing import Literal

import torch
from torch import Tensor

Encoding = Literal["bessel", "fourier", "gaussian"]


def bessel(dist: Tensor, num_features: int, cutoff: float) -> Tensor:
    """Orthonormal Bessel radial basis from DimeNet (https://arxiv.org/abs/2003.03123).

    ``e_n(d) = sqrt(2/c) * sin(n*pi*d/c) / d``, n = 1..num_features, evaluated via
    ``torch.sinc`` for numerical safety near d = 0. The basis is orthonormal on [0, cutoff]
    and vanishes at d = cutoff, giving a smooth implicit cutoff with no extra envelope.

    :param dist: True L2 distances (..., 1), non-negative.
    :param num_features: Number of Bessel basis functions.
    :param cutoff: Support of the basis; all functions vanish at d = cutoff.
    :return: Bessel encoding (..., num_features)."""

    freq = (
        torch.arange(1, num_features + 1, device=dist.device, dtype=dist.dtype)
        * torch.pi
        / cutoff
    )
    return (2.0 / cutoff) ** 0.5 * freq * torch.sinc(freq * dist / torch.pi)


def fourier(dist: Tensor, num_features: int, cutoff: float) -> Tensor:
    """Sinusoidal Fourier features over ``num_features`` octave-spaced frequency bands.

    :param dist: True L2 distances (..., 1), non-negative.
    :param num_features: Number of frequency bands; the output width is ``2 * num_features``.
    :param cutoff: Length scale; the lowest band has wavelength ~cutoff.
    :return: Concatenated sin/cos features (..., 2 * num_features)."""

    bands = 2.0 ** torch.arange(num_features, device=dist.device, dtype=dist.dtype)
    scaled = dist * (torch.pi / cutoff) * bands
    return torch.cat([scaled.sin(), scaled.cos()], dim=-1)


def gaussian(dist: Tensor, num_features: int, cutoff: float) -> Tensor:
    """Gaussian radial basis with fixed centers linearly spaced on [0, cutoff].

    :param dist: True L2 distances (..., 1), non-negative.
    :param num_features: Number of Gaussian centers.
    :param cutoff: Position of the last center; also sets the shared width.
    :return: Gaussian RBF encoding (..., num_features)."""

    centers = torch.linspace(
        0.0, cutoff, num_features, device=dist.device, dtype=dist.dtype
    )
    spacing = cutoff / max(num_features - 1, 1)
    return torch.exp(-0.5 * ((dist - centers) / spacing) ** 2)


def encode_distance(
    dist: Tensor, kind: Encoding, num_features: int, cutoff: float
) -> Tensor:
    """Dispatch to the requested radial basis.

    :param dist: True L2 distances (..., 1), non-negative.
    :param kind: Which encoding to use.
    :param num_features: Number of basis functions / frequency bands.
    :param cutoff: Radial length scale.
    :return: Encoded distances (..., ``encoding_width(kind, num_features)``)."""

    match kind:
        case "bessel":
            return bessel(dist, num_features, cutoff)
        case "fourier":
            return fourier(dist, num_features, cutoff)
        case "gaussian":
            return gaussian(dist, num_features, cutoff)
        case _:
            raise ValueError(f"unknown distance encoding: {kind!r}")


def encoding_width(kind: Encoding, num_features: int) -> int:
    """Output feature width of an encoding, so a backbone can size its edge MLP.

    :param kind: Which encoding to use.
    :param num_features: Number of basis functions / frequency bands.
    :return: Width of the last axis produced by :func:`encode_distance`."""

    return 2 * num_features if kind == "fourier" else num_features


def polynomial_envelope(dist: Tensor, cutoff: float, exponent: int = 6) -> Tensor:
    """DimeNet polynomial cutoff envelope, smooth with zero value and slope at d = cutoff.

    :param dist: True L2 distances (..., 1), non-negative.
    :param cutoff: Cutoff radius; the envelope is 0 for d >= cutoff.
    :param exponent: Polynomial exponent p controlling smoothness.
    :return: Envelope weights in [0, 1] (..., 1)."""

    p = exponent
    d = dist / cutoff
    env = (
        1.0
        - (p + 1) * (p + 2) / 2 * d**p
        + p * (p + 2) * d ** (p + 1)
        - p * (p + 1) / 2 * d ** (p + 2)
    )
    return torch.where(dist < cutoff, env, torch.zeros_like(env))


def cosine_envelope(dist: Tensor, cutoff: float) -> Tensor:
    """Cosine (Behler) cutoff envelope, 0.5 * (cos(pi * d / cutoff) + 1) below the cutoff.

    :param dist: True L2 distances (..., 1), non-negative.
    :param cutoff: Cutoff radius; the envelope is 0 for d >= cutoff.
    :return: Envelope weights in [0, 1] (..., 1)."""

    env = 0.5 * (torch.cos(torch.pi * dist / cutoff) + 1.0)
    return torch.where(dist < cutoff, env, torch.zeros_like(env))
