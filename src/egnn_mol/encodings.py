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


def bessel_derivative(dist: Tensor, num_features: int, cutoff: float) -> Tensor:
    """Derivative of :func:`bessel` with respect to the distance.

    ``de_n/dd = sqrt(2/c) * freq^2 * (z cos z - sin z) / z^2`` with ``z = freq * d``. The
    numerator is O(z^3/3) while its two terms are O(z), so evaluating it directly loses all
    precision below ``z ~ 0.1`` in float32; a three-term Taylor branch covers that range and
    meets the direct form to ~1e-11 relative at the crossover.

    :param dist: True L2 distances (..., 1), non-negative.
    :param num_features: Number of Bessel basis functions.
    :param cutoff: Support of the basis.
    :return: Basis derivative (..., num_features)."""

    freq = (
        torch.arange(1, num_features + 1, device=dist.device, dtype=dist.dtype)
        * torch.pi
        / cutoff
    )
    z = freq * dist
    small = z.abs() < 0.1

    # the unused branch must stay finite: at z = 0 the direct form divides by zero, and
    # torch.where propagates the resulting NaN into the gradient of the taken branch.
    z_safe = torch.where(small, torch.ones_like(z), z)
    direct = (z_safe * z_safe.cos() - z_safe.sin()) / z_safe**2
    series = -z / 3.0 + z**3 / 30.0 - z**5 / 840.0

    return (2.0 / cutoff) ** 0.5 * freq**2 * torch.where(small, series, direct)


def fourier(dist: Tensor, num_features: int, cutoff: float) -> Tensor:
    """Sinusoidal Fourier features over ``num_features`` octave-spaced frequency bands.

    :param dist: True L2 distances (..., 1), non-negative.
    :param num_features: Number of frequency bands; the output width is ``2 * num_features``.
    :param cutoff: Length scale; the lowest band has wavelength ~cutoff.
    :return: Concatenated sin/cos features (..., 2 * num_features)."""

    bands = 2.0 ** torch.arange(num_features, device=dist.device, dtype=dist.dtype)
    scaled = dist * (torch.pi / cutoff) * bands
    return torch.cat([scaled.sin(), scaled.cos()], dim=-1)


def fourier_derivative(dist: Tensor, num_features: int, cutoff: float) -> Tensor:
    """Derivative of :func:`fourier` with respect to the distance.

    :param dist: True L2 distances (..., 1), non-negative.
    :param num_features: Number of frequency bands.
    :param cutoff: Length scale of the lowest band.
    :return: Feature derivative (..., 2 * num_features), in the same sin/cos order."""

    bands = 2.0 ** torch.arange(num_features, device=dist.device, dtype=dist.dtype)
    rate = (torch.pi / cutoff) * bands
    scaled = dist * rate
    return torch.cat([rate * scaled.cos(), -rate * scaled.sin()], dim=-1)


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


def gaussian_derivative(dist: Tensor, num_features: int, cutoff: float) -> Tensor:
    """Derivative of :func:`gaussian` with respect to the distance.

    :param dist: True L2 distances (..., 1), non-negative.
    :param num_features: Number of Gaussian centers.
    :param cutoff: Position of the last center.
    :return: Basis derivative (..., num_features)."""

    centers = torch.linspace(
        0.0, cutoff, num_features, device=dist.device, dtype=dist.dtype
    )
    spacing = cutoff / max(num_features - 1, 1)
    return -(dist - centers) / spacing**2 * gaussian(dist, num_features, cutoff)


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


def encode_distance_derivative(
    dist: Tensor, kind: Encoding, num_features: int, cutoff: float
) -> Tensor:
    """Dispatch to the derivative of the requested radial basis.

    This is what makes a velocity field built as a linear combination of the basis carry a
    closed-form divergence: the field's only position dependence is through ``dist``, so the
    chain rule needs nothing but ``d(basis)/d(dist)``.

    :param dist: True L2 distances (..., 1), non-negative.
    :param kind: Which encoding to differentiate.
    :param num_features: Number of basis functions / frequency bands.
    :param cutoff: Radial length scale.
    :return: Basis derivative (..., ``encoding_width(kind, num_features)``)."""

    match kind:
        case "bessel":
            return bessel_derivative(dist, num_features, cutoff)
        case "fourier":
            return fourier_derivative(dist, num_features, cutoff)
        case "gaussian":
            return gaussian_derivative(dist, num_features, cutoff)
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


def polynomial_envelope_derivative(
    dist: Tensor, cutoff: float, exponent: int = 6
) -> Tensor:
    """Derivative of :func:`polynomial_envelope` with respect to the distance.

    The three polynomial terms collapse to ``-p(p+1)(p+2)/2 * u^(p-1) * (1-u)^2 / cutoff``,
    which is manifestly zero at both ends of the support.

    :param dist: True L2 distances (..., 1), non-negative.
    :param cutoff: Cutoff radius; the derivative is 0 for d >= cutoff.
    :param exponent: Polynomial exponent p, matching the envelope it differentiates.
    :return: Envelope derivative (..., 1)."""

    p = exponent
    u = dist / cutoff
    grad = -(p * (p + 1) * (p + 2) / 2) * u ** (p - 1) * (1.0 - u) ** 2 / cutoff
    return torch.where(dist < cutoff, grad, torch.zeros_like(grad))


def cosine_envelope(dist: Tensor, cutoff: float) -> Tensor:
    """Cosine (Behler) cutoff envelope, 0.5 * (cos(pi * d / cutoff) + 1) below the cutoff.

    :param dist: True L2 distances (..., 1), non-negative.
    :param cutoff: Cutoff radius; the envelope is 0 for d >= cutoff.
    :return: Envelope weights in [0, 1] (..., 1)."""

    env = 0.5 * (torch.cos(torch.pi * dist / cutoff) + 1.0)
    return torch.where(dist < cutoff, env, torch.zeros_like(env))
