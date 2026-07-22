import torch

from egnn_mol import (
    bessel,
    cosine_envelope,
    encode_distance,
    encoding_width,
    fourier,
    gaussian,
    polynomial_envelope,
)


def test_bessel_matches_reference_formula():
    """Bessel encoding equals the DimeNet basis sqrt(2/c) * sin(n*pi*d/c) / d."""
    cutoff, num = 10.0, 8
    dist = torch.linspace(0.1, 9.0, 50).unsqueeze(-1)
    got = bessel(dist, num, cutoff)
    freq = torch.arange(1, num + 1) * torch.pi / cutoff
    ref = (2.0 / cutoff) ** 0.5 * torch.sin(freq * dist) / dist
    assert torch.allclose(got, ref, atol=1e-5)


def test_bessel_finite_at_zero():
    """The sinc form stays finite at d = 0 (no division blow-up)."""
    enc = bessel(torch.zeros(3, 1), num_features=6, cutoff=5.0)
    assert torch.isfinite(enc).all()


def test_encoding_widths():
    dist = torch.rand(7, 1)
    assert encode_distance(dist, "bessel", 8, 5.0).shape[-1] == encoding_width("bessel", 8) == 8
    assert encode_distance(dist, "fourier", 8, 5.0).shape[-1] == encoding_width("fourier", 8) == 16
    assert encode_distance(dist, "gaussian", 8, 5.0).shape[-1] == encoding_width("gaussian", 8) == 8


def test_gaussian_and_fourier_finite():
    dist = torch.linspace(0.0, 5.0, 20).unsqueeze(-1)
    assert torch.isfinite(gaussian(dist, 16, 5.0)).all()
    assert torch.isfinite(fourier(dist, 4, 5.0)).all()


def test_envelopes_bounded_and_zero_beyond_cutoff():
    cutoff = 5.0
    dist = torch.linspace(0.0, 8.0, 40).unsqueeze(-1)
    for env in (polynomial_envelope(dist, cutoff), cosine_envelope(dist, cutoff)):
        assert (env >= -1e-6).all() and (env <= 1.0 + 1e-6).all()
        assert torch.allclose(env[dist.squeeze(-1) >= cutoff], torch.zeros(()), atol=1e-6)


def test_unknown_encoding_raises():
    try:
        encode_distance(torch.rand(2, 1), "nope", 4, 5.0)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown encoding")
