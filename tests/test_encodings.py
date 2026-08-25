import pytest
import torch

from egnn_mol import (
    bessel,
    bessel_derivative,
    cosine_envelope,
    encode_distance,
    encode_distance_derivative,
    encoding_width,
    fourier,
    fourier_derivative,
    gaussian,
    gaussian_derivative,
    polynomial_envelope,
    polynomial_envelope_derivative,
)

ENCODINGS = ("bessel", "fourier", "gaussian")


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


class TestDerivatives:
    """The analytic basis derivatives, which is what makes a closed-form divergence possible."""

    @pytest.mark.parametrize("kind", ENCODINGS)
    @pytest.mark.parametrize("cutoff,num", [(10.0, 8), (1.0, 16), (0.5, 32)])
    def test_matches_autograd(self, kind, cutoff, num):
        """Each derivative equals autograd of its own value function, in double precision.

        The lower end of the range is 1e-3 * cutoff, not 0: `torch.sinc`'s own backward suffers
        the same small-argument cancellation the bessel derivative guards against, so below that
        it is the *reference* that loses precision, not the analytic form (which is checked
        against its closed-form leading term separately).

        :param kind: Encoding under test.
        :param cutoff: Radial length scale.
        :param num: Number of basis functions."""
        dist = torch.linspace(
            1e-3 * cutoff, 2 * cutoff, 400, dtype=torch.float64
        ).unsqueeze(-1)
        dist.requires_grad_(True)

        value = encode_distance(dist, kind, num, cutoff)
        reference = torch.autograd.grad(value.sum(), dist)[0]
        got = encode_distance_derivative(dist, kind, num, cutoff).sum(-1, keepdim=True)

        assert torch.allclose(got, reference, rtol=1e-8, atol=1e-8)

    def test_bessel_small_argument_branch(self):
        """Below z ~ 0.1 the Taylor branch must reproduce the exact leading term.

        Evaluated directly, ``z cos z - sin z`` is a difference of two O(z) terms whose result is
        O(z^3/3), so float32 keeps nothing of it -- this is where a silent bug would live."""
        cutoff, num = 0.5, 32
        dist = torch.tensor([[1e-8], [1e-6], [1e-4]], dtype=torch.float64)
        freq = torch.arange(1, num + 1, dtype=torch.float64) * torch.pi / cutoff

        # d/dd [sqrt(2/c) sin(freq d)/d] -> -sqrt(2/c) freq^3 d / 3 as d -> 0.
        leading = -((2.0 / cutoff) ** 0.5) * freq**3 * dist / 3.0
        got = bessel_derivative(dist, num, cutoff)

        assert torch.allclose(got, leading, rtol=1e-4)

    def test_bessel_derivative_finite_in_float32(self):
        """The small-argument branch must not produce NaN or inf in single precision."""
        got = bessel_derivative(torch.zeros(3, 1), num_features=6, cutoff=5.0)

        assert torch.isfinite(got).all()

    def test_derivative_widths_match_the_encodings(self):
        """A derivative has the same width as the basis it differentiates."""
        dist = torch.rand(7, 1)
        for kind in ENCODINGS:
            width = encoding_width(kind, 8)
            assert encode_distance_derivative(dist, kind, 8, 5.0).shape[-1] == width

    @pytest.mark.parametrize("exponent", [4, 6, 8])
    def test_polynomial_envelope_derivative(self, exponent):
        """The envelope derivative equals autograd of the envelope and vanishes at both ends.

        :param exponent: Polynomial exponent under test."""
        cutoff = 5.0
        dist = torch.linspace(0.0, 1.5 * cutoff, 200, dtype=torch.float64).unsqueeze(-1)
        dist.requires_grad_(True)

        reference = torch.autograd.grad(
            polynomial_envelope(dist, cutoff, exponent).sum(), dist
        )[0]
        got = polynomial_envelope_derivative(dist, cutoff, exponent)

        assert torch.allclose(got, reference, rtol=1e-10, atol=1e-12)
        assert torch.allclose(got[0], torch.zeros_like(got[0]))
        beyond = got[dist.detach().squeeze(-1) >= cutoff]
        assert torch.allclose(beyond, torch.zeros_like(beyond))

    def test_unknown_derivative_raises(self):
        """An unknown encoding name raises, as the value dispatcher does."""
        with pytest.raises(ValueError, match="unknown distance encoding"):
            encode_distance_derivative(torch.rand(2, 1), "nope", 4, 5.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value,derivative",
        [(bessel, bessel_derivative), (fourier, fourier_derivative), (gaussian, gaussian_derivative)],
    )
    def test_per_basis_functions_match_the_dispatcher(self, value, derivative):
        """The named functions and the dispatcher agree, so neither can drift alone.

        :param value: Value function.
        :param derivative: Its derivative."""
        dist = torch.linspace(0.1, 4.0, 20).unsqueeze(-1)
        kind = value.__name__

        assert torch.allclose(encode_distance(dist, kind, 6, 5.0), value(dist, 6, 5.0))
        assert torch.allclose(
            encode_distance_derivative(dist, kind, 6, 5.0), derivative(dist, 6, 5.0)
        )
