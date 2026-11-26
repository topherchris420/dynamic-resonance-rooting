import numpy as np
import pytest

from drr_framework.mellin import (
    EULER_GAMMA,
    benchmark,
    fit_log_mode,
    gamma_window_mass,
    log_spectrum,
    smoothed_prime_sum,
    window_excess,
)


def test_smoothed_prime_sum_direct():
    x = np.array([1.0, 2.0, 3.5, 8.0, 10.5])
    n = np.array([2, 3, 4, 5, 7, 8, 9])
    weights = np.log([2, 3, 2, 5, 7, 2, 3])
    kernel = np.maximum(x[:, None] / n - 1, 0)
    assert smoothed_prime_sum(x) == pytest.approx(
        kernel @ weights - x * np.log(x) + (1 + EULER_GAMMA) * x
    )
    chi = np.array([0, 1, -1])
    assert smoothed_prime_sum(x, chi, 0.25) == pytest.approx(
        kernel @ (weights * chi[n % 3]) + 0.25 * x
    )


def test_arithmetic_benchmark():
    result = benchmark()
    assert result["samples"] == 991
    errors = [r["normalized_rmse"] for r in result["cases"]]
    assert errors == pytest.approx([0.00820000, 0.00485025, 0.00236244, 0.00115507], abs=1e-7)


def test_log_spectrum_and_mode_recovery():
    u = np.arange(256) * (2 * np.pi / 256)
    x = np.exp(u)
    y = np.sqrt(x) * np.cos(14 * u + 0.4)
    frequency, amplitude = log_spectrum(x, y)
    assert frequency[np.argmax(amplitude)] == pytest.approx(14)
    assert amplitude.max() == pytest.approx(1)
    result = fit_log_mode(x, y, [0.4, 0.5, 0.6], [13, 14, 15])
    assert result["beta"] == 0.5
    assert result["omega"] == 14
    assert result["relative_sse"] < 1e-20


@pytest.mark.parametrize("n", [64, 67], ids=["even-power-of-two", "odd-prime"])
def test_log_spectrum_matches_direct_dft_in_every_bin(n):
    du = 0.125
    beta = 0.3
    samples = np.arange(n)
    x = np.exp(0.75 + du * samples)
    # DC, an off-bin tone, and nonzero energy in the highest positive bin.
    weighted = (
        0.8
        + 0.35 * np.sin(2 * np.pi * 4.25 * samples / n + 0.1)
        + 0.6 * np.cos(2 * np.pi * (n // 2) * samples / n)
    )
    frequency, amplitude = log_spectrum(x, x**beta * weighted, beta=beta)

    bins = np.arange(n // 2 + 1)
    basis = np.exp(-2j * np.pi * bins[:, None] * samples / n)
    two_sided = np.abs(basis @ weighted) / n
    # Real inputs have conjugate pairs, except for DC and the even Nyquist bin.
    multiplicity = np.where((bins == 0) | (2 * bins == n), 1, 2)
    np.testing.assert_allclose(frequency, 2 * np.pi * bins / (n * du), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(amplitude, multiplicity * two_sided, rtol=1e-12, atol=1e-12)


def test_window_and_kernel():
    result = window_excess([1, 4, 9], [-10, 3, 9], 0.5, 2)
    assert result["max_normalized"] == pytest.approx(3)
    assert result["violations"] == 1
    assert result["sampled_only"] is True
    assert gamma_window_mass(0, 2, 1, 2) == pytest.approx(np.exp(-1) - np.exp(-2))
    assert gamma_window_mass(20, 3, 0, 100) == pytest.approx(1)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        smoothed_prime_sum([1_000_001])
    with pytest.raises(ValueError):
        smoothed_prime_sum([2], [0, 1, -1])
    with pytest.raises(ValueError):
        log_spectrum([1, 2, 3, 4], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        gamma_window_mass(2, 1, 0, 2)
    with pytest.raises(ValueError):
        fit_log_mode([1, 2, 3, 4, 5], np.zeros(5), [0.5], [14])
