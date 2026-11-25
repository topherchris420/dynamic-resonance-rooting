import numpy as np
import pytest

from drr_framework.arithmetic_spectrum import benchmark, chebyshev_table, reconstruct
from drr_framework.spectral_structure import phase_closure


def test_chebyshev_prime_powers():
    psi = chebyshev_table(10)
    assert psi[1] == 0
    assert psi[4] - psi[3] == pytest.approx(np.log(2))
    assert psi[8] - psi[7] == pytest.approx(np.log(2))
    assert psi[9] - psi[8] == pytest.approx(np.log(3))
    assert psi[10] == psi[9]


def test_arithmetic_reconstruction():
    report = benchmark()
    errors = [case["normalized_rmse"] for case in report["cases"]]
    assert report["samples"] == 991
    assert errors == pytest.approx([0.200539, 0.178484, 0.149980, 0.125609], abs=1e-6)


@pytest.mark.parametrize("x,count", [([1.0], 1), ([np.nan], 1), ([2.0], 11)])
def test_invalid_reconstruction(x, count):
    with pytest.raises(ValueError):
        reconstruct(x, count)


def test_phase_closure_and_shift_invariance():
    rng = np.random.default_rng(19)
    a, b = rng.uniform(-np.pi, np.pi, (2, 32))
    spec = np.fft.rfft(rng.normal(size=(32, 128)), axis=1)
    spec[:, [7, 11, 18]] = 32 * np.exp(1j * np.stack([a, b, a + b], axis=1))
    blocks = np.fft.irfft(spec, n=128, axis=1)
    triads = np.array([[7, 11, 18], [7, 9, 16]])
    result = phase_closure(blocks, triads, seed=20)
    shifted = np.array([np.roll(row, i) for i, row in enumerate(blocks)])
    replay = phase_closure(shifted, triads, seed=20)
    assert result["closure"][0] == pytest.approx(1)
    assert result["adjusted_p"][0] == 1 / 200
    assert result["closure"] == pytest.approx(replay["closure"])
    assert result["adjusted_p"] == replay["adjusted_p"]
    single = phase_closure(blocks, triads[:1], seed=20)
    assert single["closure"][0] == pytest.approx(result["closure"][0])


@pytest.mark.parametrize("triads", [[[1, 2, 4]], [[0, 2, 2]], [[1, 3, 4]], [[1.0, 1.0, 2.0]]])
def test_invalid_triads(triads):
    with pytest.raises(ValueError):
        phase_closure(np.ones((4, 8)), np.array(triads))


def test_undefined_phase_rejected():
    with pytest.raises(ValueError, match="phase undefined"):
        phase_closure(np.zeros((4, 16)), np.array([[1, 2, 3]]))


@pytest.mark.parametrize(
    "dtype,n,triad",
    [(np.uint8, 512, [130, 130, 4]), (np.uint16, 131072, [32770, 32770, 4])],
)
def test_wrapped_triads_rejected(dtype, n, triad):
    blocks = np.random.default_rng(13).normal(size=(4, n))
    with pytest.raises(ValueError, match="a\\+b=c"):
        phase_closure(blocks, np.array([triad], dtype=dtype))


def test_valid_unsigned_triads():
    blocks = np.random.default_rng(13).normal(size=(8, 512))
    triads = np.array([[100, 130, 230]])
    assert phase_closure(blocks, triads) == phase_closure(blocks, triads.astype(np.uint8))
