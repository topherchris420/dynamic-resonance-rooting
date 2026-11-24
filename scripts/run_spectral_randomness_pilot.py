"""Exploratory matched-spectrum phase-closure experiment (NumPy only)."""

import json

import numpy as np


def run():
    rng = np.random.default_rng(20261124)
    trials, blocks, n, surrogates = 300, 32, 128, 199
    bins = [7, 11, 18]
    rows = []
    max_error = 0.0
    for sigma in (None, 0.0, 0.5, 1.0, 2.0):
        # Independent circular Gaussian blocks; the alternative adds a triad.
        x = rng.standard_normal((trials, blocks, n))
        spectrum = np.fft.rfft(x, axis=-1)
        if sigma is not None:
            a, b = rng.uniform(-np.pi, np.pi, (2, trials, blocks))
            c = a + b + rng.normal(0, sigma, (trials, blocks))
            spectrum[..., bins] += (n / 4) * np.exp(1j * np.stack([a, b, c], axis=-1))
        phase = np.angle(spectrum[..., bins])
        closure = phase[..., 0] + phase[..., 1] - phase[..., 2]
        observed = np.abs(np.mean(np.exp(1j * closure), axis=-1))
        # Each null closure is uniform conditional on all observed magnitudes.
        null_phase = rng.uniform(-np.pi, np.pi, (trials, surrogates, blocks))
        null = np.abs(np.mean(np.exp(1j * null_phase), axis=-1))
        p = (1 + np.sum(null >= observed[:, None], axis=1)) / (surrogates + 1)
        # Demonstrate an exact magnitude-matched surrogate in the time domain.
        matched = spectrum.copy()
        matched[..., 1:-1] = np.abs(matched[..., 1:-1]) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, matched[..., 1:-1].shape)
        )
        original = np.fft.rfft(np.fft.irfft(spectrum, n=n, axis=-1), axis=-1)
        replay = np.fft.rfft(np.fft.irfft(matched, n=n, axis=-1), axis=-1)
        error = np.max(np.abs(np.abs(original) - np.abs(replay)))
        max_error = max(max_error, float(error))
        rejected = int(np.sum(p <= 0.05))
        rows.append(
            {
                "phase_jitter_rad": sigma,
                "rejections": rejected,
                "rate": rejected / trials,
                "mean_closure": float(observed.mean()),
            }
        )
    return {
        "seed": 20261124,
        "trials_per_case": trials,
        "blocks": blocks,
        "samples_per_block": n,
        "surrogates": surrogates,
        "bins": bins,
        "alpha": 0.05,
        "max_magnitude_error": max_error,
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
