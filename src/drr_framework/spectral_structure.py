"""Phase-sensitive diagnostics with an explicit independent-phase null."""

import numpy as np


def phase_closure(blocks: np.ndarray, triads: np.ndarray, surrogates=199, seed=0) -> dict:
    """Test fixed sum-frequency triads with shared-phase max-statistic correction.

    Rows must be independent, equally sampled real blocks. The null assumes
    independent uniform interior Fourier phases, conditional on magnitudes.
    Nonstationarity, overlapping blocks, leakage and harmonics can invalidate
    interpretation. This is not a causality or chaos detector.
    """
    blocks = np.asarray(blocks)
    triads = np.asarray(triads)
    if (
        blocks.ndim != 2
        or blocks.shape[0] < 2
        or blocks.shape[1] < 8
        or not np.isrealobj(blocks)
        or not np.isfinite(blocks).all()
    ):
        raise ValueError("provide at least two finite real blocks of length >= 8")
    if (
        triads.ndim != 2
        or triads.shape[1] != 3
        or len(triads) == 0
        or not np.issubdtype(triads.dtype, np.integer)
        or np.any(triads <= 0)
        or np.any(triads > (blocks.shape[1] - 1) // 2)
    ):
        raise ValueError("triads must be positive interior integer bins with a+b=c")
    # Bounds above ensure the cast and sum fit in the native index dtype.
    triads = triads.astype(np.intp, copy=False)
    if np.any(triads[:, 0] + triads[:, 1] != triads[:, 2]):
        raise ValueError("triads must be positive interior integer bins with a+b=c")
    if not isinstance(surrogates, (int, np.integer)) or surrogates < 19:
        raise ValueError("at least 19 surrogates required")
    spectrum = np.fft.rfft(blocks, axis=1)
    bins, indices = np.unique(triads, return_inverse=True)
    indices = indices.reshape(triads.shape)
    selected = spectrum[:, bins]
    scale = np.max(np.abs(spectrum), axis=1, keepdims=True)
    if np.any(np.abs(selected) <= np.finfo(float).eps * scale):
        raise ValueError("phase undefined at a numerically zero selected coefficient")

    def statistic(phase):
        closure = phase[:, indices[:, 0]] + phase[:, indices[:, 1]] - phase[:, indices[:, 2]]
        return np.abs(np.mean(np.exp(1j * closure), axis=0))

    observed = statistic(np.angle(selected))
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(len(triads), dtype=int)
    for _ in range(surrogates):
        # Shared phases preserve dependence between overlapping triads.
        null = statistic(rng.uniform(-np.pi, np.pi, selected.shape))
        exceedances += null.max() >= observed
    return {
        "triads": triads.tolist(),
        "closure": observed.tolist(),
        "adjusted_p": ((exceedances + 1) / (surrogates + 1)).tolist(),
        "surrogates": int(surrogates),
        "seed": seed,
        "null": "independent uniform interior Fourier phases",
    }
