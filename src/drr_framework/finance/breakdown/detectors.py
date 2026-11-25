"""Causal fingerprints and conventional sequential comparators."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

from ...modules import DepthCalculator, RootingAnalyzer


def fingerprint(window):
    """The existing DRR operator, including its sign-insensitive rooting scores."""
    root = RootingAnalyzer().analyze(window, max_lag=3, n_surrogates=0)
    rooting = [root["score_matrix"][a, b] for a, b in [(0, 1), (1, 0)]]
    rooting += [root["effective_lag"][a, b] / 3 for a, b in [(0, 1), (1, 0)]]
    spectral = []
    for j in range(2):
        result = DepthCalculator().calculate(window[:, j], window_size=32, sampling_rate=1)
        spectral += [
            result["resonance_depth"],
            *result["components"].values(),
            result["target_frequency_hz"],
        ]
    signed = []
    for a, b in [(0, 1), (1, 0)]:
        for lag in [1, 2, 3]:
            x, y = window[:-lag, a], window[lag:, b]
            denominator = np.std(x) * np.std(y)
            signed.append(
                float(np.mean((x - x.mean()) * (y - y.mean())) / denominator)
                if denominator > 1e-12
                else 0.0
            )
    return np.array(rooting + spectral), np.array(signed)


def var_errors(data, fit=256):
    """One-step VAR(3) and zero-return errors in fit-only standardized units."""
    x = np.asarray(data, float)
    scale = x[:fit].std(axis=0).clip(1e-8)
    z = (x - x[:fit].mean(axis=0)) / scale
    design = np.column_stack([np.ones(len(z) - 3), z[2:-1], z[1:-2], z[:-3]])
    penalty = np.eye(7) * 0.001
    penalty[0, 0] = 0
    train = design[: fit - 3]
    beta = np.linalg.solve(train.T @ train + penalty, train.T @ z[3:fit])
    errors = np.full_like(x, np.nan)
    errors[3:] = z[3:] - design @ beta
    # Zero return is zero in original coordinates, not the demeaned zero vector.
    zero = x / scale
    variance = np.mean(errors[3:fit] ** 2, axis=0).clip(1e-8)
    return errors, zero, variance


def bocpd(values, hazard=1 / 128):
    """Exact run-length recursion; use short-run mass, not constant reset mass."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 < hazard < 1:
        raise ValueError("Invalid BOCPD observations or hazard")
    probability = np.ones(1)
    mu, kappa, alpha, beta = (np.array([v], float) for v in [0, 1, 3, 2])
    output = []
    for value in values:
        df = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        log_pdf = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi * scale2)
            - (df + 1) / 2 * np.log1p((value - mu) ** 2 / (df * scale2))
        )
        joint = np.log(probability.clip(np.finfo(float).tiny)) + log_pdf
        evidence = logsumexp(joint)
        probability = np.r_[hazard, np.exp(joint - evidence) * (1 - hazard)]
        output.append(float(probability[:9].sum()))
        beta = np.r_[2, beta + kappa * (value - mu) ** 2 / (2 * (kappa + 1))]
        mu = np.r_[0, (kappa * mu + value) / (kappa + 1)]
        kappa = np.r_[1, kappa + 1]
        alpha = np.r_[3, alpha + 0.5]
    return np.asarray(output)


def score_stream(data, protocol):
    x = np.asarray(data, float)
    spec = protocol["detector"]
    fit, window, stride = spec["fit"], spec["window"], spec["stride"]
    if x.ndim != 2 or x.shape[1] != 2 or len(x) <= fit or not np.isfinite(x).all():
        raise ValueError("Expected finite two-channel observations longer than fit period")
    times = np.arange(window - 1, len(x), stride)
    drr, signed = zip(*(fingerprint(x[t - window + 1 : t + 1]) for t in times))
    drr, signed = np.asarray(drr), np.asarray(signed)
    features = {
        "drr": drr,
        "rooting_only": drr[:, :4],
        "spectral_only": drr[:, 4:],
        "signed_correlation": signed,
    }
    scores = {}
    for name, values in features.items():
        reference = values[times < fit]
        scores[name] = np.sqrt(
            np.mean(
                ((values - reference.mean(axis=0)) / reference.std(axis=0, ddof=1).clip(0.05)) ** 2,
                axis=1,
            )
        )
    errors, zero, variance = var_errors(x, fit)
    energy = np.mean(errors**2 / variance, axis=1)
    acc, cusum = 0.0, []
    for value in energy[fit:]:
        acc = max(0.0, acc + value - 1.5)
        cusum.append(acc)
    log_reference = np.log(energy[3:fit].clip(1e-12))
    # Initialize run-length state on fit residuals so the first eight monitored
    # observations do not trivially have probability one of a short run.
    log_energy = (np.log(energy[3:].clip(1e-12)) - log_reference.mean()) / max(
        log_reference.std(ddof=1), 0.05
    )
    posterior = bocpd(log_energy)
    selected = times >= fit
    scores = {name: values[selected] for name, values in scores.items()}
    times = times[selected]
    scores["var_cusum"] = np.asarray(cusum)[times - fit]
    scores["var_bocpd"] = posterior[times - 3]
    return times, scores, errors, zero


def simulate(protocol, seed, scenario):
    spec = protocol["simulation"]
    if scenario not in spec["nulls"] + spec["changes"] + spec["challenges"]:
        raise ValueError("Unknown scenario")
    n, change = spec["length"], spec["change"]
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(5, size=(n, 3)) / np.sqrt(5 / 3)
    x, y = np.zeros(n), np.zeros(n)
    for t in range(3, n):
        coefficient, lag = 0.7, 1
        if t >= change:
            if scenario == "drop":
                coefficient = 0.0
            elif scenario == "reverse":
                coefficient = -0.7
            elif scenario == "lag_shift":
                lag = 3
            elif scenario == "gradual_drop":
                coefficient *= max(0, 1 - (t - change + 1) / 64)
        x[t] = 0.4 * x[t - 1] + noise[t, 0]
        y[t] = 0.2 * y[t - 1] + coefficient * x[t - lag] + noise[t, 1]
    result = np.column_stack([x, y])
    if scenario == "volatility_only":
        result[change:] *= 3
    if scenario == "common_driver":
        result[change:] += 2 * noise[change:, 2, None]
    return result


def threshold(maxima, budget=0.05):
    values = np.asarray(maxima, float)
    rank = int(np.ceil((len(values) + 1) * (1 - budget)))
    if not np.isfinite(values).all() or not 1 <= rank <= len(values):
        raise ValueError("Insufficient or invalid calibration scores")
    return float(np.sort(values)[rank - 1])


def first_alarm(times, scores, cutoff, change=512, length=768):
    crossings = times[np.asarray(scores) > cutoff]
    first = int(crossings[0]) if len(crossings) else None
    early = first is not None and first < change
    hit = first is not None and not early
    return {
        "first_alarm": first,
        "pre_change_alarm": early,
        "hit": hit,
        "delay": first - change if hit else None,
        "penalized_delay": first - change if hit else length - change,
    }
