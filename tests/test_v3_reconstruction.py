"""Tests for the reconstructed v3 operators and study plumbing."""

import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework.calibration import simulate_ar1, simulate_directed_pair
from drr_framework.v3_reconstruction import (
    ar2_coefficients,
    ar2_variance,
    build_tasks,
    gaussian_surrogate,
    granger_pvalue,
    holm,
    lambda_statistic,
    lambda_test,
    monotone_fit,
    monte_carlo_p,
    periodogram,
    prewhiten,
    rooting_size_row,
    simulate_ar2,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "v3_reconstruction" / "protocol.json"
ARTIFACT = ROOT / "results" / "expected" / "v3_reconstruction" / "v3_reconstruction.json"


def test_monotone_fit_is_monotone_and_picks_better_direction():
    rng = np.random.default_rng(0)
    down = np.linspace(5, 1, 50) + 0.1 * rng.standard_normal(50)
    fit = monotone_fit(down)
    assert np.all(np.diff(fit) <= 1e-12)
    up_fit = monotone_fit(down[::-1])
    assert np.all(np.diff(up_fit) >= -1e-12)
    assert np.isclose(fit.mean(), down.mean())


def test_ar2_scaling_gives_unit_variance():
    a1, a2 = ar2_coefficients(0.9, 0.0625)
    assert ar2_variance(a1, a2) > 1
    x = simulate_ar2(200_000, 1, 0.9, np.random.default_rng(1))
    assert abs(x.var() - 1.0) < 0.05


def test_prewhitening_recovers_ar1_and_whitens():
    x = simulate_ar1(4000, 2, 0.9, np.random.default_rng(2))
    innovations, orders = prewhiten(x, 10)
    assert innovations.shape == (3990, 2)
    assert all(1 <= p <= 3 for p in orders)
    z = innovations[:, 0] - innovations[:, 0].mean()
    assert abs(np.dot(z[:-1], z[1:]) / np.dot(z, z)) < 0.05


def test_prewhitening_removes_reverse_lead():
    data = simulate_directed_pair(100_000, 0.3, 2, 0.9, np.random.default_rng(3))
    raw = np.corrcoef(data[:-1, 1], data[1:, 0])[0, 1]
    innovations, _ = prewhiten(data[:, :2], 10)
    white = np.corrcoef(innovations[:-1, 1], innovations[1:, 0])[0, 1]
    assert abs(raw - 0.3 * 0.9**3) < 0.02
    assert abs(white) < 0.01


def test_granger_detects_true_direction_only():
    data = simulate_directed_pair(512, 0.5, 2, 0.5, np.random.default_rng(4))
    p_forward, lag = granger_pvalue(data[:, 0], data[:, 1])
    p_reverse, _ = granger_pvalue(data[:, 1], data[:, 0])
    assert p_forward < 1e-6 and lag == 2
    assert p_reverse > 1e-3


def test_surrogate_periodogram_has_requested_mean():
    spectrum = np.linspace(4.0, 1.0, 255)
    rng = np.random.default_rng(5)
    mean = np.mean([periodogram(gaussian_surrogate(spectrum, 512, rng)) for _ in range(400)], 0)
    assert np.allclose(mean, spectrum, rtol=0.25)


def test_lambda_finds_line_in_red_noise():
    rng = np.random.default_rng(6)
    background = simulate_ar1(512, 1, 0.9, rng)[:, 0] * np.sqrt(1 - 0.81)
    x = background + 0.75 * np.sin(2 * np.pi * 0.0625 * np.arange(512) + 0.3)
    _, _, ordinates, s_hat = lambda_statistic(x)
    ratio = ordinates / s_hat
    # The line bin stands far above its isotonic continuum. (With this seed the
    # last Fourier bin is the argmax, a boundary effect the study measures.)
    assert ratio[31] > 20
    assert lambda_test(x, 99, rng)["p_value"] <= 0.05


def test_monte_carlo_p_and_holm():
    assert monte_carlo_p(1.0, [0.0] * 99) == 0.01
    assert monte_carlo_p(0.0, [1.0] * 99) == 1.0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_v2_column_reproduces_committed_size_prefix():
    committed = json.loads(
        (ROOT / "results" / "expected" / "rooting_calibration_study.json").read_text()
    )
    row = next(
        r
        for r in committed["size"]
        if r["ar_coefficient"] == 0.0 and r["surrogate_method"] == "circular_shift"
    )
    shared = dict(n_samples=512, max_lag=4, n_surrogates=99, alpha=0.05, n_channels=3)
    out = rooting_size_row({"kind": "ar1", "phi": 0.0}, 20260925 + 10_000, 40, shared)
    want = [t for t in row["false_positive_trials"] if t < 40]
    assert out["false_positive_trials"]["v2"] == want


def test_task_list_covers_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    sections = [t[0] for t in build_tasks(protocol)]
    assert sections.count("size") == 5
    assert sections.count("memory") == 6
    assert sections.count("power_curve") == 5
    assert sections.count("long_run") == 3
    assert sections.count("resonance") == 16
    assert sections.count("qbo") == 1


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
def test_committed_artifact_is_consistent():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["smoke"] is False
    assert {f["id"] for f in artifact["findings"]} == {f"F{i}" for i in range(1, 13)}
    assert all(check["match"] for check in artifact["exact_reproductions"])
    protocol_text = json.dumps(json.loads(PROTOCOL.read_text()), sort_keys=True)
    import hashlib

    assert artifact["protocol_sha256"] == hashlib.sha256(protocol_text.encode()).hexdigest()
