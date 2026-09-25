"""The landing page's JavaScript engine computes what the Python package computes.

The page at ``index.html`` runs ``assets/web-demo/drr-engine.js`` live in the
browser. These tests execute that file under Node and compare every
deterministic output with :mod:`drr_framework.modules`. They are skipped when
Node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from drr_framework._spectral import hilbert, welch
from drr_framework.modules import DepthCalculator, RootingAnalyzer

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "assets" / "web-demo" / "drr-engine.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def _run_node(script: str, payload: dict) -> dict:
    program = (
        f"const DRR = require({json.dumps(str(ENGINE))});\n"
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        f"{script}\n"
    )
    completed = subprocess.run(
        [NODE, "-e", program],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def _signals() -> dict:
    rng = np.random.default_rng(2026)
    t = np.arange(1024)
    switch = np.where(t < 512, np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.15 * t))
    return {
        "off_bin_tone": np.sin(2 * np.pi * 0.0671 * t) + 0.2 * rng.standard_normal(1024),
        "frequency_switch": switch + 0.05 * rng.standard_normal(1024),
        "white_noise": rng.standard_normal(1024),
        "red_noise": np.cumsum(rng.standard_normal(1024)) * 0.1 + rng.standard_normal(1024),
        "short_odd_length": np.sin(2 * np.pi * 0.11 * np.arange(301)) + rng.standard_normal(301),
    }


@pytest.mark.parametrize("sampling_rate", [200.0, 1.0])
def test_depth_matches_python_for_every_component(sampling_rate):
    signals = _signals()
    windows = {name: (256 if len(x) >= 1024 else 128) for name, x in signals.items()}
    js = _run_node(
        "const out = {};\n"
        "for (const [name, series] of Object.entries(input.signals)) {\n"
        "  out[name] = DRR.depth(series, input.windows[name], input.fs, null);\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));",
        {
            "signals": {name: x.tolist() for name, x in signals.items()},
            "windows": windows,
            "fs": sampling_rate,
        },
    )
    calculator = DepthCalculator()
    for name, series in signals.items():
        python = calculator.calculate(series, windows[name], sampling_rate=sampling_rate)
        assert js[name]["method"] == python["method"]
        assert js[name]["target_frequency"] == pytest.approx(
            python["target_frequency_hz"], rel=1e-9, abs=1e-12
        ), name
        for component, value in python["components"].items():
            assert js[name]["components"][component] == pytest.approx(value, abs=1e-9), (
                name,
                component,
            )
        assert js[name]["resonance_depth"] == pytest.approx(python["resonance_depth"], abs=1e-9)


def test_welch_and_hilbert_match_scipy():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(700)
    js = _run_node(
        "const w = DRR.welch(input.x, 50, 128);\n"
        "const h = DRR.hilbert(input.x.slice(0, 256));\n"
        "const g = DRR.hilbert(input.x.slice(0, 255));\n"
        "process.stdout.write(JSON.stringify({freqs: Array.from(w.freqs),\n"
        "  psd: Array.from(w.psd), hre: Array.from(h.re), him: Array.from(h.im),\n"
        "  gre: Array.from(g.re), gim: Array.from(g.im)}));",
        {"x": x.tolist()},
    )
    freqs, psd = welch(x, fs=50.0, nperseg=128)
    np.testing.assert_allclose(js["freqs"], freqs, rtol=1e-12)
    np.testing.assert_allclose(js["psd"], psd, rtol=1e-9, atol=1e-15)
    for key, n in (("h", 256), ("g", 255)):
        analytic = hilbert(x[:n])
        np.testing.assert_allclose(js[f"{key}re"], analytic.real, atol=1e-9)
        np.testing.assert_allclose(js[f"{key}im"], analytic.imag, atol=1e-9)


def test_rooting_scores_lags_and_candidates_match_python():
    rng = np.random.default_rng(9)
    source = rng.standard_normal(600)
    target = 0.6 * np.roll(source, 3) + 0.8 * rng.standard_normal(600)
    data = np.column_stack((source, target, rng.standard_normal(600), np.full(600, 2.0)))
    js = _run_node(
        "const r = DRR.rooting(input.columns, {maxLag: 5, nSurrogates: 0});\n"
        "process.stdout.write(JSON.stringify({scores: Array.from(r.scores),\n"
        "  lags: Array.from(r.lags), threshold: r.edge_threshold,\n"
        "  candidates: r.candidate_edges.map(e => [e.source, e.target, e.lag])}));",
        {"columns": data.T.tolist()},
    )
    python = RootingAnalyzer().analyze(data, max_lag=5, n_surrogates=0)
    np.testing.assert_allclose(js["scores"], python["score_matrix"].ravel(), atol=1e-12)
    assert js["lags"] == python["effective_lag"].ravel().tolist()
    assert js["threshold"] == pytest.approx(python["edge_threshold"], abs=1e-12)
    expected = [
        [int(edge["source"][4:]), int(edge["target"][4:]), edge["lag"]]
        for edge in python["candidate_edges"]
    ]
    assert js["candidates"] == expected


def test_javascript_surrogate_test_holds_its_size_and_finds_real_edges():
    js = _run_node(
        "const rng = DRR.createRng(11);\n"
        "const trials = 300, n = 256;\n"
        "let nullHits = 0, coupledHits = 0;\n"
        "for (let i = 0; i < trials; i += 1) {\n"
        "  const cols = [0, 1, 2].map(() => Float64Array.from({length: n}, rng.normal));\n"
        "  const r = DRR.rooting(cols, {maxLag: 4, nSurrogates: 49, seed: 1000 + i});\n"
        "  if (r.significant_edges.length) nullHits += 1;\n"
        "  const driven = Float64Array.from(cols[1], (v, t) =>\n"
        "    0.5 * cols[0][(t - 2 + n) % n] + 0.866 * v);\n"
        "  const c = DRR.rooting([cols[0], driven, cols[2]], {maxLag: 4, nSurrogates: 49,\n"
        "    seed: 5000 + i});\n"
        "  if (c.significant_edges.some(e => e.source === 0 && e.target === 1 && e.lag === 2))\n"
        "    coupledHits += 1;\n"
        "}\n"
        "process.stdout.write(JSON.stringify({nullRate: nullHits / trials,\n"
        "  power: coupledHits / trials}));",
        {},
    )
    assert js["nullRate"] <= 0.09
    assert js["power"] >= 0.95


def test_javascript_offsets_are_uniform_over_valid_configurations():
    js = _run_node(
        "const rng = DRR.createRng(5);\n"
        "const counts = {};\n"
        "for (let i = 0; i < 24000; i += 1) {\n"
        "  const key = DRR.circularShiftOffsets(12, 3, 3, rng).join(',');\n"
        "  counts[key] = (counts[key] || 0) + 1;\n"
        "}\n"
        "process.stdout.write(JSON.stringify(counts));",
        {},
    )
    rng = np.random.default_rng(0)
    python_support = {
        ",".join(map(str, RootingAnalyzer._circular_shift_offsets(12, 3, 3, rng)))
        for _ in range(5000)
    }
    assert set(js) == python_support
    expected = 24000 / len(js)
    assert max(abs(count - expected) for count in js.values()) < 0.15 * expected
