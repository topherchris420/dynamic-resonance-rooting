"""DRR-Mellin hidden-mode recovery benchmark: operators, statistics and the committed artifact."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sps
from scipy import stats

from drr_framework import mellin_recovery_benchmark as bench
from drr_framework.mellin import fit_log_mode, log_mode_profile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "expected" / "mellin_recovery"
ARTIFACT = RESULTS / "mellin_recovery_benchmark.json"
CFG = bench.DEFAULT_CONFIG


def _load():
    if not ARTIFACT.exists():
        pytest.skip("committed artifact not generated yet")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ DRR-Mellin profile


@pytest.mark.parametrize("kind", ["log_uniform", "integers"])
def test_profile_selects_the_fit_log_mode_candidate(kind):
    rng = np.random.default_rng(7)
    betas, omegas = bench.search_grids(CFG)
    x = np.exp(np.linspace(0, 6.9, 300)) if kind == "log_uniform" else 20 + np.arange(300.0)
    u = np.log(x)
    for _ in range(6):
        y = x ** rng.uniform(-0.6, 0.6) * np.cos(rng.uniform(8, 40) * u + rng.uniform(0, 6))
        y = y + rng.normal(0, rng.uniform(0.5, 20) * np.std(y), x.size)
        surface = log_mode_profile(x, y, betas, omegas)
        row, col = np.unravel_index(np.nanargmin(surface), surface.shape)
        reference = fit_log_mode(x, y, betas, omegas)
        assert (betas[row], omegas[col]) == (reference["beta"], reference["omega"])
        assert surface[row, col] == pytest.approx(reference["relative_sse"], abs=1e-12)


def test_profile_noiseless_and_skips_unidentifiable_candidates():
    u = np.arange(256) * (2 * np.pi / 256)
    x = np.exp(u)
    y = np.sqrt(x) * np.cos(14 * u + 0.4)
    surface = log_mode_profile(x, y, [0.4, 0.5, 0.6], [13, 14, 15])
    assert np.unravel_index(np.nanargmin(surface), surface.shape) == (1, 1)
    assert surface[1, 1] < 1e-12
    # omega = 256 cycles over a 2*pi span aliases sin to zero: rank one, skipped as in fit_log_mode.
    aliased = log_mode_profile(x, y, [0.5], [256.0])
    assert np.isnan(aliased[0, 0])
    with pytest.raises(ValueError):
        fit_log_mode(x, y, [0.5], [256.0])
    with pytest.raises(ValueError):
        log_mode_profile(x, np.zeros_like(x), [0.5], [14.0])


# ------------------------------------------------------------------ simulators and methods


@pytest.mark.parametrize("noise", ["white", "scale", "student_t3"])
def test_signal_trial_has_the_requested_integrated_snr(noise):
    design = CFG["holdout"]["designs"][0]
    x = bench.design_grid(design)
    rng = np.random.default_rng(11)
    signal_power = noise_power = 0.0
    for _ in range(400):
        y, truth = bench.simulate_signal_trial(x, noise, 0.6, 14.0, CFG["omega_draw"], rng)
        u = np.log(x)
        clean = np.exp(0.6 * (u - u.mean())) * np.cos(truth["omega"] * u + truth["phase"])
        signal_power += np.mean(clean**2)
        noise_power += np.mean((y - clean) ** 2)
    # The definition uses the expected noise power, so compare a ratio of means.
    assert 10 * np.log10(x.size * signal_power / noise_power) == pytest.approx(14.0, abs=0.4)


def test_scale_noise_follows_the_envelope_and_null_has_no_signal():
    x = bench.design_grid(CFG["holdout"]["designs"][0])
    shape = bench.noise_shape(x, "scale", 0.6)
    assert np.mean(shape**2) == pytest.approx(1.0)
    assert shape[-1] / shape[0] == pytest.approx(x[-1] ** 0.6 / x[0] ** 0.6)
    assert np.all(bench.noise_shape(x, "white", 0.6) == 1.0)
    nulls = np.array(
        [bench.simulate_null_trial(x, "white", 0.0, np.random.default_rng(s)) for s in range(50)]
    )
    assert abs(nulls.mean()) < 0.02


@pytest.mark.parametrize("design_index", [0, 1])
@pytest.mark.parametrize("beta", [-0.6, 0.0, 0.6])
def test_every_method_recovers_a_clean_off_grid_mode(design_index, beta):
    design = CFG["holdout"]["designs"][design_index]
    x = bench.design_grid(design)
    rng = np.random.default_rng(100 + design_index)
    y, truth = bench.simulate_signal_trial(x, "white", beta, 35.0, [17.33, 17.33], rng)
    tolerance = bench.recovery_tolerance(x)
    for method, (omega, beta_hat, _) in bench.analyze_record(x, y, CFG).items():
        assert abs(omega - truth["omega"]) <= tolerance, method
        if method in ("drr_mellin_scan", "nls", "esprit"):
            assert beta_hat == pytest.approx(beta, abs=0.15), method


def test_lomb_scargle_statistic_is_the_explained_energy_of_a_beta_zero_fit():
    x = bench.design_grid(CFG["development"]["designs"][1])
    rng = np.random.default_rng(5)
    y = np.cos(12.3 * np.log(x)) + rng.normal(0, 2, x.size)
    centred = y - y.mean()
    omegas = np.array([12.3, 20.0])
    power = sps.lombscargle(np.log(x), centred, omegas)[0]
    u = np.log(x)
    design = np.column_stack((np.cos(12.3 * u), np.sin(12.3 * u)))
    residual = centred - design @ np.linalg.lstsq(design, centred, rcond=None)[0]
    explained = 1 - residual @ residual / (centred @ centred)
    assert 2 * power / (centred @ centred) == pytest.approx(explained, rel=1e-6)


def test_esprit_recovers_a_noiseless_pole():
    du = 6.0 / 299
    u = np.arange(300) * du
    x = np.exp(u)
    omega, beta, energy = bench.method_esprit(x, np.exp(-0.4 * u) * np.cos(9.5 * u + 1.0), 3)
    assert omega == pytest.approx(9.5, abs=1e-3)
    assert beta == pytest.approx(-0.4, abs=0.02)
    assert energy > 0.99


# ------------------------------------------------------------------ statistics


def test_calibration_threshold_is_a_conservative_rank_quantile():
    values = np.arange(1, 500)  # 499 calibration statistics
    threshold = bench.calibration_threshold(values, 0.05)
    assert threshold == 475  # ceil(0.95 * 500) = 475th smallest
    rng = np.random.default_rng(1)
    exceed = []
    for _ in range(2000):
        draws = rng.standard_normal(100)
        exceed.append(rng.standard_normal() > bench.calibration_threshold(draws, 0.05))
    assert np.mean(exceed) == pytest.approx(0.05, abs=0.015)


def test_codes_round_trip():
    codes = "".join(bench._code(r, d) for r, d in [(0, 0), (1, 0), (0, 1), (1, 1)])
    assert codes == "0123"
    recovered, detected = bench.decode(codes)
    assert recovered.tolist() == [False, True, False, True]
    assert detected.tolist() == [False, False, True, True]
    assert bench.endpoint(codes, "detected_and_recovered").tolist() == [0, 0, 0, 1]
    assert bench.endpoint(codes, "detected_not_recovered").tolist() == [0, 0, 1, 0]


def test_mcnemar_holm_and_paired_difference():
    first = np.array([1] * 30 + [0] * 10 + [1] * 50 + [0] * 10, dtype=bool)
    second = np.array([0] * 30 + [1] * 10 + [1] * 50 + [0] * 10, dtype=bool)
    test = bench.mcnemar_greater(first, second)
    assert (test["wins"], test["losses"]) == (30, 10)
    assert test["p_value"] == pytest.approx(stats.binom.sf(29, 40, 0.5))
    assert bench.mcnemar_greater(second, second)["p_value"] == 1.0
    assert bench.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    diff = bench.paired_difference(first, second)
    assert diff["mean"] == pytest.approx(0.2)
    d = first.astype(float) - second
    assert diff["se"] == pytest.approx(d.std(ddof=1) / 10)
    assert bench.wilson_interval(5, 100) == pytest.approx([0.02154, 0.11175], abs=1e-4)


def _row(noise, beta, codes):
    return {
        "noise": noise,
        "design": "d",
        "beta": beta,
        "integrated_snr_db": 8.0,
        "codes": codes,
    }


def test_hypothesis_rules_on_synthetic_rows():
    n = 400
    strong = {m: "1" * n for m in bench.METHODS}
    strong["lomb_scargle"] = "1" * 300 + "0" * 100
    strong["nls"] = "1" * 320 + "0" * 80
    strong["esprit"] = "0" * n
    flat = {m: "1" * 200 + "0" * 200 for m in bench.METHODS}
    scale = {m: "1" * 200 + "0" * 200 for m in bench.METHODS}
    scale["lomb_scargle"] = "1" * 260 + "0" * 140
    rows = [
        _row("white", 0.6, strong),
        _row("white", -0.6, strong),
        _row("white", 0.0, flat),
        _row("scale", 0.6, scale),
    ]
    out = {h["id"]: h for h in bench.evaluate_hypotheses(rows, CFG)}
    assert out["H1"]["outcome"] == "supported"
    assert out["H2"]["outcome"] == "not supported"  # nothing detected
    assert out["H3"]["outcome"] == "supported"
    assert out["H4"]["outcome"] == "supported"
    assert out["H5"]["outcome"] == "supported"
    worse = copy.deepcopy(rows)
    worse[2]["codes"][bench.PRIMARY] = "1" * 150 + "0" * 250
    assert {h["id"]: h for h in bench.evaluate_hypotheses(worse, CFG)}["H4"][
        "outcome"
    ] == "not supported"


def test_conditions_have_disjoint_seeds_and_one_group_per_noise_shape():
    for split in ("development", "holdout"):
        built = bench.build_conditions(CFG, split)
        offsets = [r["seed_offset"] for r in built["signal"]]
        offsets += [g["calibration_seed_offset"] for g in built["null"]]
        offsets += [g["fresh_seed_offset"] for g in built["null"]]
        assert len(offsets) == len(set(offsets))
        assert {r["group"] for r in built["signal"]} == {g["group"] for g in built["null"]}
    holdout = bench.build_conditions(CFG, "holdout")
    assert len(holdout["signal"]) == 120
    assert len(holdout["null"]) == 2 * (2 + 5)


def test_split_rows_do_not_depend_on_worker_count():
    cfg = copy.deepcopy(CFG)
    cfg["development"].update(
        designs=cfg["development"]["designs"][:1],
        betas=[0.5],
        integrated_snr_db=[12.0, 15.0],
        n_signal=4,
        n_null_calibration=20,
        n_null_fresh=6,
    )
    serial = bench.run_split(cfg, "development", workers=1)
    parallel = bench.run_split(cfg, "development", workers=2)
    assert serial == parallel


# ------------------------------------------------------------------ committed artifact


def test_preregistration_fingerprint_is_frozen():
    assert bench.FROZEN_PREREGISTRATION_SHA256
    assert bench.preregistration_sha256(CFG) == bench.FROZEN_PREREGISTRATION_SHA256


def test_committed_artifact_is_consistent():
    artifact = _load()
    assert artifact["schema_version"] == bench.SCHEMA_VERSION
    assert artifact["config"] == json.loads(json.dumps(CFG))
    assert artifact["matches_frozen_preregistration"] is True
    assert all(artifact["checks"].values())
    assert (RESULTS / "report.md").read_text(encoding="utf-8") == bench.render_report(artifact)
    if artifact["holdout"]:
        recomputed = bench.evaluate_hypotheses(artifact["holdout"]["signal"], artifact["config"])
        assert [h["outcome"] for h in recomputed] == [h["outcome"] for h in artifact["hypotheses"]]
        for row in artifact["holdout"]["signal"]:
            assert all(len(c) == row["n_trials"] for c in row["codes"].values())


def test_committed_rows_replay_exactly():
    artifact = _load()
    cfg = artifact["config"]
    for split in ("development", "holdout"):
        part = artifact[split]
        if not part:
            continue
        row = part["signal"][0]
        spec = next(s for s in bench.build_conditions(cfg, split)["signal"] if s["id"] == row["id"])
        thresholds = next(g["thresholds"] for g in part["null"] if g["group"] == row["group"])
        replay = bench.run_signal_condition(spec, cfg, 5, thresholds, check_fit_log_mode=False)
        for method, codes in replay["codes"].items():
            assert codes == row["codes"][method][:5], (split, method)


def test_report_states_what_was_and_was_not_supported():
    artifact = _load()
    report = (RESULTS / "report.md").read_text(encoding="utf-8")
    for h in artifact["hypotheses"]:
        assert f"| {h['id']} | **{h['outcome']}** |" in report
