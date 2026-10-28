"""Evidence records for DRR's mathematical features.

A result can only be checked if its record says what was computed, from which
data, under which assumptions, and how sure anyone can be. A
:class:`MathematicalEvidenceRecord` keeps, in one JSON-serializable object:

* the definition of every feature used (equation, units, range, version,
  references, assumptions, limitations);
* the provenance of the data (SHA-256 of the analysed values, shape, sampling
  rate, preprocessing);
* every assumption with its status: checked and passed, checked and failed,
  not checkable from the data, or not applicable;
* the configuration, the measurements, and negative controls run on the same
  data;
* known limitations;
* statements, each classed as **proven** (a theorem, with its reference),
  **empirical** (a measurement, with the artifact or run that holds it), or
  **hypothesis** (untested);
* a conclusion of ``supported``, ``not_detected``, ``inconclusive`` or
  ``not_applicable``, with reasons, and reproducibility information.

The record is deterministic. It holds no wall-clock time, and its identifier is
the SHA-256 of its canonical JSON, so the same inputs always give the same
record. It complements :class:`drr_framework.evidence_card.DRREvidenceCard`,
which summarizes a single signal for review workflows.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EVIDENCE_SCHEMA_VERSION = "drr-math-evidence-v1"
STATEMENT_CLASSES = ("proven", "empirical", "hypothesis")
ASSUMPTION_STATUSES = ("checked_passed", "checked_failed", "unverifiable", "not_applicable")
CONCLUSIONS = ("supported", "not_detected", "inconclusive", "not_applicable")

BENCHMARK_ARTIFACT = "results/expected/resonance_math_benchmark.json"


@dataclass(frozen=True)
class MathematicalStatement:
    """A statement with its epistemic class and what supports it."""

    statement: str
    status: str
    support: str

    def __post_init__(self) -> None:
        if self.status not in STATEMENT_CLASSES:
            raise ValueError(f"status must be one of {STATEMENT_CLASSES}")
        if not self.statement.strip() or not self.support.strip():
            raise ValueError("statement and support must be non-empty")

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AssumptionCheck:
    """One assumption of an analysis and whether the data could check it."""

    assumption: str
    status: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in ASSUMPTION_STATUSES:
            raise ValueError(f"status must be one of {ASSUMPTION_STATUSES}")

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureDefinition:
    """Canonical definition of one mathematical feature."""

    name: str
    definition: str
    units: str
    value_range: str
    version: str
    references: Tuple[str, ...]
    assumptions: Tuple[str, ...]
    limitations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        for key in ("references", "assumptions", "limitations"):
            record[key] = list(record[key])
        return record


_PHASE_VERSION = "drr_phase_sync_v1"
_QUALITY_VERSION = "drr_data_quality_v1"
_RECURRENCE_VERSION = "drr_recurrence_v1"
_GEOMETRY_VERSION = "drr_trajectory_geometry_v1"

FEATURE_DEFINITIONS: Dict[str, FeatureDefinition] = {
    definition.name: definition
    for definition in (
        FeatureDefinition(
            name="phase_locking_value",
            definition=(
                "PLV_{n:m} = |mean_t exp(i (n phi_x(t) - m phi_y(t)))| with phi the phase of the "
                "Gaussian band (sd f0/6) analytic signal around each channel's dominant frequency"
            ),
            units="dimensionless",
            value_range="[0, 1]",
            version=_PHASE_VERSION,
            references=(
                "Tass et al. (1998), Phys. Rev. Lett. 81, 3291",
                "Lachaux et al. (1999), Hum. Brain Mapp. 8, 194",
            ),
            assumptions=(
                "each channel is a band-limited oscillation around its dominant frequency",
                "no aliasing",
            ),
            limitations=(
                "undirected",
                "ignores amplitude",
                "a delay longer than the phase-coherence time decorrelates the phases",
            ),
        ),
        FeatureDefinition(
            name="phase_shift_p_value",
            definition=(
                "fraction of circular shifts of the seam-corrected exp(i m phi_y) whose PLV with "
                "exp(i n phi_x) is at least the observed PLV; min-p over candidate ratios; Holm "
                "across channel pairs"
            ),
            units="probability",
            value_range="[1 / n_shifts, 1]",
            version=_PHASE_VERSION,
            references=(
                "Lehmann and Romano, Testing Statistical Hypotheses, 3rd ed., Section 15.2",
                "Westfall and Young (1993), Resampling-Based Multiple Testing",
                "Holm (1979), Scand. J. Stat. 6, 65",
            ),
            assumptions=(
                "independence of the channels under the null",
                "the shifted phasor is invariant in law under circular shifts (stationarity)",
            ),
            limitations=(
                "exact only for i.i.d. phase increments; approximate for band-pass filtered phases",
                "invalid under frequency drift or trends shared by the channels",
            ),
        ),
        FeatureDefinition(
            name="effective_phase_samples",
            definition=(
                "N / sum_k w(k) g_a(k) conj(g_b(k)) / (1 - |k|/N): Bartlett's variance formula for "
                "the mean phasor product of two independent channels, Bartlett lag window to N/4"
            ),
            units="samples",
            value_range="[1, N]",
            version=_PHASE_VERSION,
            references=("Bartlett (1935), J. R. Stat. Soc. 98, 536",),
            assumptions=("stationary phasors",),
            limitations=(
                "an estimate; truncation at N/4 understates very long phase memory",
                "the threshold of 20 for an inconclusive label is a judgment, not a theorem",
            ),
        ),
        FeatureDefinition(
            name="frequency_ratio",
            definition=(
                "coprime (n, m), n, m <= max order, with |n/m - f_y/f_x| <= tolerance * f_y/f_x; "
                "the Farey fractions of that order and their reciprocals"
            ),
            units="dimensionless",
            value_range="n/m in [1/max_order, max_order]",
            version=_PHASE_VERSION,
            references=("Hardy and Wright, An Introduction to the Theory of Numbers, Chapter 3",),
            assumptions=("each channel has one dominant frequency",),
            limitations=("ratios with larger integers are not searched",),
        ),
        FeatureDefinition(
            name="phase_difference_dynamics",
            definition=(
                "slope of psi = n phi_x - m phi_y over time / 2 pi (frequency mismatch, Hz); "
                "Var[psi(t + tau) - psi(t)] / (2 tau) after detrending (phase diffusion, rad^2/s); "
                "whole turns of psi over the record (net slips)"
            ),
            units="Hz; rad^2/s; turns",
            value_range="real; >= 0; integer",
            version=_PHASE_VERSION,
            references=("Pikovsky, Rosenblum and Kurths (2001), Synchronization, Chapter 9",),
            assumptions=("per-sample phase increments below pi (no aliasing)",),
            limitations=("descriptive; no test is attached",),
        ),
        FeatureDefinition(
            name="dfa_alpha",
            definition=(
                "slope of log F(s) against log s, F(s) the rms residual of linear fits to the "
                "cumulative centred series in windows of s samples, s from two periods to N/4"
            ),
            units="dimensionless",
            value_range="approximately (0, 2)",
            version=_QUALITY_VERSION,
            references=(
                "Peng et al. (1994), Phys. Rev. E 49, 1685",
                "Bardet and Kammoun (2008), IEEE Trans. Inf. Theory 54, 2041",
            ),
            assumptions=("a single scaling exponent describes the scale range",),
            limitations=(
                "finite-sample estimation error of about 0.1 at a few thousand samples",
                "oscillations and crossovers bend the log-log curve",
            ),
        ),
        FeatureDefinition(
            name="frequency_drift",
            definition=(
                "least-squares slope of the dominant frequency over four consecutive segments, "
                "times the record length"
            ),
            units="Hz",
            value_range="real",
            version=_QUALITY_VERSION,
            references=("DepthCalculator target-frequency estimate (this package)",),
            assumptions=("one dominant oscillation per channel",),
            limitations=("drift slower than one segment resolution is not detected",),
        ),
        FeatureDefinition(
            name="recurrence_determinism",
            definition=(
                "share of recurrence points (fixed recurrence rate, Theiler window excluded) on "
                "diagonal lines of at least two points in the delay-embedded trajectory"
            ),
            units="dimensionless",
            value_range="[0, 1]",
            version=_RECURRENCE_VERSION,
            references=(
                "Eckmann, Kamphorst and Ruelle (1987), Europhys. Lett. 4, 973",
                "Marwan et al. (2007), Phys. Rep. 438, 237",
                "Theiler (1986), Phys. Rev. A 34, 2427",
            ),
            assumptions=("the embedding unfolds the dynamics",),
            limitations=(
                "sensitive to noise, embedding and record length",
                "memory grows with the square of the record length",
            ),
        ),
        FeatureDefinition(
            name="second_order_recurrence",
            definition=(
                "least-squares x_t = a1 x_{t-1} + a2 x_{t-2}; roots of z^2 - a1 z - a2 give "
                "frequency omega fs / 2 pi, damping time -1/(fs ln rho), quality factor pi f tau"
            ),
            units="Hz; s; dimensionless",
            value_range="frequency in (0, fs/2)",
            version=_RECURRENCE_VERSION,
            references=("Box, Jenkins and Reinsel, Time Series Analysis, Section 3.2",),
            assumptions=("an AR(2) resonance describes the series",),
            limitations=("additive noise biases rho toward zero (overstates damping)",),
        ),
        FeatureDefinition(
            name="oriented_area",
            definition=(
                "A_ij = (1/2) sum_t [x_i(t) dx_j(t) - x_j(t) dx_i(t)] / ((N - 1) sigma_i sigma_j), "
                "identically (1/2)(rho_ij(1) - rho_ji(1)) up to normalization"
            ),
            units="dimensionless",
            value_range="[-1, 1]",
            version=_GEOMETRY_VERSION,
            references=(
                "Baryshnikov and Schlafly (2016), IEEE Conference on Decision and Control",
            ),
            assumptions=("none beyond finite variance",),
            limitations=(
                "a function of lag-one cross-covariance: no information beyond second-order "
                "statistics",
                "zero for in-phase and anti-phase coupling; lead is defined only modulo a period",
            ),
        ),
    )
}

#: Mathematical facts the features rely on. Each is a theorem or a direct
#: computation with its reference; none is an empirical claim.
PROVEN_STATEMENTS: Tuple[MathematicalStatement, ...] = (
    MathematicalStatement(
        "The phase-locking value lies in [0, 1], ignores amplitude, and is unchanged by a "
        "constant phase offset, hence by a delay that acts as one.",
        "proven",
        "|exp(i psi)| = 1 and the triangle inequality; a constant offset multiplies the mean "
        "phasor by a unit complex number.",
    ),
    MathematicalStatement(
        "Over the full cyclic group of shifts, the randomization p-value satisfies "
        "P(p <= alpha) <= alpha when the shifted channel is independent of the other and its "
        "law is invariant under circular shifts.",
        "proven",
        "Lehmann and Romano, Testing Statistical Hypotheses, 3rd ed., Section 15.2.",
    ),
    MathematicalStatement(
        "With independent, identically distributed phase increments, the ramp-corrected phasor "
        "is invariant in law under circular shifts.",
        "proven",
        "Cyclic shifts preserve the law of exchangeable increments, and the ramp depends only on "
        "their sum; docs/mathematical-extensions.md gives the argument.",
    ),
    MathematicalStatement(
        "Choosing the ratio n:m from each channel's own spectrum leaves the null distribution of "
        "the shift test unchanged.",
        "proven",
        "A circular shift leaves a channel's periodogram, hence its dominant frequency, "
        "unchanged, so the choice is constant on every orbit of the shift group.",
    ),
    MathematicalStatement(
        "Two pure tones at the same frequency have a phase-locking value of 1 whether or not they "
        "interact, so coupling is identifiable only when phases vary during the record.",
        "proven",
        "Direct computation: psi(t) is constant.",
    ),
    MathematicalStatement(
        "The oriented area of a centred two-channel trajectory equals ((N - 1) / 2) times the "
        "antisymmetrized lag-one cross-covariance.",
        "proven",
        "Elementary expansion; drr_framework.trajectory_geometry gives the proof.",
    ),
    MathematicalStatement(
        "Frequencies f and fs - f produce identical samples, so aliasing cannot be detected from "
        "the samples alone.",
        "proven",
        "cos(2 pi (fs - f) k / fs + a) = cos(2 pi f k / fs - a) for every integer k.",
    ),
    MathematicalStatement(
        "Asymptotically, DFA-1 of fractional Gaussian noise scales with exponent H and of "
        "fractional Brownian motion with exponent H + 1.",
        "proven",
        "Bardet and Kammoun (2008), IEEE Trans. Inf. Theory 54, 2041.",
    ),
)


def _canonical(value: Any) -> Any:
    """JSON-ready copy: NumPy to Python, non-finite floats to None, tuples to lists."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def array_sha256(values: Any) -> str:
    """SHA-256 of an array as little-endian float64 in C order, with its shape."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def environment() -> Dict[str, str]:
    """Versions that can change a numerical result."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("drr-framework", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


@dataclass
class MathematicalEvidenceRecord:
    """Everything needed to check one mathematical analysis."""

    feature_definitions: List[FeatureDefinition]
    data_provenance: Dict[str, Any]
    assumptions: List[AssumptionCheck]
    configuration: Dict[str, Any]
    measurements: Dict[str, Any]
    negative_controls: Dict[str, Any]
    limitations: List[str]
    statements: List[MathematicalStatement]
    conclusion: Dict[str, Any]
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.conclusion.get("status") not in CONCLUSIONS:
            raise ValueError(f"conclusion status must be one of {CONCLUSIONS}")

    def content(self) -> Dict[str, Any]:
        """The record without its identifier, in canonical JSON-ready form."""
        return _canonical(
            {
                "schema_version": self.schema_version,
                "feature_definitions": [d.to_dict() for d in self.feature_definitions],
                "data_provenance": self.data_provenance,
                "assumptions": [a.to_dict() for a in self.assumptions],
                "configuration": self.configuration,
                "measurements": self.measurements,
                "negative_controls": self.negative_controls,
                "limitations": list(self.limitations),
                "statements": [s.to_dict() for s in self.statements],
                "conclusion": self.conclusion,
                "reproducibility": self.reproducibility,
            }
        )

    @property
    def record_id(self) -> str:
        encoded = json.dumps(self.content(), sort_keys=True, allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"record_id": self.record_id, **self.content()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)

    def statements_by_class(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {status: [] for status in STATEMENT_CLASSES}
        for statement in self.statements:
            grouped[statement.status].append(statement.statement)
        return grouped


def binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    """``P(X >= successes)`` for ``X ~ Binomial(trials, probability)``."""
    from math import comb

    if trials < 0 or not 0 <= successes:
        raise ValueError("trials and successes must be non-negative")
    if successes > trials:
        return 0.0
    return float(
        sum(
            comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
            for k in range(successes, trials + 1)
        )
    )


def assumption_checks(
    quality: Mapping[str, Any], phase: Optional[Mapping[str, Any]]
) -> List[AssumptionCheck]:
    """Translate a data-quality report and a phase result into assumption checks."""
    reasons = list(quality.get("reasons", []))
    warnings = list(quality.get("warnings", []))
    drift = [r for r in reasons if "drifts" in r]
    scaling = [r for r in reasons if "DFA exponent" in r]
    sampling = quality.get("regularization")
    checks = [
        AssumptionCheck(
            "No random-walk-like nonstationarity (DFA exponent at most its threshold)",
            "checked_failed" if scaling else "checked_passed",
            "; ".join(scaling),
        ),
        AssumptionCheck(
            "The dominant frequency of each channel does not drift",
            "checked_failed" if drift else "checked_passed",
            "; ".join(drift) or "drift is assessed only where segment peaks can be resolved",
        ),
        AssumptionCheck(
            "Samples are regularly spaced and complete",
            (
                "checked_passed"
                if sampling is None
                else (
                    "checked_failed"
                    if any("interpolated" in r or "gap" in r for r in reasons)
                    else "checked_passed"
                )
            ),
            (
                "no regularization was needed"
                if sampling is None
                else f"{sampling.get('missing_fraction', 0.0):.1%} missing, interval CV "
                f"{sampling.get('interval_cv', 0.0):.3f}; values were linearly interpolated"
            ),
        ),
        AssumptionCheck(
            "No aliasing (all power below Nyquist)",
            "unverifiable",
            "; ".join(w for w in warnings if "Nyquist" in w)
            or "a sampled series cannot reveal aliasing",
        ),
        AssumptionCheck(
            "Channels are independent under the null hypothesis",
            "unverifiable",
            "this is the hypothesis being tested, not a property of the data",
        ),
    ]
    if phase is None:
        checks.append(
            AssumptionCheck(
                "Each channel has a usable oscillation for phase analysis", "not_applicable"
            )
        )
    else:
        unusable = [
            f"{name}: {record['reason']}"
            for name, record in phase.get("channels", {}).items()
            if not record.get("usable")
        ]
        narrow = [w for w in warnings if "of the power lies" in w]
        checks.append(
            AssumptionCheck(
                "Each channel has a usable, narrowband oscillation for phase analysis",
                "checked_failed" if unusable else "checked_passed",
                "; ".join(unusable + narrow),
            )
        )
    return checks


def build_evidence_record(
    *,
    analysed_data: Any,
    sampling_rate: float,
    column_names: Sequence[str],
    configuration: Mapping[str, Any],
    quality: Mapping[str, Any],
    phase: Optional[Mapping[str, Any]],
    measurements: Mapping[str, Any],
    negative_controls: Mapping[str, Any],
    input_sha256: Optional[str] = None,
    data_source: Optional[str] = None,
    features_used: Sequence[str] = (),
) -> MathematicalEvidenceRecord:
    """Assemble the record and decide its conclusion.

    The conclusion is ``inconclusive`` when the data-quality gate abstains or
    the negative controls reject more often than chance allows, ``supported``
    when at least one channel pair is phase synchronized, ``not_detected`` when
    every tested pair had the power to detect it and did not, and
    ``inconclusive`` otherwise. A single channel gives ``not_applicable``.
    """
    analysed = np.asarray(analysed_data, dtype=float)
    reasons: List[str] = []
    pairs = list(phase.get("pairs", [])) if phase is not None else []
    if phase is None:
        status = "not_applicable"
        reasons.append("fewer than two channels: no relationship was tested")
    elif quality.get("abstain"):
        status = "inconclusive"
        reasons.extend(quality.get("reasons", []))
    elif negative_controls.get("flagged"):
        status = "inconclusive"
        reasons.append(
            "phase-randomized negative controls were declared synchronized more often than "
            "chance allows; the test may be miscalibrated for these data"
        )
    elif any(p["status"] == "synchronized" for p in pairs):
        status = "supported"
        for pair in pairs:
            if pair["status"] == "synchronized":
                n_x, m_y = pair["selected_ratio"]
                reasons.append(
                    f"{pair['channels'][0]} and {pair['channels'][1]} are {n_x}:{m_y} phase "
                    f"synchronized (PLV {pair['plv']:.3f}, Holm-adjusted p = "
                    f"{pair['adjusted_p_value']:.4g})"
                )
    elif pairs and all(p["status"] == "not_detected" for p in pairs):
        status = "not_detected"
        reasons.append("no tested pair was synchronized, and every pair had phase variability")
    else:
        status = "inconclusive"
        reasons.extend(
            f"{p['channels'][0]}-{p['channels'][1]}: {p['reason']}"
            for p in pairs
            if p.get("reason")
        )

    statements: List[MathematicalStatement] = list(PROVEN_STATEMENTS)
    statements.append(
        MathematicalStatement(
            "On the synthetic benchmark's held-out conditions, the phase test's false-relationship "
            "rate and detection rate are those recorded in the artifact.",
            "empirical",
            BENCHMARK_ARTIFACT,
        )
    )
    if status == "supported":
        statements.append(
            MathematicalStatement(
                "The synchronized channels interact through a physical or causal mechanism.",
                "hypothesis",
                "untested: a phase relationship is consistent with coupling, a common driver, or "
                "shared nonstationarity the checks did not detect",
            )
        )
    limitations = [
        "Phase synchronization is undirected and does not establish causation.",
        "The data-quality checks detect random-walk scaling and frequency drift; other forms of "
        "nonstationarity can pass them.",
        "Negative controls are phase-randomized copies, which are stationary by construction; they "
        "check the test, not the stationarity of the data.",
        "Benchmark performance was measured on synthetic oscillators and does not transfer to a "
        "domain without a study of that domain.",
    ]
    names = list(features_used) or list(FEATURE_DEFINITIONS)
    return MathematicalEvidenceRecord(
        feature_definitions=[FEATURE_DEFINITIONS[name] for name in names],
        data_provenance={
            "source": data_source,
            "input_sha256": input_sha256,
            "analysed_sha256": array_sha256(analysed),
            "analysed_shape": list(analysed.shape),
            "column_names": list(column_names),
            "sampling_rate_hz": float(sampling_rate),
            "regularization": quality.get("regularization"),
        },
        assumptions=assumption_checks(quality, phase),
        configuration=dict(configuration),
        measurements=dict(measurements),
        negative_controls=dict(negative_controls),
        limitations=limitations,
        statements=statements,
        conclusion={"status": status, "reasons": reasons},
        reproducibility={
            "environment": environment(),
            "configuration_sha256": hashlib.sha256(
                json.dumps(_canonical(dict(configuration)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    )


__all__ = [
    "ASSUMPTION_STATUSES",
    "AssumptionCheck",
    "CONCLUSIONS",
    "EVIDENCE_SCHEMA_VERSION",
    "FEATURE_DEFINITIONS",
    "FeatureDefinition",
    "MathematicalEvidenceRecord",
    "MathematicalStatement",
    "PROVEN_STATEMENTS",
    "STATEMENT_CLASSES",
    "array_sha256",
    "assumption_checks",
    "binomial_upper_tail",
    "build_evidence_record",
    "environment",
]
