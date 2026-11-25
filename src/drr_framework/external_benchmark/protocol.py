"""Decision rules for the preregistered external comparison.

Score construction lives with the dataset. This module only calibrates
thresholds, counts hits, draws the block bootstrap, and assigns a claim status.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

CLAIM_STATUSES = ("supported", "not_supported", "inconclusive")
REASON_TEXT = {
    "repeated_unique_detections": (
        "H1, H2, and H3 hold, and the bootstrap interval for the unique-hit count "
        "stays at or above 1."
    ),
    "drr_false_alarm_rate": (
        "The full DRR holdout false-alarm rate exceeds the preregistered tolerance."
    ),
    "no_unique_detection": (
        "Full DRR did not flag a disruption that every calibrated conventional model missed."
    ),
    "ablation_sufficient": (
        "An ablation inside the tolerance flagged every disruption that was unique to full DRR."
    ),
    "drr_uncalibrated": (
        "Full DRR had no threshold that kept the estimation false-alarm rate inside the tolerance."
    ),
    "single_unique_detection": (
        "Full DRR uniquely flagged one disruption. The preregistered repeat count is two."
    ),
    "uncertainty_includes_zero": (
        "The point estimate met H1, H2, and H3, and the bootstrap interval for the "
        "unique-hit count includes values below 1."
    ),
    "no_calibrated_challenge": (
        "No conventional model calibrated inside the tolerance, so the comparison was not run."
    ),
    "unscored_event": "A preregistered disruption window had no finite scores.",
}


def validate_preregistration(spec: Mapping) -> None:
    """Reject a plan that does not carry the frozen comparison."""
    analysis = spec["analysis"]
    hypotheses = [item["id"] for item in spec["hypotheses"]]
    if hypotheses != ["H1", "H2", "H3"]:
        raise ValueError("Preregistration must contain H1, H2, and H3 in that order")
    if spec["universal"] is not False or spec["claim_class"] != "domain_adapter":
        raise ValueError("This comparison is a domain-adapter study")
    if not 0 < float(analysis["false_alarm_tolerance"]) < 1:
        raise ValueError("False-alarm tolerance must be between 0 and 1")
    if int(analysis["minimum_repeat_count"]) < 2:
        raise ValueError("The repeat count must require more than one disruption")
    events = spec["events"]
    if len(events) < int(analysis["minimum_repeat_count"]):
        raise ValueError("The plan must name at least as many events as the repeat count")
    roles = {item["role"] for item in spec["models"].values()}
    if roles != {"conventional", "ablation", "drr"}:
        raise ValueError("The plan must name conventional models, ablations, and full DRR")
    if sum(item["role"] == "drr" for item in spec["models"].values()) != 1:
        raise ValueError("The plan must name one full DRR model")
    bootstrap = analysis["bootstrap"]
    if int(bootstrap["replicates"]) < 1 or int(bootstrap["block_length_months"]) < 1:
        raise ValueError("Bootstrap replicates and block length must be positive")


def calibrate(scores: np.ndarray, tolerance: float) -> Optional[float]:
    """Least strict threshold whose alarm rate is at most ``tolerance``.

    An alarm is a finite score at or above the threshold. The search walks the
    finite scores upward and stops at the first candidate that satisfies the
    budget, which is the least strict admissible threshold.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not 0 < float(tolerance) < 1:
        return None
    for candidate in np.sort(values):
        if float(np.mean(values >= candidate)) <= float(tolerance):
            return float(candidate)
    return None


def alarm_rate(scores: np.ndarray, threshold: Optional[float], mask: np.ndarray) -> Optional[float]:
    if threshold is None:
        return None
    values = np.asarray(scores, dtype=float)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values >= threshold))


def window_call(scores: np.ndarray, threshold: Optional[float], mask: np.ndarray) -> str:
    """Return hit, miss, or unscored for one disruption window."""
    values = np.asarray(scores, dtype=float)[np.asarray(mask, dtype=bool)]
    if not np.any(np.isfinite(values)):
        return "unscored"
    if threshold is None or not np.any(values[np.isfinite(values)] >= threshold):
        return "miss"
    return "hit"


def classify_claim(
    *,
    events_comparable: bool,
    drr_calibrated: bool,
    drr_holdout_fpr: Optional[float],
    tolerance: float,
    calibrated_challenge_count: int,
    unique_hit_count: int,
    minimum_repeat_count: int,
    ablation_explains_unique_set: bool,
    bootstrap_unique_low: Optional[float],
) -> Tuple[str, str]:
    """Apply the preregistered decision rule. The interval cannot rescue a miss."""
    if not events_comparable:
        return "inconclusive", "unscored_event"
    if not drr_calibrated:
        return "not_supported", "drr_uncalibrated"
    if calibrated_challenge_count < 1:
        return "inconclusive", "no_calibrated_challenge"
    if drr_holdout_fpr is None or drr_holdout_fpr > tolerance:
        return "not_supported", "drr_false_alarm_rate"
    if unique_hit_count <= 0:
        return "not_supported", "no_unique_detection"
    if unique_hit_count < minimum_repeat_count:
        return "inconclusive", "single_unique_detection"
    if ablation_explains_unique_set:
        return "not_supported", "ablation_sufficient"
    if bootstrap_unique_low is None or bootstrap_unique_low < 1:
        return "inconclusive", "uncertainty_includes_zero"
    return "supported", "repeated_unique_detections"


def claim_paragraph(status: str, reason: str, domain: str) -> str:
    if status == "supported":
        lead = (
            "Full DRR flagged repeated preregistered disruptions that the calibrated "
            "conventional models missed, inside the false-alarm tolerance, and neither "
            "ablation reproduced that set inside the tolerance."
        )
    elif status == "not_supported":
        lead = "The preregistered claim is not supported."
    elif status == "inconclusive":
        lead = "The preregistered claim is inconclusive."
    else:
        raise ValueError("Unknown claim status")
    return f"{lead} {REASON_TEXT[reason]} Domain: {domain}. " "The result stays inside that domain."


def _resample_positions(n_rows: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length > n_rows:
        raise ValueError("Bootstrap block length must fit inside the estimation sample")
    n_blocks = int(np.ceil(n_rows / block_length))
    starts = rng.integers(0, n_rows, size=n_blocks)
    offsets = np.arange(block_length)
    positions = np.concatenate([(start + offsets) % n_rows for start in starts])
    return positions[:n_rows]


def _quantile_interval(
    values: Union[Sequence[float], np.ndarray], lower: float, upper: float
) -> Tuple[float, float]:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Bootstrap interval requires finite replicates")
    low, high = np.percentile(finite, [100 * lower, 100 * upper])
    return float(low), float(high)


def evaluate_panel(panel: Mapping, spec: Mapping) -> Dict:
    """Calibrate every scored model and assign the claim.

    ``panel`` carries aligned monthly scores. Withheld models are recorded and
    then left out of the horse race.
    """
    validate_preregistration(spec)
    months = list(panel["months"])
    n_rows = len(months)
    estimation = np.asarray([split == "estimation" for split in panel["split"]], dtype=bool)
    holdout = ~estimation
    if estimation.sum() == 0 or holdout.sum() == 0:
        raise ValueError("The panel must contain estimation months and holdout months")
    event_masks = {
        event_id: np.asarray(mask, dtype=bool) for event_id, mask in panel["events"].items()
    }
    if any(len(mask) != n_rows for mask in event_masks.values()):
        raise ValueError("Event masks must align with the monthly panel")
    event_union = np.zeros(n_rows, dtype=bool)
    for mask in event_masks.values():
        event_union |= mask
    if np.any(event_union & estimation):
        raise ValueError("A preregistered event falls inside the estimation period")
    holdout_non_event = holdout & ~event_union
    tolerance = float(spec["analysis"]["false_alarm_tolerance"])
    roles = {name: body["role"] for name, body in spec["models"].items()}
    overrides = dict(panel.get("withheld") or {})
    summaries = []
    calibrated_scores: Dict[str, Tuple[np.ndarray, float]] = {}

    for name, body in spec["models"].items():
        if name in overrides:
            summaries.append(
                {
                    "name": name,
                    "role": body["role"],
                    "definition": body["definition"],
                    "status": "withheld",
                    "reason": overrides[name],
                    "threshold": None,
                    "estimation_false_alarm_rate": None,
                    "holdout_false_alarm_rate": None,
                    "events": [],
                    "alarm_months": [],
                }
            )
            continue
        scores = np.asarray(panel["scores"][name], dtype=float)
        if len(scores) != n_rows:
            raise ValueError(f"{name} scores do not align with the panel")
        threshold = calibrate(scores[estimation], tolerance)
        events = []
        for event_id, mask in event_masks.items():
            events.append({"id": event_id, "call": window_call(scores, threshold, mask)})
        alarm_months = [
            months[index]
            for index in range(n_rows)
            if threshold is not None
            and holdout_non_event[index]
            and np.isfinite(scores[index])
            and scores[index] >= threshold
        ]
        status = "calibrated" if threshold is not None else "uncalibrated"
        summary = {
            "name": name,
            "role": body["role"],
            "definition": body["definition"],
            "status": status,
            "reason": (
                None if status == "calibrated" else "no threshold inside the false-alarm tolerance"
            ),
            "threshold": threshold,
            "estimation_false_alarm_rate": alarm_rate(scores, threshold, estimation),
            "holdout_false_alarm_rate": alarm_rate(scores, threshold, holdout_non_event),
            "events": events,
            "alarm_months": alarm_months,
        }
        summaries.append(summary)
        if status == "calibrated":
            calibrated_scores[name] = (scores, float(cast(float, threshold)))

    by_name = {item["name"]: item for item in summaries}
    drr_name = next(name for name, role in roles.items() if role == "drr")
    drr = by_name[drr_name]
    conventional = [item for item in summaries if item["role"] == "conventional"]
    challenges = [item for item in conventional if item["status"] == "calibrated"]
    ablations = [item for item in summaries if item["role"] == "ablation"]
    drr_scores = np.asarray(panel["scores"][drr_name], dtype=float)

    def comparable(event_id: str) -> bool:
        mask = event_masks[event_id]
        if window_call(drr_scores, drr["threshold"], mask) == "unscored":
            return False
        for challenge in challenges:
            scores, threshold = calibrated_scores[challenge["name"]]
            if window_call(scores, threshold, mask) == "unscored":
                return False
        return True

    events_comparable = all(comparable(event_id) for event_id in event_masks)
    unique_ids = []
    if events_comparable and drr["status"] == "calibrated":
        for event_id, mask in event_masks.items():
            drr_hit = window_call(drr_scores, drr["threshold"], mask) == "hit"
            challenge_hit = False
            for challenge in challenges:
                scores, threshold = calibrated_scores[challenge["name"]]
                if window_call(scores, threshold, mask) == "hit":
                    challenge_hit = True
                    break
            if drr_hit and not challenge_hit:
                unique_ids.append(event_id)

    def ablation_explains(summary: Mapping) -> bool:
        if summary["status"] != "calibrated":
            return False
        rate = summary["holdout_false_alarm_rate"]
        if rate is None or rate > tolerance:
            return False
        hits = {item["id"] for item in summary["events"] if item["call"] == "hit"}
        return set(unique_ids).issubset(hits) and bool(unique_ids)

    explained = any(ablation_explains(item) for item in ablations)
    bootstrap = _bootstrap_unique_hits(
        estimation=estimation,
        holdout_non_event=holdout_non_event,
        event_masks=event_masks,
        scores={name: np.asarray(panel["scores"][name], dtype=float) for name in panel["scores"]},
        roles=roles,
        drr_name=drr_name,
        tolerance=tolerance,
        bootstrap=spec["analysis"]["bootstrap"],
    )
    status, reason = classify_claim(
        events_comparable=events_comparable,
        drr_calibrated=drr["status"] == "calibrated",
        drr_holdout_fpr=drr["holdout_false_alarm_rate"],
        tolerance=tolerance,
        calibrated_challenge_count=len(challenges),
        unique_hit_count=len(unique_ids),
        minimum_repeat_count=int(spec["analysis"]["minimum_repeat_count"]),
        ablation_explains_unique_set=explained,
        bootstrap_unique_low=bootstrap["unique_hit_count"]["low"],
    )
    return {
        "models": summaries,
        "unique_event_ids": unique_ids,
        "uncertainty": bootstrap,
        "claim": {
            "status": status,
            "reason": reason,
            "text": claim_paragraph(status, reason, spec["domain"]),
            "hypotheses": list(spec["hypotheses"]),
            "domain": spec["domain"],
            "claim_class": spec["claim_class"],
            "universal": False,
        },
    }


def _bootstrap_unique_hits(
    *,
    estimation: np.ndarray,
    holdout_non_event: np.ndarray,
    event_masks: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    roles: Mapping[str, str],
    drr_name: str,
    tolerance: float,
    bootstrap: Mapping,
) -> Dict:
    estimation_positions = np.flatnonzero(estimation)
    rng = np.random.default_rng(int(bootstrap["seed"]))
    replicates = int(bootstrap["replicates"])
    block = int(bootstrap["block_length_months"])
    unique_counts = np.zeros(replicates, dtype=float)
    false_alarm_rates = np.ones(replicates, dtype=float)
    challenge_names = [
        name for name, role in roles.items() if role == "conventional" and name in scores
    ]
    ablation_names = [name for name, role in roles.items() if role == "ablation" and name in scores]
    for replicate in range(replicates):
        draw = _resample_positions(len(estimation_positions), block, rng)
        thresholds = {}
        for name in list(challenge_names) + ablation_names + [drr_name]:
            sample = scores[name][estimation_positions][draw]
            thresholds[name] = calibrate(sample, tolerance)
        drr_threshold = thresholds[drr_name]
        if drr_threshold is None:
            unique_counts[replicate] = 0.0
            false_alarm_rates[replicate] = 1.0
            continue
        rate = alarm_rate(scores[drr_name], drr_threshold, holdout_non_event)
        false_alarm_rates[replicate] = 1.0 if rate is None else rate
        unique = 0
        for mask in event_masks.values():
            if window_call(scores[drr_name], drr_threshold, mask) != "hit":
                continue
            challenge_hit = False
            for name in challenge_names:
                threshold = thresholds[name]
                if threshold is None:
                    continue
                if window_call(scores[name], threshold, mask) == "hit":
                    challenge_hit = True
                    break
            if not challenge_hit and challenge_names:
                unique += 1
        unique_counts[replicate] = float(unique)
    low, high = _quantile_interval(
        unique_counts, float(bootstrap["lower_quantile"]), float(bootstrap["upper_quantile"])
    )
    fpr_low, fpr_high = _quantile_interval(
        false_alarm_rates, float(bootstrap["lower_quantile"]), float(bootstrap["upper_quantile"])
    )
    return {
        "method": bootstrap["method"],
        "replicates": replicates,
        "block_length_months": block,
        "seed": int(bootstrap["seed"]),
        "quantiles": [float(bootstrap["lower_quantile"]), float(bootstrap["upper_quantile"])],
        "unique_hit_count": {"low": low, "high": high},
        "drr_holdout_false_alarm_rate": {"low": fpr_low, "high": fpr_high},
        "note": (
            "One circular-block draw resamples estimation months and is applied to every "
            "model. Holdout scores stay fixed. A replicate that cannot calibrate full DRR "
            "counts as no unique hits and a false-alarm rate of 1."
        ),
    }


def render_report(artifact: Mapping) -> str:
    """Render the claim from the artifact. The prose does not add a result."""
    claim = artifact["claim"]
    lines = [
        "# QBO structural-change benchmark",
        "",
        f"Claim status: **{claim['status']}**",
        "",
        claim["text"],
        "",
        f"Reason code: `{claim['reason']}`.",
        "",
        "This is one domain adapter, equatorial stratospheric zonal wind. "
        "A result here does not transfer to physics, sensing, macro policy, "
        "supervision, financial markets, or another adapter.",
        "",
        "## Preregistered question",
        "",
        "Does full DRR flag repeated, published QBO disruptions that rolling volatility, "
        "rolling correlation, and a VAR residual miss, without a holdout false-alarm rate "
        "above the preregistered tolerance, after spectral-only and rooting-only ablations?",
        "",
        "## Hypotheses",
        "",
    ]
    for item in claim["hypotheses"]:
        lines.append(f"- **{item['id']}.** {item['statement']}")
    lines.extend(["", "## Models", ""])
    lines.append("| Model | Role | Status | Estimation FPR | Holdout FPR | Windows |")
    lines.append("| --- | --- | --- | ---: | ---: | --- |")
    for model in artifact["models"]:
        windows = ", ".join(f"{item['id']} {item['call']}" for item in model["events"]) or "—"
        lines.append(
            "| {name} | {role} | {status} | {est} | {hold} | {windows} |".format(
                name=model["name"],
                role=model["role"],
                status=model["status"],
                est=_fmt(model["estimation_false_alarm_rate"]),
                hold=_fmt(model["holdout_false_alarm_rate"]),
                windows=windows,
            )
        )
    unique = artifact["unique_event_ids"]
    uncertainty = artifact["uncertainty"]["unique_hit_count"]
    lines.extend(
        [
            "",
            "## Unique detections and uncertainty",
            "",
            "Unique full-DRR detections: " + (", ".join(unique) if unique else "none") + ".",
            "",
            "Bootstrap interval for the unique-hit count: "
            f"{uncertainty['low']:.3f} to {uncertainty['high']:.3f} "
            f"({artifact['uncertainty']['replicates']} circular-block replicates).",
            "",
            artifact["uncertainty"]["note"],
            "",
            "## Dataset",
            "",
            f"{artifact['dataset']['name']}, original-data section, retrieved "
            f"{artifact['dataset']['retrieved']}. "
            f"Finite months {artifact['dataset']['first_month']} through "
            f"{artifact['dataset']['last_month']} "
            f"({artifact['dataset']['finite_months']} months). "
            f"Holdout starts {artifact['analysis']['holdout_start']}. "
            f"False-alarm tolerance {artifact['analysis']['false_alarm_tolerance']}.",
            "",
            "Checksums of the vendored CPC snapshots are in the preregistration. "
            "The JSON artifact is the machine-readable record.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def json_ready(value):
    """Convert NumPy values into objects the standard library can encode."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def month_span(start: str, end: str) -> Iterable[str]:
    year, month = (int(part) for part in start.split("-"))
    last_year, last_month = (int(part) for part in end.split("-"))
    while (year, month) <= (last_year, last_month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year, month = year + 1, 1
