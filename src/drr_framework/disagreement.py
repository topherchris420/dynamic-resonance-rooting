"""
Disagreement Principle and Representation Scope Registry for DRR Architecture.

This module implements the Disagreement Principle and Representation Scope Registry for
Dynamic Resonance Rooting (DRR), enforcing non-erasure scope resolution across divergent
observational scales. High-confidence macro-level signals are prevented from silently overwriting,
filtering, or smoothing out conflicting metrics from local/micro-level observations.
"""

from copy import deepcopy
from dataclasses import dataclass
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


class ObservationalScale(str, Enum):
    """Observational scales within the DRR representation scope registry."""

    MACRO = "MACRO"
    MESO = "MESO"
    MICRO = "MICRO"
    LOCAL = "LOCAL"


POSITIVE_POLARITY_TERMS: Set[str] = {
    "stabilizing",
    "stable",
    "moderating",
    "expanding",
    "growing",
    "positive",
    "robust",
    "healthy",
    "strong",
}

NEGATIVE_POLARITY_TERMS: Set[str] = {
    "deteriorating",
    "contracting",
    "distressed",
    "negative",
    "declining",
    "fragile",
    "weak",
    "vulnerable",
}


@dataclass(frozen=True)
class ObservationalPerspective:
    """
    Strict, typed ledger entry carrying an observational perspective.

    Attributes:
        source_id: Identifier of the entity/model (e.g., 'Fed_Macroprudential_Model').
        scale: Observational scale enum value ['MACRO', 'MESO', 'MICRO', 'LOCAL'].
        confidence_score: Statistical confidence or model weight in range [0.0, 1.0].
        indicators: Key-value metrics (e.g., {'inflation_pressure': 'moderating'}).
        evidence_provenance: References to backing data streams or vintage hashes.
        dissent_logged: Flag indicating if alternative views exist within this perspective.
    """

    source_id: str
    scale: Union[ObservationalScale, str]
    confidence_score: float
    indicators: Dict[str, Any]
    evidence_provenance: str
    dissent_logged: bool = False

    def __post_init__(self) -> None:
        """Validate fields upon instantiation."""
        for name in ("source_id", "evidence_provenance"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.dissent_logged, bool):
            raise TypeError("dissent_logged must be a bool")
        # Normalize and validate scale
        scale_val = self.scale
        if isinstance(scale_val, ObservationalScale):
            scale_enum = scale_val
        elif isinstance(scale_val, str):
            try:
                scale_enum = ObservationalScale(scale_val.upper())
            except ValueError as err:
                valid_scales = [s.value for s in ObservationalScale]
                raise ValueError(
                    f"Invalid scale '{scale_val}'. Must be one of {valid_scales}"
                ) from err
        else:
            raise TypeError(
                f"Scale must be ObservationalScale or str, got {type(scale_val).__name__}"
            )

        # Override scale to normalized Enum using object.__setattr__ due to frozen=True
        object.__setattr__(self, "scale", scale_enum)

        # Validate confidence_score range
        if isinstance(self.confidence_score, bool) or not isinstance(
            self.confidence_score, (int, float)
        ):
            raise TypeError(
                f"confidence_score must be float or int, got {type(self.confidence_score).__name__}"
            )
        if not (0.0 <= float(self.confidence_score) <= 1.0):
            raise ValueError(
                f"confidence_score must be in range [0.0, 1.0], got {self.confidence_score}"
            )

        # Validate indicators and isolate mapping against external caller mutations
        if not isinstance(self.indicators, dict):
            raise TypeError(f"indicators must be a dict, got {type(self.indicators).__name__}")
        if any(not isinstance(k, str) or not k.strip() for k in self.indicators):
            raise ValueError("Indicator names must be nonempty strings")
        object.__setattr__(self, "indicators", deepcopy(self.indicators))


class DRR_ScopeResolver:
    """
    Scope Resolver mechanism that manages multiple instances of ObservationalPerspective.

    Enforces the critical invariant: High confidence_score or large scale (MACRO)
    MUST NOT silently overwrite, filter, or smooth out conflicting metrics from smaller
    scales (LOCAL/MICRO/MESO).
    """

    def __init__(self, perspectives: Optional[Sequence[ObservationalPerspective]] = None) -> None:
        """Initialize the scope resolver with an optional sequence of perspectives."""
        self._perspectives: List[ObservationalPerspective] = []
        if perspectives:
            for p in perspectives:
                self.register_perspective(p)

    def register_perspective(self, perspective: ObservationalPerspective) -> None:
        """
        Register an ObservationalPerspective into the scope registry.

        Preserves all perspectives in full fidelity without aggregation loss or erasure.
        """
        if not isinstance(perspective, ObservationalPerspective):
            raise TypeError(
                f"Expected ObservationalPerspective instance, got {type(perspective).__name__}"
            )
        self._perspectives.append(deepcopy(perspective))

    def add_perspective(self, perspective: ObservationalPerspective) -> None:
        """Alias for register_perspective."""
        self.register_perspective(perspective)

    @property
    def perspectives(self) -> List[ObservationalPerspective]:
        """Return a copy of the registered perspectives."""
        return deepcopy(self._perspectives)

    def _extract_state_string(
        self,
        target_scales: Sequence[ObservationalScale],
        default: str = "undetermined",
    ) -> str:
        """Helper to extract representative state strings for given scales."""
        matching_perspectives = [p for p in self._perspectives if p.scale in target_scales]
        if not matching_perspectives:
            return default

        states: List[str] = []
        for p in matching_perspectives:
            # A primary state label must not conceal a conflicting secondary
            # indicator in the human-readable summary either.
            states.extend(str(value) for value in p.indicators.values())

        if states:
            # Remove duplicate strings while preserving order
            unique_states = list(dict.fromkeys(states))
            return ", ".join(unique_states)
        return default

    @staticmethod
    def _get_text_polarity(text: str) -> Optional[str]:
        """Classify an exact descriptor, never infer sentiment from arbitrary prose.

        Negated, qualified, mixed and numeric indicators require human interpretation.
        The label is a lexical observation, not a claim about economic desirability.
        """
        descriptor = re.sub(r"[.,!?:;]+$", "", text.strip().lower()).strip()
        if descriptor in POSITIVE_POLARITY_TERMS:
            return "positive"
        if descriptor in NEGATIVE_POLARITY_TERMS:
            return "negative"
        return None

    def generate_drr_conclusion(self) -> Dict[str, Any]:
        """Preserve each indicator and report only the scopes actually supplied.

        Textual polarity cannot establish comparable populations, time horizons,
        indicator meanings, model accuracy or a substantive consensus.
        """
        assessments = [
            {
                "perspective_index": index,
                "source_id": p.source_id,
                "scale": p.scale.value,
                "indicator": key,
                "value": deepcopy(value),
                "polarity": self._get_text_polarity(value) if isinstance(value, str) else None,
            }
            for index, p in enumerate(self._perspectives)
            for key, value in p.indicators.items()
        ]
        # Compare individual indicators. Combining text at a scale first would erase
        # opposing observations whenever one scale contains both polarities.
        positive = [i for i, a in enumerate(assessments) if a["polarity"] == "positive"]
        negative = [i for i, a in enumerate(assessments) if a["polarity"] == "negative"]
        pairs = [
            {
                "positive_indicator": i,
                "negative_indicator": j,
                "cross_scale": assessments[i]["scale"] != assessments[j]["scale"],
            }
            for i in positive
            for j in negative
        ]
        has_divergence = bool(pairs)
        represented = [
            s for s in ObservationalScale if any(p.scale == s for p in self._perspectives)
        ]
        states = {s.value: self._extract_state_string([s]) for s in represented}
        macro_state = states.get("MACRO", "undetermined")
        local_state = self._extract_state_string(
            [ObservationalScale.LOCAL, ObservationalScale.MICRO]
        )
        unclassified = sum(a["polarity"] is None for a in assessments)
        fully_classified = (
            bool(assessments) and not unclassified and all(p.indicators for p in self._perspectives)
        )
        if has_divergence:
            cross_scale = any(pair["cross_scale"] for pair in pairs)
            status = (
                "divergent_scopes_preserved" if cross_scale else "divergent_perspectives_preserved"
            )
            conclusion_text = (
                "Opposing indicator descriptions are present in the supplied observations. "
                + "; ".join(f"{scale}: {state}" for scale, state in states.items())
                + ". Every indicator remains attributed in the ledger; confidence does not resolve the disagreement."
            )
        elif len(represented) >= 2 and fully_classified:
            # Retained for compatibility; this is lexical concordance, not a
            # verified consensus about the underlying system.
            status = "concordant_consensus"
            conclusion_text = (
                "Recognized indicator descriptions exhibit textual concordance across the supplied scopes. "
                "This does not establish agreement about the underlying system."
            )
        elif not self._perspectives:
            status = "no_perspectives"
            conclusion_text = "No observational perspectives registered in scope resolver."
        else:
            status = "indeterminate"
            conclusion_text = (
                "The supplied observations do not establish cross-scope agreement or divergence. "
                "Missing scopes and unclassified indicators remain unresolved."
            )
        macro = [p for p in self._perspectives if p.scale == ObservationalScale.MACRO]
        local = [
            p
            for p in self._perspectives
            if p.scale in (ObservationalScale.LOCAL, ObservationalScale.MICRO)
        ]

        def confidence_label(perspectives):
            return (
                f"{sum(p.confidence_score for p in perspectives) / len(perspectives):.2f}"
                if perspectives
                else "unavailable"
            )

        humility = {
            "accuracy_within_representation": (
                "Accuracy is not verified by this resolver. Confidence is supplied by the caller "
                f"(MACRO avg confidence: {confidence_label(macro)}, "
                f"LOCAL/MICRO avg confidence: {confidence_label(local)}). "
                "These descriptive averages neither validate evidence nor weight the conclusion."
            ),
            "completeness_of_representation": (
                "No single observational scale captures the whole system. Population, period, "
                "measurement meaning and evidence quality require separate review."
            ),
            "non_erasure_invariant_maintained": True,
            "dissent_logged_summary": {
                source: any(p.dissent_logged for p in self._perspectives if p.source_id == source)
                for source in dict.fromkeys(p.source_id for p in self._perspectives)
            },
            "dissent_records": [
                {
                    "perspective_index": i,
                    "source_id": p.source_id,
                    "dissent_logged": p.dissent_logged,
                }
                for i, p in enumerate(self._perspectives)
            ],
        }
        ledger = [
            {
                "source_id": p.source_id,
                "scale": p.scale.value,
                "confidence_score": p.confidence_score,
                "indicators": deepcopy(p.indicators),
                "evidence_provenance": p.evidence_provenance,
                "dissent_logged": p.dissent_logged,
            }
            for p in self._perspectives
        ]
        return {
            "status": status,
            "has_divergence": has_divergence,
            "conclusion": conclusion_text,
            "macro_state": macro_state,
            "local_state": local_state,
            "perspectives_evaluated": len(ledger),
            "registered_perspectives": ledger,
            "indicator_assessments": assessments,
            "divergence_pairs": pairs,
            "scope_coverage": {
                "represented": [s.value for s in represented],
                "missing": [s.value for s in ObservationalScale if s not in represented],
                "unclassified_indicators": unclassified,
                "empty_perspectives": sum(not p.indicators for p in self._perspectives),
            },
            "comparison_basis": "lexical descriptors only; cross-scope comparability is not established",
            "intellectual_humility": humility,
        }
