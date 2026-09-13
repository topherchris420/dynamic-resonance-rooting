"""
Disagreement Principle and Representation Scope Registry for DRR.

Implements multi-scale observational perspectives, non-erasure scope resolution,
and intellectual humility metadata for Dynamic Resonance Rooting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ObservationalScale(str, Enum):
    """Observational scopes for systemic and local perspectives."""

    MACRO = "MACRO"
    MESO = "MESO"
    MICRO = "MICRO"
    LOCAL = "LOCAL"


@dataclass
class ObservationalPerspective:
    """Strict, typed data structure representing a single observational perspective."""

    source_id: str
    scale: ObservationalScale | str
    confidence_score: float
    indicators: Dict[str, Any]
    evidence_provenance: str
    dissent_logged: bool = False

    def __post_init__(self) -> None:
        """Validate and coerce scale and confidence score fields."""
        if isinstance(self.scale, str):
            scale_str = self.scale.upper()
            try:
                self.scale = ObservationalScale(scale_str)
            except ValueError:
                valid_scales = [s.value for s in ObservationalScale]
                raise ValueError(
                    f"Invalid scale '{self.scale}'. Must be one of {valid_scales}"
                )
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"confidence_score must be in range [0.0, 1.0], got {self.confidence_score}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert perspective to a dictionary representation."""
        res = asdict(self)
        res["scale"] = (
            self.scale.value
            if isinstance(self.scale, ObservationalScale)
            else str(self.scale)
        )
        return res


class DRR_ScopeResolver:
    """
    Scope Resolver managing multiple incoming ObservationalPerspective instances.

    Guarantees the non-erasure invariant: High confidence scores or larger observational
    scales (e.g. MACRO) MUST NOT silently overwrite, filter, or smooth out conflicting
    metrics from smaller scales (e.g. LOCAL/MICRO).
    """

    # Indicators indicating positive / stabilizing / expanding state trends
    POSITIVE_INDICATORS: Set[str] = {
        "stabilizing",
        "moderating",
        "expanding",
        "robust",
        "growth",
        "resilient",
        "recovering",
        "stable",
        "positive",
    }

    # Indicators indicating negative / deteriorating / contracting state trends
    NEGATIVE_INDICATORS: Set[str] = {
        "deteriorating",
        "contracting",
        "distressed",
        "fragile",
        "declining",
        "strained",
        "vulnerable",
        "negative",
        "eroding",
    }

    def __init__(self) -> None:
        self._perspectives: List[ObservationalPerspective] = []

    def register_perspective(self, perspective: ObservationalPerspective) -> None:
        """
        Register an observational perspective in the ledger.

        Ensures full persistence without mutual erasure or scale-based filtering.
        """
        if not isinstance(perspective, ObservationalPerspective):
            raise TypeError(
                "Expected perspective to be an instance of ObservationalPerspective"
            )
        self._perspectives.append(perspective)

    @property
    def perspectives(self) -> List[ObservationalPerspective]:
        """Retrieve all registered perspectives in the ledger."""
        return list(self._perspectives)

    def clear(self) -> None:
        """Clear all registered perspectives from the resolver."""
        self._perspectives.clear()

    def _extract_state_summary(
        self, perspectives: List[ObservationalPerspective]
    ) -> str:
        """Extract a natural language state summary from a list of perspectives."""
        states: List[str] = []
        for p in perspectives:
            for val in p.indicators.values():
                val_str = str(val).strip().lower()
                if val_str and val_str not in [s.lower() for s in states]:
                    states.append(str(val))
        return ", ".join(states) if states else "evaluated state"

    def _detect_divergence(
        self,
    ) -> Tuple[
        bool,
        Optional[ObservationalPerspective],
        Optional[ObservationalPerspective],
    ]:
        """
        Inspect indicators across perspectives to detect orthogonal or divergent states
        between macro/meso scales and local/micro scales.
        """
        macro_perspectives = [
            p
            for p in self._perspectives
            if p.scale in (ObservationalScale.MACRO, ObservationalScale.MESO)
        ]
        local_perspectives = [
            p
            for p in self._perspectives
            if p.scale in (ObservationalScale.LOCAL, ObservationalScale.MICRO)
        ]

        if not macro_perspectives or not local_perspectives:
            return False, None, None

        for macro_p in macro_perspectives:
            macro_vals = {str(v).lower() for v in macro_p.indicators.values()}
            for local_p in local_perspectives:
                local_vals = {str(v).lower() for v in local_p.indicators.values()}

                # Check for explicit positive vs negative divergence
                has_macro_pos = bool(macro_vals & self.POSITIVE_INDICATORS)
                has_macro_neg = bool(macro_vals & self.NEGATIVE_INDICATORS)
                has_local_pos = bool(local_vals & self.POSITIVE_INDICATORS)
                has_local_neg = bool(local_vals & self.NEGATIVE_INDICATORS)

                if (has_macro_pos and has_local_neg) or (
                    has_macro_neg and has_local_pos
                ):
                    return True, macro_p, local_p

                # Check for shared keys with contradictory direct value strings
                shared_keys = set(macro_p.indicators.keys()) & set(
                    local_p.indicators.keys()
                )
                if any(
                    str(macro_p.indicators[k]).lower()
                    != str(local_p.indicators[k]).lower()
                    for k in shared_keys
                ):
                    return True, macro_p, local_p

        return False, None, None

    def generate_drr_conclusion(self) -> Dict[str, Any]:
        """
        Evaluate registered perspectives and synthesize a DRR conclusion.

        If divergent trends are detected across scales, the engine deliberately refuses
        to force a single consensus state. It returns a structured payload adhering to the
        Disagreement Principle and embodying intellectual humility.
        """
        is_divergent, macro_p, local_p = self._detect_divergence()

        macro_perspectives = [
            p
            for p in self._perspectives
            if p.scale in (ObservationalScale.MACRO, ObservationalScale.MESO)
        ]
        local_perspectives = [
            p
            for p in self._perspectives
            if p.scale in (ObservationalScale.LOCAL, ObservationalScale.MICRO)
        ]

        if is_divergent and macro_p is not None and local_p is not None:
            macro_state = self._extract_state_summary([macro_p])
            local_state = self._extract_state_summary([local_p])

            conclusion_text = (
                f"Aggregate financial-system indicators support {macro_state}, "
                f"while material indicators for the evaluated population support {local_state}. "
                "These findings operate at different observational scopes and should be "
                "interpreted together rather than collapsed into a single state."
            )

            payload = {
                "status": "DIVERGENCE_DETECTED",
                "forced_consensus_rejected": True,
                "conclusion": conclusion_text,
                "macro_state": macro_state,
                "local_state": local_state,
                "registered_perspectives": [p.to_dict() for p in self._perspectives],
                "divergence_summary": {
                    "macro_source": macro_p.source_id,
                    "macro_scale": macro_p.scale.value,
                    "macro_confidence": macro_p.confidence_score,
                    "local_source": local_p.source_id,
                    "local_scale": local_p.scale.value,
                    "local_confidence": local_p.confidence_score,
                },
                "metadata": {
                    "accuracy_within_representation": {
                        "macro_confidence": macro_p.confidence_score,
                        "local_confidence": local_p.confidence_score,
                        "statement": (
                            "High internal statistical validity within individual perspective bounds. "
                            "Each model accurately measures its respective observational domain."
                        ),
                    },
                    "completeness_of_representation": {
                        "is_complete": False,
                        "statement": (
                            "No single observational scope possesses complete representation of total system dynamics. "
                            "Aggregate macro metrics omit local distributional variance, while local metrics do not capture macro-prudential liquidity buffers."
                        ),
                    },
                    "disagreement_principle_applied": True,
                },
            }
            return payload

        # Consensus or single-scale output handling
        macro_state = (
            self._extract_state_summary(macro_perspectives)
            if macro_perspectives
            else "aligned"
        )
        local_state = (
            self._extract_state_summary(local_perspectives)
            if local_perspectives
            else "aligned"
        )
        combined_state = macro_state if macro_perspectives else local_state

        conclusion_text = (
            f"Observational perspectives across evaluated scopes are consistent with {combined_state}."
        )

        return {
            "status": "CONSENSUS",
            "forced_consensus_rejected": False,
            "conclusion": conclusion_text,
            "registered_perspectives": [p.to_dict() for p in self._perspectives],
            "metadata": {
                "accuracy_within_representation": {
                    "statement": "High internal statistical validity across registered perspectives."
                },
                "completeness_of_representation": {
                    "is_complete": True if len(self._perspectives) > 1 else False,
                    "statement": (
                        "Perspectives show alignment across evaluated scopes, but representation remains bounded by registered data sources."
                    ),
                },
                "disagreement_principle_applied": False,
            },
        }
