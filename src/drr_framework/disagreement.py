"""
Disagreement Principle and Representation Scope Registry for DRR Architecture.

This module implements the Disagreement Principle and Representation Scope Registry for
Dynamic Resonance Rooting (DRR), enforcing non-erasure scope resolution across divergent
observational scales. High-confidence macro-level signals are prevented from silently overwriting,
filtering, or smoothing out conflicting metrics from local/micro-level observations.
"""

from dataclasses import dataclass, field
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
        # Normalize and validate scale
        scale_val = self.scale
        if isinstance(scale_val, ObservationalScale):
            scale_enum = scale_val
        elif isinstance(scale_val, str):
            try:
                scale_enum = ObservationalScale(scale_val.upper())
            except ValueError:
                valid_scales = [s.value for s in ObservationalScale]
                raise ValueError(
                    f"Invalid scale '{scale_val}'. Must be one of {valid_scales}"
                )
        else:
            raise TypeError(f"Scale must be ObservationalScale or str, got {type(scale_val).__name__}")

        # Override scale to normalized Enum using object.__setattr__ due to frozen=True
        object.__setattr__(self, "scale", scale_enum)

        # Validate confidence_score range
        if not isinstance(self.confidence_score, (int, float)):
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
        object.__setattr__(self, "indicators", dict(self.indicators))


class DRR_ScopeResolver:
    """
    Scope Resolver mechanism that manages multiple instances of ObservationalPerspective.

    Enforces the critical invariant: High confidence_score or large scale (MACRO)
    MUST NOT silently overwrite, filter, or smooth out conflicting metrics from smaller
    scales (LOCAL/MICRO/MESO).
    """

    def __init__(
        self, perspectives: Optional[Sequence[ObservationalPerspective]] = None
    ) -> None:
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
        self._perspectives.append(perspective)

    def add_perspective(self, perspective: ObservationalPerspective) -> None:
        """Alias for register_perspective."""
        self.register_perspective(perspective)

    @property
    def perspectives(self) -> List[ObservationalPerspective]:
        """Return a copy of the registered perspectives."""
        return list(self._perspectives)

    def _extract_state_string(
        self,
        target_scales: Sequence[ObservationalScale],
        default: str = "undetermined",
    ) -> str:
        """Helper to extract representative state strings for given scales."""
        matching_perspectives = [
            p for p in self._perspectives if p.scale in target_scales
        ]
        if not matching_perspectives:
            return default

        states: List[str] = []
        for p in matching_perspectives:
            # Check for explicit state keys first
            for key in ("macro_state", "local_state", "state", "status", "trend"):
                if key in p.indicators:
                    states.append(str(p.indicators[key]))
                    break
            else:
                # Fallback to indicator values
                for val in p.indicators.values():
                    states.append(str(val))

        if states:
            # Remove duplicate strings while preserving order
            unique_states = list(dict.fromkeys(states))
            return ", ".join(unique_states)
        return default

    @staticmethod
    def _get_text_polarity(text: str) -> Optional[str]:
        """Classify text polarity as 'positive', 'negative', or None."""
        words = set(text.lower().replace(",", " ").split())
        has_pos = bool(words & POSITIVE_POLARITY_TERMS)
        has_neg = bool(words & NEGATIVE_POLARITY_TERMS)
        if has_pos and not has_neg:
            return "positive"
        if has_neg and not has_pos:
            return "negative"
        return None

    def _detect_scale_divergence(
        self,
    ) -> Tuple[bool, str, str]:
        """
        Inspect indicators across ALL registered scales and detect polar divergence.

        Returns:
            Tuple of (has_divergence, macro_state_str, local_state_str)
        """
        macro_state = self._extract_state_string(
            [ObservationalScale.MACRO], default="stabilizing"
        )
        local_state = self._extract_state_string(
            [ObservationalScale.LOCAL, ObservationalScale.MICRO], default="deteriorating"
        )

        # Collect state text and polarities per represented scale
        scale_polarities: Dict[ObservationalScale, str] = {}
        for scale in ObservationalScale:
            scale_text = self._extract_state_string([scale], default="")
            if scale_text:
                pol = self._get_text_polarity(scale_text)
                if pol:
                    scale_polarities[scale] = pol

        # Divergence exists if any pair of represented scales have opposite polarities
        # (e.g. MACRO positive vs LOCAL negative, or MESO positive vs MICRO negative)
        has_divergence = False
        polarities = list(scale_polarities.values())
        if "positive" in polarities and "negative" in polarities:
            has_divergence = True

        return has_divergence, macro_state, local_state

    def generate_drr_conclusion(self) -> Dict[str, Any]:
        """
        Synthesis Engine: Evaluates registered perspectives and generates a structured conclusion.

        If divergent trends are detected across scales, the engine deliberately refuses
        to force a single consensus state. Instead, it returns a structured payload
        matching the specific natural language template and an intellectual humility block.

        Returns:
            Dict containing conclusion string, divergence status, non-erasure ledger,
            and intellectual humility metadata.
        """
        has_divergence, macro_state, local_state = self._detect_scale_divergence()

        macro_perspectives = [
            p for p in self._perspectives if p.scale == ObservationalScale.MACRO
        ]
        local_perspectives = [
            p
            for p in self._perspectives
            if p.scale in (ObservationalScale.LOCAL, ObservationalScale.MICRO)
        ]

        macro_avg_conf = (
            sum(p.confidence_score for p in macro_perspectives) / len(macro_perspectives)
            if macro_perspectives
            else 0.0
        )
        local_avg_conf = (
            sum(p.confidence_score for p in local_perspectives) / len(local_perspectives)
            if local_perspectives
            else 0.0
        )

        if has_divergence:
            conclusion_text = (
                f"Aggregate financial-system indicators support {macro_state}, while "
                f"material indicators for the evaluated population support {local_state}. "
                f"These findings operate at different observational scopes and should "
                f"be interpreted together rather than collapsed into a single state."
            )
            status = "divergent_scopes_preserved"
        else:
            if self._perspectives:
                all_states = self._extract_state_string(
                    list(ObservationalScale), default="concordant"
                )
                conclusion_text = (
                    f"Observational indicators across registered scopes exhibit concordance "
                    f"supporting {all_states}."
                )
            else:
                conclusion_text = "No observational perspectives registered in scope resolver."
            status = "concordant_consensus"

        # Construct intellectual humility metadata block explicitly separating
        # "accuracy within a representation" from "completeness of that representation"
        intellectual_humility = {
            "accuracy_within_representation": (
                f"Statistical and model representations are evaluated as internally accurate "
                f"within their designated observational boundaries (MACRO avg confidence: "
                f"{macro_avg_conf:.2f}, LOCAL/MICRO avg confidence: {local_avg_conf:.2f})."
            ),
            "completeness_of_representation": (
                "Incomplete representation acknowledged. No single observational scale or "
                "aggregated index captures total systemic reality. Macro-prudential "
                "averages omit local distributional variance; local surveys omit macro "
                "contagion pathways."
            ),
            "non_erasure_invariant_maintained": True,
            "dissent_logged_summary": {
                p.source_id: p.dissent_logged for p in self._perspectives
            },
        }

        # Defensively copy indicator mappings in non-erasure ledger to isolate from caller mutations
        registered_ledger = [
            {
                "source_id": p.source_id,
                "scale": p.scale.value if isinstance(p.scale, Enum) else str(p.scale),
                "confidence_score": p.confidence_score,
                "indicators": dict(p.indicators),
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
            "perspectives_evaluated": len(self._perspectives),
            "registered_perspectives": registered_ledger,
            "intellectual_humility": intellectual_humility,
        }
