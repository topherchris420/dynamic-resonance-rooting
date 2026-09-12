"""Versioned MDRM semantics. Code suffixes never imply equivalence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from .common import day, instant, source_url, stable_id


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class MetricDefinition:
    code: str
    label: str
    form: str
    schedule: str
    line: str
    definition: str
    unit: str
    effective_from: str
    version: str
    source: str
    verification_date: str
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    effective_to: Optional[str] = None
    frequency: str = "quarterly"
    source_hash: str = ""
    provenance: str = "public MDRM and reporting instructions"
    transformation_status: str = "reported"
    known_as_of: Optional[str] = None
    domain: str = "balance_sheet"

    def __post_init__(self):
        object.__setattr__(self, "status", VerificationStatus(self.status))
        if not all((self.code, self.label, self.form, self.version, self.definition, self.unit)):
            raise ValueError(
                "Metric definition requires code, label, form, version, definition and unit"
            )
        day(self.effective_from)
        instant(self.verification_date)
        if self.effective_to and day(self.effective_to) < day(self.effective_from):
            raise ValueError("Metric effective dates are reversed")
        if self.status == VerificationStatus.VERIFIED:
            if not re.fullmatch(r"[A-Z]{4}[A-Z0-9]{4}", self.code):
                raise ValueError("Invalid MDRM code")
            source_url(self.source, authoritative=True)
            if (
                not self.schedule
                or not self.line
                or not re.fullmatch(r"[a-f0-9]{64}", self.source_hash)
            ):
                raise ValueError(
                    "Verified mapping requires sourced schedule, line and source SHA-256"
                )
        if self.status == VerificationStatus.SYNTHETIC and not self.code.startswith("SYN_"):
            raise ValueError("Synthetic metrics must use the SYN_ namespace")


class SemanticRegistry:
    """An immutable registry snapshot, explicitly provided by the analyst."""

    def __init__(self, definitions: Tuple[MetricDefinition, ...]):
        self.definitions = tuple(
            sorted(definitions, key=lambda d: (d.form, d.code, d.effective_from, d.version))
        )
        keys = [(d.form, d.code, d.version) for d in self.definitions]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate semantic version")

    @property
    def version(self) -> str:
        return stable_id(self.definitions)

    def resolve(
        self,
        form: str,
        code: str,
        period: str,
        *,
        as_of: str,
        version: Optional[str] = None,
        allow_synthetic: bool = False,
    ) -> MetricDefinition:
        matches = [
            d
            for d in self.definitions
            if d.form == form
            and d.code == code
            and day(d.effective_from) <= day(period)
            and (d.effective_to is None or day(period) <= day(d.effective_to))
            and (version is None or d.version == version)
            and instant(d.known_as_of or d.verification_date) <= instant(as_of)
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous as-of definition for {form}/{code}/{period}")
        definition = matches[0]
        if definition.status != VerificationStatus.VERIFIED and not (
            allow_synthetic and definition.status == VerificationStatus.SYNTHETIC
        ):
            raise ValueError(f"Unverified regulatory mapping blocked: {form}/{code}")
        return definition

    @classmethod
    def from_json(cls, path: str) -> "SemanticRegistry":
        return cls(tuple(MetricDefinition(**row) for row in json.loads(Path(path).read_text())))


def bundled_registry() -> SemanticRegistry:
    return SemanticRegistry.from_json(str(Path(__file__).with_name("data") / "mdrm_registry.json"))
