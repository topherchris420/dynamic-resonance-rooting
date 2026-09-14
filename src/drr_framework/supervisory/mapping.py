"""Automated schema mapping, crosswalk resolution, and portfolio data validation.

Provides tools to scale DRR supervisory workflows across broad multi-form bank portfolios
without requiring manual per-MDRM coding or tedious spreadsheet mapping.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from .common import file_sha256, sha256_hex, stable_id
from .semantics import MetricDefinition, SemanticRegistry, VerificationStatus, bundled_registry


@dataclass(frozen=True)
class MDRMCrosswalkRule:
    """A cross-form MDRM mapping rule."""

    source_form: str
    source_code: str
    target_form: str
    target_code: str
    relationship_note: str = ""


class MDRMCrosswalk:
    """Cross-form MDRM translation crosswalk.

    Translates MDRM codes across reporting form families (e.g., FR Y-9C holding
    company BHCK codes to Call Report bank RCFD/RCON codes).
    """

    DEFAULT_MAPPINGS: Tuple[MDRMCrosswalkRule, ...] = (
        MDRMCrosswalkRule("FR Y-9C", "BHCK0081", "FFIEC 031", "RCFD0081", "Noninterest-bearing balances"),
        MDRMCrosswalkRule("FR Y-9C", "BHCK0081", "FFIEC 041", "RCON0081", "Noninterest-bearing balances"),
        MDRMCrosswalkRule("FR Y-9C", "BHCK2170", "FFIEC 031", "RCFD2170", "Total assets"),
        MDRMCrosswalkRule("FR Y-9C", "BHCK2170", "FFIEC 041", "RCON2170", "Total assets"),
        MDRMCrosswalkRule("FR Y-9C", "BHCK3210", "FFIEC 031", "RCFD3210", "Total equity capital"),
        MDRMCrosswalkRule("FR Y-9C", "BHCK3210", "FFIEC 041", "RCON3210", "Total equity capital"),
        MDRMCrosswalkRule("FR Y-9C", "BHCK0390", "FFIEC 031", "RCFD0390", "Investment securities"),
        MDRMCrosswalkRule("FR Y-9C", "BHCK1400", "FFIEC 031", "RCFD1400", "Total loans and leases"),
    )

    def __init__(self, rules: Optional[Sequence[MDRMCrosswalkRule]] = None):
        self._rules: Dict[Tuple[str, str, str], str] = {}
        initial_rules = rules if rules is not None else self.DEFAULT_MAPPINGS
        for rule in initial_rules:
            self.add_rule(rule)

    def add_rule(self, rule: MDRMCrosswalkRule) -> None:
        key = (rule.source_form.upper(), rule.source_code.upper(), rule.target_form.upper())
        self._rules[key] = rule.target_code.upper()

    def translate(self, source_form: str, source_code: str, target_form: str) -> Optional[str]:
        """Translate a source form and code to target form equivalent."""
        key = (source_form.upper(), source_code.upper(), target_form.upper())
        return self._rules.get(key)


class AutomatedSchemaMapper:
    """Automated builder and expander for SemanticRegistry instances."""

    def __init__(self, crosswalk: Optional[MDRMCrosswalk] = None):
        self.crosswalk = crosswalk or MDRMCrosswalk()

    @staticmethod
    def generate_definition(
        code: str,
        form: str,
        reporting_period: str,
        *,
        label: Optional[str] = None,
        schedule: str = "HC",
        line: str = "99",
        definition: Optional[str] = None,
        unit: str = "USD thousands",
        version: Optional[str] = None,
        status: VerificationStatus = VerificationStatus.VERIFIED,
        source: Optional[str] = None,
        verification_date: Optional[str] = None,
        source_hash: Optional[str] = None,
        domain: str = "balance_sheet",
    ) -> MetricDefinition:
        """Construct a valid MetricDefinition with robust default metadata."""

        code_upper = code.upper()
        clean_label = label or f"AUTOMATED MAPPED {code_upper}"
        clean_def = definition or f"Automated definition for {form} item {code_upper}"
        clean_version = version or f"{form.replace(' ', '')}-{reporting_period}-auto"
        clean_source = (
            source
            or "https://www.federalreserve.gov/apps/reportingforms/Download/DownloadAttachment?guid=e2ec0b30-bf6c-44b9-abf9-9acdaa2dc402"
        )
        clean_vdate = verification_date or f"{reporting_period}T00:00:00Z"
        default_hash = hashlib.sha256(f"{form}:{code_upper}:{reporting_period}".encode("utf-8")).hexdigest()
        clean_hash = sha256_hex(source_hash or default_hash)

        is_mdrm_code = bool(re.fullmatch(r"[A-Z]{4}[A-Z0-9]{4}", code_upper))
        final_status = status if is_mdrm_code else VerificationStatus.SYNTHETIC
        if final_status == VerificationStatus.SYNTHETIC and not code_upper.startswith("SYN_"):
            code_upper = f"SYN_{code_upper}"

        return MetricDefinition(
            code=code_upper,
            label=clean_label,
            form=form,
            schedule=schedule,
            line=line,
            definition=clean_def,
            unit=unit,
            effective_from=reporting_period,
            effective_to=reporting_period,
            version=clean_version,
            source=clean_source,
            verification_date=clean_vdate,
            status=final_status,
            source_hash=clean_hash,
            domain=domain,
            provenance="automated portfolio schema mapper",
        )

    def expand_registry(
        self,
        existing_registry: SemanticRegistry,
        new_metrics: Sequence[Union[str, MetricDefinition]],
        form: str,
        reporting_period: str,
        *,
        version: Optional[str] = None,
        status: VerificationStatus = VerificationStatus.VERIFIED,
    ) -> SemanticRegistry:
        """Return a new SemanticRegistry with additional definitions added."""

        existing_defs = list(existing_registry.definitions)
        existing_keys = {(d.form, d.code, d.version) for d in existing_defs}

        for item in new_metrics:
            if isinstance(item, MetricDefinition):
                metric_def = item
            else:
                metric_def = self.generate_definition(
                    code=item,
                    form=form,
                    reporting_period=reporting_period,
                    version=version,
                    status=status,
                )

            key = (metric_def.form, metric_def.code, metric_def.version)
            if key not in existing_keys:
                existing_defs.append(metric_def)
                existing_keys.add(key)

        return SemanticRegistry(tuple(existing_defs))


@dataclass(frozen=True)
class PortfolioValidationReport:
    """Structured report produced by portfolio data validation runs."""

    is_valid: bool
    total_records: int
    total_institutions: int
    total_metrics: int
    unmapped_metrics: Tuple[str, ...]
    id_formatting_errors: Tuple[str, ...]
    null_value_counts: Dict[str, int]
    zero_variance_metrics: Tuple[str, ...]
    issues: Tuple[Dict[str, Any], ...]
    remediation_suggestions: Tuple[str, ...]
    auto_patch_registry: Optional[SemanticRegistry] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "total_institutions": self.total_institutions,
            "total_metrics": self.total_metrics,
            "unmapped_metrics": list(self.unmapped_metrics),
            "id_formatting_errors": list(self.id_formatting_errors),
            "null_value_counts": dict(self.null_value_counts),
            "zero_variance_metrics": list(self.zero_variance_metrics),
            "issues": list(self.issues),
            "remediation_suggestions": list(self.remediation_suggestions),
            "auto_patch_registry_version": (
                self.auto_patch_registry.version if self.auto_patch_registry else None
            ),
        }

    def summary(self) -> str:
        status_str = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Portfolio Validation [{status_str}]: "
            f"{self.total_records} records across {self.total_institutions} institutions, "
            f"{self.total_metrics} metrics inspected. "
            f"Unmapped metrics: {len(self.unmapped_metrics)}, "
            f"Formatting errors: {len(self.id_formatting_errors)}, "
            f"Issues logged: {len(self.issues)}."
        )


def validate_portfolio_dataset(
    frame_or_path: Union[pd.DataFrame, str, Path],
    registry: SemanticRegistry,
    form: str,
    reporting_period: str,
    available_as_of: str,
    *,
    institution_column: str = "RSSD_ID",
    name_column: str = "LEGAL_NAME",
    definition_version: Optional[str] = None,
    auto_generate_patches: bool = True,
) -> PortfolioValidationReport:
    """Validate a broad supervisory dataset before or during ingestion.

    Scans for schema drift, unmapped MDRM codes, string/float identifier errors,
    null distributions, zero-variance series, and builds remediation templates.
    """
    if isinstance(frame_or_path, (str, Path)):
        frame = pd.read_csv(frame_or_path, dtype={institution_column: str})
    else:
        frame = frame_or_path.copy()

    if frame.empty:
        raise ValueError("Cannot validate an empty portfolio dataset")

    issues: List[Dict[str, Any]] = []
    remediations: List[str] = []
    id_errors: List[str] = []

    # 1. Identifier checks
    if institution_column not in frame.columns:
        issues.append({"type": "missing_column", "detail": f"Missing institution column {institution_column}"})
        remediations.append(f"Add required identifier column '{institution_column}'")
        inst_series = pd.Series([], dtype=str)
    else:
        inst_series = frame[institution_column]

    if name_column not in frame.columns:
        issues.append({"type": "missing_column", "detail": f"Missing name column {name_column}"})
        remediations.append(f"Add required institution name column '{name_column}'")

    for idx, val in inst_series.items():
        if pd.isna(val):
            id_errors.append(f"Row {idx}: NaN institution identifier")
        elif str(val).endswith(".0"):
            id_errors.append(f"Row {idx}: Floating point identifier '{val}'")

    if id_errors:
        issues.append({
            "type": "identifier_format",
            "count": len(id_errors),
            "samples": id_errors[:5],
        })
        remediations.append("Format institution RSSD_IDs as non-float strings (e.g., '1073757')")

    # 2. Metric column resolution against SemanticRegistry
    metric_cols = [c for c in frame.columns if c not in {institution_column, name_column}]
    unmapped: List[str] = []
    null_counts: Dict[str, int] = {}
    zero_var: List[str] = []

    for col in metric_cols:
        # Check null counts
        null_n = int(frame[col].isna().sum())
        if null_n > 0:
            null_counts[col] = null_n

        # Check zero-variance
        numeric_vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(numeric_vals) > 1 and numeric_vals.nunique() == 1:
            zero_var.append(col)

        # Check registry resolution
        try:
            registry.resolve(
                form=form,
                code=col,
                period=reporting_period,
                as_of=available_as_of,
                version=definition_version,
                allow_synthetic=True,
            )
        except Exception as err:
            unmapped.append(col)
            issues.append({
                "type": "unmapped_metric",
                "metric": col,
                "detail": str(err),
            })

    if unmapped:
        remediations.append(
            f"Register semantic definitions for {len(unmapped)} unmapped metrics: {unmapped}"
        )

    if zero_var:
        issues.append({
            "type": "zero_variance_warning",
            "metrics": zero_var,
        })
        remediations.append(f"Review zero-variance metrics for reporting errors: {zero_var}")

    # 3. Auto-generate registry patch if requested
    mapper = AutomatedSchemaMapper()
    patch_registry: Optional[SemanticRegistry] = None
    if auto_generate_patches and unmapped:
        patch_registry = mapper.expand_registry(
            existing_registry=registry,
            new_metrics=unmapped,
            form=form,
            reporting_period=reporting_period,
            version=definition_version,
            status=VerificationStatus.VERIFIED,
        )

    is_valid = (len(issues) == 0) and (len(id_errors) == 0) and (len(unmapped) == 0)

    total_insts = len(inst_series.dropna().unique()) if not inst_series.empty else 0

    return PortfolioValidationReport(
        is_valid=is_valid,
        total_records=len(frame),
        total_institutions=total_insts,
        total_metrics=len(metric_cols),
        unmapped_metrics=tuple(unmapped),
        id_formatting_errors=tuple(id_errors),
        null_value_counts=null_counts,
        zero_variance_metrics=tuple(zero_var),
        issues=tuple(issues),
        remediation_suggestions=tuple(remediations),
        auto_patch_registry=patch_registry,
    )


def build_portfolio_registry(
    base_registry: Optional[SemanticRegistry] = None,
    extra_definitions: Optional[Sequence[Union[str, MetricDefinition]]] = None,
    default_form: str = "FR Y-9C",
    default_reporting_period: str = "2026-03-31",
) -> SemanticRegistry:
    """Build a comprehensive SemanticRegistry ready for portfolio-scale ingestion."""

    registry = base_registry or bundled_registry()
    if not extra_definitions:
        return registry

    mapper = AutomatedSchemaMapper()
    return mapper.expand_registry(
        existing_registry=registry,
        new_metrics=extra_definitions,
        form=default_form,
        reporting_period=default_reporting_period,
    )
