"""Normalize a public filing export without inventing MDRM mappings or filing dates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .common import file_sha256, instant, sha256_hex, source_url
from .vintage import RegulatoryObservation, ObservationProvenance, VintageStore


@dataclass(frozen=True)
class FilingContext:
    form: str
    reporting_period: str
    original_filing_date: str
    ingestion_date: str
    available_as_of: str
    source_vintage: str
    source: str
    source_hash: str
    definition_version: str
    amendment_date: Optional[str] = None
    perimeter_version: str = "unspecified"

    def __post_init__(self):
        source_url(self.source)
        object.__setattr__(self, "source_hash", sha256_hex(self.source_hash))
        instant(self.available_as_of)


def ingest_wide_filing(
    frame,
    registry,
    context,
    *,
    institution_column="RSSD_ID",
    name_column="LEGAL_NAME",
    metric_columns=None,
):
    """Import exact MDRM-code columns from a local public CSV/DataFrame.

    A caller supplies actual filing/vintage metadata. Unsupported financial
    columns are rejected when explicitly selected; suffixes are never matched.
    Ingestion is atomic: one invalid record rejects the batch for reconciliation.
    """
    source_url(context.source)
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("A nonempty filing with unique column names is required")
    if institution_column not in frame or name_column not in frame:
        raise ValueError("Institution identifier/name columns are required")
    metrics = (
        tuple(metric_columns)
        if metric_columns is not None
        else tuple(c for c in frame.columns if c not in {institution_column, name_column})
    )
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Select distinct MDRM-code columns explicitly")
    definitions = {
        m: registry.resolve(
            context.form,
            m,
            context.reporting_period,
            as_of=context.available_as_of,
            version=context.definition_version,
        )
        for m in metrics
    }
    rows = []
    for _, record in frame.iterrows():
        if pd.isna(record[institution_column]) or pd.isna(record[name_column]):
            raise ValueError("Institution identifier/name is missing")
        institution = str(record[institution_column])
        if institution.endswith(".0"):
            raise ValueError(
                "Load identifiers as strings; floating-point identifiers are not accepted"
            )
        for metric in metrics:
            value = None if pd.isna(record[metric]) else float(record[metric])
            rows.append(
                RegulatoryObservation(
                    institution_id=institution,
                    institution_name=str(record[name_column]),
                    form=context.form,
                    metric=metric,
                    reporting_period=context.reporting_period,
                    value=value,
                    unit=definitions[metric].unit,
                    original_filing_date=context.original_filing_date,
                    ingestion_date=context.ingestion_date,
                    source_vintage=context.source_vintage,
                    available_as_of=context.available_as_of,
                    definition_version=context.definition_version,
                    source=context.source,
                    provenance=ObservationProvenance.RAW_REPORTED,
                    amendment_date=context.amendment_date,
                    perimeter_version=context.perimeter_version,
                    source_hash=context.source_hash,
                )
            )
    store = VintageStore(rows)
    from .reconciliation import reconcile_store

    exceptions = reconcile_store(store, context.available_as_of)
    if exceptions:
        raise ValueError(
            "Duplicate/conflicting filing rows require reconciliation before ingestion"
        )
    return store


def ingest_wide_csv(path, registry, context, **kwargs):
    expected_hash = context.source_hash
    if file_sha256(path) != expected_hash:
        raise ValueError("Local filing content does not match FilingContext source_hash")
    frame = pd.read_csv(path, dtype={kwargs.get("institution_column", "RSSD_ID"): str})
    # Hash again after parsing so a concurrent replacement cannot pair one
    # artifact's bytes with another artifact's normalized rows.
    if file_sha256(path) != expected_hash:
        raise ValueError("Local filing content changed while it was being read")
    return ingest_wide_filing(
        frame,
        registry,
        context,
        **kwargs,
    )
