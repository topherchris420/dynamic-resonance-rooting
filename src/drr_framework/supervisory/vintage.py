"""Append-only public observations and conservative point-in-time reconstruction."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from .common import canonical, canonical_json, day, instant, source_url, stable_id
from .semantics import SemanticRegistry


class ObservationProvenance(str, Enum):
    RAW_REPORTED = "raw_reported"
    IMPUTED_CAUSAL = "imputed_causal"
    DERIVED_COMPUTED = "derived_computed"
    EXTERNAL_REFERENCE = "external_reference"
    ANALYST_ENTERED = "analyst_entered"
    SYNTHETIC_TEST = "synthetic_test"


@dataclass(frozen=True)
class CalculationLineage:
    formula: str
    input_ids: Tuple[str, ...]
    input_metrics: Tuple[str, ...]
    input_periods: Tuple[str, ...]
    software_version: str
    parameters: Tuple[Tuple[str, str], ...]
    unit: str
    timestamp: str

    def __post_init__(self):
        for name in ("input_ids", "input_metrics", "input_periods"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "parameters", tuple(tuple(p) for p in self.parameters))
        if not self.formula or not self.input_ids or not self.software_version or not self.unit:
            raise ValueError("Derived values require full calculation lineage")
        if len(self.input_ids) != len(self.input_metrics) or len(self.input_ids) != len(
            self.input_periods
        ):
            raise ValueError("Lineage dimensions must match")
        instant(self.timestamp)


@dataclass(frozen=True)
class RegulatoryObservation:
    institution_id: str
    institution_name: str
    form: str
    metric: str
    reporting_period: str
    value: Optional[float]
    unit: str
    original_filing_date: str
    ingestion_date: str
    source_vintage: str
    available_as_of: str
    definition_version: str
    source: str
    provenance: ObservationProvenance = ObservationProvenance.RAW_REPORTED
    amendment_date: Optional[str] = None
    supersedes: Optional[str] = None
    perimeter_version: str = "unspecified"
    lineage: Optional[CalculationLineage] = None

    def __post_init__(self):
        object.__setattr__(self, "provenance", ObservationProvenance(self.provenance))
        object.__setattr__(self, "reporting_period", day(self.reporting_period))
        for name in ("original_filing_date", "ingestion_date", "available_as_of", "amendment_date"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, instant(getattr(self, name)).isoformat())
        if not all(
            (
                self.institution_id,
                self.institution_name,
                self.form,
                self.metric,
                self.unit,
                self.source_vintage,
                self.definition_version,
                self.source,
            )
        ):
            raise ValueError("Observation identity, units, vintage and source are required")
        if self.value is not None:
            if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
                raise ValueError(
                    "Missing observations must be null; infinities and booleans are invalid"
                )
            object.__setattr__(self, "value", float(self.value))
        if instant(self.original_filing_date) < instant(self.reporting_period):
            raise ValueError("A filing cannot precede its reporting period")
        if self.amendment_date and instant(self.amendment_date) < instant(
            self.original_filing_date
        ):
            raise ValueError("Amendment predates original filing")
        if instant(self.available_as_of) < max(
            instant(self.ingestion_date), instant(self.filing_date)
        ):
            raise ValueError("Availability cannot precede ingestion or filing")
        derived = self.provenance in {
            ObservationProvenance.DERIVED_COMPUTED,
            ObservationProvenance.IMPUTED_CAUSAL,
        }
        if derived and self.lineage is None:
            raise ValueError("Derived and imputed observations require lineage")
        if self.lineage and instant(self.lineage.timestamp) > instant(self.available_as_of):
            raise ValueError("Calculation is unavailable until computed")
        if self.provenance == ObservationProvenance.SYNTHETIC_TEST:
            if not self.metric.startswith("SYN_"):
                raise ValueError("Synthetic values require SYN_ metric codes")
        elif self.provenance in {
            ObservationProvenance.RAW_REPORTED,
            ObservationProvenance.EXTERNAL_REFERENCE,
        }:
            source_url(self.source)

    @property
    def filing_date(self):
        return self.amendment_date or self.original_filing_date

    @property
    def key(self):
        return self.institution_id, self.form, self.metric, self.reporting_period

    @property
    def observation_id(self):
        return stable_id(self)

    @classmethod
    def from_dict(cls, row):
        values = dict(row)
        values.pop("observation_id", None)
        if isinstance(values.get("lineage"), dict):
            values["lineage"] = CalculationLineage(**values["lineage"])
        if values.get("value") == "":
            values["value"] = None
        for name in ("amendment_date", "supersedes", "lineage"):
            if values.get(name) == "":
                values[name] = None
        return cls(**values)


@dataclass(frozen=True)
class FilingRevision:
    previous: RegulatoryObservation
    current: RegulatoryObservation
    raw_change: Optional[float]


class VintageStore:
    """Immutable snapshots; appending returns a new store, never mutates history."""

    def __init__(self, observations=()):
        self.observations = tuple(
            sorted(observations, key=lambda o: (o.key, o.filing_date, o.observation_id))
        )
        by_id = {o.observation_id: o for o in self.observations}
        for o in self.observations:
            if o.supersedes:
                parent = by_id.get(o.supersedes)
                if (
                    parent is None
                    or parent.key != o.key
                    or instant(parent.filing_date) >= instant(o.filing_date)
                ):
                    raise ValueError(
                        "Superseded observation must be an earlier filing of the same key"
                    )
            if o.lineage:
                for oid, metric, period in zip(
                    o.lineage.input_ids, o.lineage.input_metrics, o.lineage.input_periods
                ):
                    parent = by_id.get(oid)
                    if (
                        parent is None
                        or parent.metric != metric
                        or parent.reporting_period != day(period)
                    ):
                        raise ValueError("Lineage references a missing or mismatched observation")
                    if instant(parent.available_as_of) > instant(o.available_as_of):
                        raise ValueError("Derived observation leaks future input")
                    if o.provenance == ObservationProvenance.IMPUTED_CAUSAL and (
                        parent.institution_id != o.institution_id
                        or parent.form != o.form
                        or parent.metric != o.metric
                        or parent.reporting_period >= o.reporting_period
                        or parent.definition_version != o.definition_version
                        or parent.unit != o.unit
                        or parent.perimeter_version != o.perimeter_version
                    ):
                        raise ValueError(
                            "Imputation must be trailing, institution-local and comparable"
                        )

    def append(self, *observations):
        return VintageStore(self.observations + tuple(observations))

    def known_records(self, as_of):
        cutoff = instant(as_of)
        return tuple(o for o in self.observations if instant(o.available_as_of) <= cutoff)

    def as_of(self, as_of):
        selected = {}
        for o in self.known_records(as_of):
            old = selected.get(o.key)
            if old is None or instant(o.filing_date) > instant(old.filing_date):
                selected[o.key] = o
            elif (
                instant(o.filing_date) == instant(old.filing_date)
                and o.observation_id != old.observation_id
            ):
                # Same reported facts ingested twice are duplicates, not amendments.
                facts = (o.value, o.unit, o.definition_version, o.perimeter_version, o.provenance)
                old_facts = (
                    old.value,
                    old.unit,
                    old.definition_version,
                    old.perimeter_version,
                    old.provenance,
                )
                if facts != old_facts:
                    raise ValueError(
                        f"Conflicting source facts for {o.key}; analyst reconciliation required"
                    )
        return tuple(selected[key] for key in sorted(selected))

    def get_observation_as_of(self, institution_id, form, metric, period, as_of):
        key = institution_id, form, metric, day(period)
        return next((o for o in self.as_of(as_of) if o.key == key), None)

    def filing_revision_history(self, institution_id, form, metric, period, *, as_of):
        key = institution_id, form, metric, day(period)
        return tuple(o for o in self.known_records(as_of) if o.key == key)

    def compare_vintages(self, previous_as_of, current_as_of):
        if instant(current_as_of) < instant(previous_as_of):
            raise ValueError("Current as-of precedes previous review")
        before = {o.key: o for o in self.as_of(previous_as_of)}
        return tuple(
            FilingRevision(
                before[o.key],
                o,
                (
                    None
                    if before[o.key].value is None or o.value is None
                    else o.value - before[o.key].value
                ),
            )
            for o in self.as_of(current_as_of)
            if o.key in before and o.observation_id != before[o.key].observation_id
        )

    def export_jsonl(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(canonical_json(o) + "\n" for o in self.observations), encoding="utf-8"
        )
        return output

    @classmethod
    def from_file(cls, path):
        """Read normalized local CSV/JSONL; never fetch or infer source fields."""
        path = Path(path)
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = (
                csv.DictReader(f)
                if path.suffix.lower() == ".csv"
                else (json.loads(line) for line in f if line.strip())
            )
            return cls(tuple(RegulatoryObservation.from_dict(r) for r in rows))


def build_regulatory_dataset(
    cls,
    store: VintageStore,
    registry: SemanticRegistry,
    *,
    institution_id: str,
    form: str,
    as_of: str,
    metrics=None,
    peer_group=None,
    allow_synthetic=False,
):
    records = tuple(
        o
        for o in store.as_of(as_of)
        if o.institution_id == institution_id
        and o.form == form
        and (metrics is None or o.metric in metrics)
    )
    if not records:
        raise ValueError("No observations available for this institution/form/as-of")
    names = tuple(metrics) if metrics is not None else tuple(sorted({o.metric for o in records}))
    if not names or len(set(names)) != len(names):
        raise ValueError("Metrics must be unique and nonempty")
    definitions = {}
    for o in records:
        definition = registry.resolve(
            form,
            o.metric,
            o.reporting_period,
            as_of=as_of,
            version=o.definition_version,
            allow_synthetic=allow_synthetic,
        )
        if o.unit != definition.unit:
            raise ValueError(f"Unit mismatch for {o.metric}: {o.unit} vs {definition.unit}")
        if definition.frequency != "quarterly":
            raise ValueError("This dataset requires quarterly definitions")
        if (
            pd.Timestamp(o.reporting_period)
            != pd.Period(o.reporting_period, freq="Q").end_time.normalize()
        ):
            raise ValueError("Regulatory periods must be quarter ends")
        definitions[f"{o.metric}:{o.definition_version}"] = canonical(definition)
    periods = pd.period_range(
        min(o.reporting_period for o in records), max(o.reporting_period for o in records), freq="Q"
    )
    dates = tuple(p.end_time.date().isoformat() for p in periods)
    frame = pd.DataFrame(index=pd.Index(dates, name="reporting_period"), columns=names, dtype=float)
    for o in records:
        frame.loc[o.reporting_period, o.metric] = np.nan if o.value is None else o.value
    flags = tuple(
        f"missing:{period}:{metric}"
        for period in dates
        for metric in names
        if pd.isna(frame.loc[period, metric])
    )
    return cls(
        frame=frame,
        values=frame.to_numpy(dtype=float),
        variable_names=names,
        dates=dates,
        sampling_rate=4.0,
        metadata={
            "source": "vintage_store",
            "form": form,
            "semantic_registry": registry.version,
            "allow_synthetic": allow_synthetic,
            "transform": "level",
            "standardize": False,
        },
        institution_id=institution_id,
        institution_name=records[-1].institution_name,
        filing_type=form,
        reporting_period=dates[-1],
        filing_vintage=tuple(sorted({o.source_vintage for o in records})),
        retrieved_at=max(o.ingestion_date for o in records),
        available_as_of=instant(as_of).isoformat(),
        peer_group=peer_group,
        provenance={o.observation_id: o.provenance.value for o in records},
        source_citations=tuple(sorted({o.source for o in records})),
        metric_metadata=definitions,
        institution_metadata={"perimeter_versions": sorted({o.perimeter_version for o in records})},
        data_quality_flags=flags,
        imputation_metadata=tuple(
            canonical(o.lineage)
            for o in records
            if o.provenance == ObservationProvenance.IMPUTED_CAUSAL
        ),
        reporting_definition_version=tuple(sorted({o.definition_version for o in records})),
        observations=records,
        _store=store,
        _registry=registry,
    )
