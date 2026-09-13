from dataclasses import replace

import numpy as np
import pytest

from drr_framework import RegulatoryAnalysisDataset
from drr_framework.supervisory.semantics import (
    MetricDefinition,
    SemanticRegistry,
    VerificationStatus,
)
from drr_framework.supervisory.vintage import (
    RegulatoryObservation,
    VintageStore,
    ObservationProvenance,
    CalculationLineage,
)
from drr_framework.supervisory.reconciliation import reconcile_dataset, reconcile_store
from drr_framework.supervisory.model_risk import (
    ModelRiskProfile,
    ValidationEvidence,
    ValidationStatus,
    MODEL_RISK_REFERENCE_BASIS,
)
from drr_framework.supervisory.common import file_sha256
from drr_framework.supervisory.ingestion import FilingContext, ingest_wide_csv
from drr_framework.supervisory.semantics import bundled_registry


def definition(code="SYN_ASSETS", **kwargs):
    values = dict(
        code=code,
        label="Assets",
        form="FR Y-9C",
        schedule="synthetic",
        line="synthetic",
        definition="Synthetic assets; no real MDRM mapping",
        unit="USD thousands",
        effective_from="2000-01-01",
        version="v1",
        source="synthetic fixture",
        verification_date="2000-01-01",
        status=VerificationStatus.SYNTHETIC,
    )
    values.update(kwargs)
    return MetricDefinition(**values)


def observation(period="2025-03-31", value=100.0, **kwargs):
    values = dict(
        institution_id="A",
        institution_name="Synthetic A",
        form="FR Y-9C",
        metric="SYN_ASSETS",
        reporting_period=period,
        value=value,
        unit="USD thousands",
        original_filing_date="2025-08-01",
        ingestion_date="2025-08-02",
        source_vintage="v1",
        available_as_of="2025-08-02",
        definition_version="v1",
        source="synthetic fixture",
        provenance=ObservationProvenance.SYNTHETIC_TEST,
    )
    values.update(kwargs)
    return RegulatoryObservation(**values)


def dataset(records, as_of="2025-09-01", registry=None):
    return RegulatoryAnalysisDataset.from_vintage_store(
        VintageStore(records),
        registry or SemanticRegistry((definition(),)),
        institution_id="A",
        form="FR Y-9C",
        as_of=as_of,
        allow_synthetic=True,
    )


def test_as_of_restatement_and_late_ingestion_do_not_leak():
    old = observation()
    revision = replace(
        old,
        value=120,
        amendment_date="2025-10-01",
        ingestion_date="2025-10-03",
        available_as_of="2025-10-03",
        source_vintage="v2",
        supersedes=old.observation_id,
    )
    store = VintageStore((old, revision))
    assert store.as_of("2025-08-01") == ()
    assert store.as_of("2025-10-02") == (old,)
    assert store.as_of("2025-10-03") == (revision,)
    assert store.compare_vintages("2025-09-01", "2025-10-03")[0].raw_change == 20
    assert dataset((old, revision)).as_of("2025-10-03").values[0, 0] == 120
    late_original = replace(old, ingestion_date="2025-11-01", available_as_of="2025-11-01")
    assert VintageStore((old, revision, late_original)).as_of("2025-12-01") == (revision,)


def test_missing_quarters_and_nulls_are_preserved_and_drr_blocks():
    d = dataset(
        (
            observation(),
            observation(
                "2025-09-30",
                None,
                original_filing_date="2025-10-01",
                ingestion_date="2025-10-01",
                available_as_of="2025-10-01",
            ),
        ),
        "2025-11-01",
    )
    assert d.values.shape == (3, 1)
    assert np.isnan(d.values[1:, 0]).all()
    with pytest.raises(ValueError, match="missing"):
        d.to_drr_input()
    assert {e.issue_type for e in reconcile_dataset(d)} >= {"missing_quarter", "missing_value"}


def test_form_namespace_and_unverified_and_future_semantics_are_blocked():
    registry = SemanticRegistry((definition(),))
    for form, code in (("FFIEC 002", "SYN_ASSETS"), ("FR Y-9C", "RCFD2170")):
        with pytest.raises(ValueError):
            registry.resolve(form, code, "2025-03-31", as_of="2025-09-01", allow_synthetic=True)
    for d in (definition(status="unverified"), definition(verification_date="2026-01-01")):
        with pytest.raises(ValueError):
            dataset((observation(),), registry=SemanticRegistry((d,)))
    with pytest.raises(ValueError):
        RegulatoryAnalysisDataset.from_vintage_store(
            VintageStore((observation(),)),
            registry,
            institution_id="A",
            form="FR Y-9C",
            as_of="2025-09-01",
        )


def test_stable_observation_ids_and_source_conflicts():
    o = observation()
    assert (
        o.observation_id
        == RegulatoryObservation.from_dict(__import__("dataclasses").asdict(o)).observation_id
    )
    assert reconcile_store(VintageStore((o, o)), "2025-09-01")[0].issue_type == "duplicate_record"
    store = VintageStore((o, replace(o, value=3)))
    assert reconcile_store(store, "2025-09-01")[0].issue_type == "source_conflict"
    with pytest.raises(ValueError, match="Conflicting"):
        store.as_of("2025-09-01")


def test_derived_lineage_rejects_future_inputs_and_cross_institution_imputation():
    raw = observation()
    lineage = CalculationLineage(
        "last_observation",
        (raw.observation_id,),
        (raw.metric,),
        (raw.reporting_period,),
        "test",
        (),
        raw.unit,
        "2025-08-02",
    )
    derived = replace(
        raw,
        reporting_period="2025-06-30",
        provenance=ObservationProvenance.IMPUTED_CAUSAL,
        lineage=lineage,
    )
    assert len(VintageStore((raw, derived)).observations) == 2
    with pytest.raises(ValueError, match="institution-local"):
        VintageStore((raw, replace(derived, institution_id="B")))
    with pytest.raises(ValueError, match="future"):
        future = replace(raw, ingestion_date="2025-09-01", available_as_of="2025-09-01")
        VintageStore(
            (future, replace(derived, lineage=replace(lineage, input_ids=(future.observation_id,))))
        )
    with pytest.raises(ValueError, match="lineage"):
        replace(raw, provenance=ObservationProvenance.DERIVED_COMPUTED)


def test_matrix_orientation_and_display_mutation_cannot_change_drr_input():
    d = dataset((observation(), observation("2025-06-30", 110)))
    assert d.to_drr_input().shape == (2, 1)
    d.frame.iloc[0, 0] = 999
    assert d.to_drr_input()[0, 0] == 100


def test_model_risk_has_current_basis_and_requires_independent_evidence():
    assert MODEL_RISK_REFERENCE_BASIS[0]["status"] == "current"
    assert "SR 26-2" in MODEL_RISK_REFERENCE_BASIS[0]["label"]
    with pytest.raises(ValueError, match="Independent"):
        ModelRiskProfile(
            "DRR",
            "research",
            ("ratings",),
            validation_status=ValidationStatus.INDEPENDENTLY_REVIEWED,
        )

    with pytest.raises(ValueError):
        ValidationEvidence("implementation", ("not-a-hash",), "Reviewer", "not-a-date")
    evidence = ValidationEvidence(
        "implementation",
        ("a" * 64,),
        "Independent reviewer",
        "2026-09-12",
        independent=True,
    )
    profile = ModelRiskProfile(
        "DRR",
        "research",
        ("ratings",),
        validation_status=ValidationStatus.INDEPENDENTLY_REVIEWED,
        evidence=(evidence,),
    )
    assert profile.evidence[0].content_addressed


def test_public_ingestion_verifies_file_hash_and_exact_semantics(tmp_path):
    filing = tmp_path / "filing.csv"
    filing.write_text(
        "RSSD_ID,LEGAL_NAME,BHCK0081,BHCK2170,BHCK3210\n"
        "1234567,Example Holding Company,25,1000,90\n",
        encoding="utf-8",
    )
    context = FilingContext(
        form="FR Y-9C",
        reporting_period="2026-03-31",
        original_filing_date="2026-05-01",
        ingestion_date="2026-09-12",
        available_as_of="2026-09-12",
        source_vintage="2026Q1-public-download",
        source="https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload",
        source_hash=file_sha256(filing),
        definition_version="FRY9C-2026-03-verified-2026-09-12",
    )
    store = ingest_wide_csv(filing, bundled_registry(), context)
    assert len(store.observations) == 3
    assert {o.source_hash for o in store.observations} == {file_sha256(filing)}
    assert {o.metric for o in store.observations} == {
        "BHCK0081",
        "BHCK2170",
        "BHCK3210",
    }
    with pytest.raises(ValueError, match="does not match"):
        ingest_wide_csv(filing, bundled_registry(), replace(context, source_hash="0" * 64))
    with pytest.raises(ValueError, match="Unverified|Missing"):
        ingest_wide_csv(
            filing,
            bundled_registry(),
            context,
            metric_columns=("RCFD2170",),
        )
