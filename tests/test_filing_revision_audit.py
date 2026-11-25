from dataclasses import replace

import pytest

from test_regulatory_foundation import observation
from drr_framework.supervisory import (
    ReviewState,
    VintageStore,
    build_revision_audit,
    compare_review_states,
)


def review(record, as_of):
    return ReviewState(as_of, (("|".join(record.key), record.observation_id),))


def amended(old, **kwargs):
    return replace(
        old,
        **dict(
            dict(
                value=120,
                amendment_date="2025-10-01",
                ingestion_date="2025-10-03",
                available_as_of="2025-10-03",
                source_vintage="v2",
            ),
            **kwargs,
        ),
    )


def audit(old, new, store=None):
    before, after = review(old, "2025-09-01"), review(new, "2025-10-04")
    return build_revision_audit(
        store or VintageStore((old, new)), before, after, compare_review_states(before, after)
    )


def test_revision_compares_exact_review_ids_and_preserves_full_source_facts():
    old = observation()
    new = amended(old)
    later = amended(
        old,
        value=900,
        amendment_date="2025-10-04",
        source_vintage="v3",
        ingestion_date="2025-10-04",
        available_as_of="2025-10-04",
    )
    row = audit(old, new, VintageStore((old, new, later)))[0]
    assert row["raw_change"] == 20
    assert row["percent_change"] == 20
    assert row["current_observation_id"] == new.observation_id
    assert row["previous_observation_id"] == old.observation_id
    assert row["current"]["value"] == 120
    assert {"value", "source_vintage", "available_as_of"} <= set(row["changed_fields"])
    assert row["comparison_status"] == "comparable"
    assert audit(old, new, VintageStore((later, new, old))) == (row,)


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"unit": "USD millions"}, "incomparable"),
        ({"definition_version": "new"}, "incomparable"),
        ({"perimeter_version": "new"}, "incomparable"),
        ({"value": None}, "missing_value"),
    ],
)
def test_incomparable_or_missing_values_never_produce_a_numeric_revision(changes, status):
    old = observation()
    row = audit(old, amended(old, **changes))[0]
    assert row["comparison_status"] == status
    assert row["raw_change"] is row["percent_change"] is None
    assert row["limitation"]


def test_zero_prior_and_metadata_only_revisions():
    old = observation(value=0)
    row = audit(old, amended(old, value=10))[0]
    assert row["raw_change"] == 10 and row["percent_change"] is None
    assert "zero" in row["limitation"]
    row = audit(old, amended(old, value=0))[0]
    assert row["raw_change"] == 0 and "value" not in row["changed_fields"]


def test_large_finite_values_do_not_silently_lose_a_valid_percent_revision():
    old = observation(value=1e308)
    row = audit(old, amended(old, value=1.5e308))[0]
    assert row["percent_change"] == pytest.approx(50)


def test_unrepresentable_numeric_revision_is_explicit():
    old = observation(value=-1e308)
    row = audit(old, amended(old, value=1e308))[0]
    assert row["comparison_status"] == "numeric_overflow"
    assert row["raw_change"] is row["percent_change"] is None
    assert "floating-point" in row["limitation"]


def test_missing_historical_source_is_explicit_and_not_reconstructed():
    old = observation()
    new = amended(old)
    row = audit(old, new, VintageStore((new,)))[0]
    assert row["comparison_status"] == "source_unavailable"
    assert row["previous"] is None and row["current"]["value"] == 120
    assert row["raw_change"] is None


def test_initial_unchanged_and_out_of_scope_reviews_have_no_revisions():
    old = observation()
    store = VintageStore((old,))
    state = review(old, "2025-09-01")
    assert build_revision_audit(store, None, state, compare_review_states(None, state)) == ()
    assert build_revision_audit(store, state, state, compare_review_states(state, state)) == ()
    empty = ReviewState("2025-10-01")
    assert build_revision_audit(store, state, empty, compare_review_states(state, empty)) == ()


def test_forged_review_identity_or_future_cutoff_is_rejected():
    old = observation()
    new = amended(old)
    before, after = review(old, "2025-09-01"), review(new, "2025-10-04")
    before = replace(before, observations=(("|".join(old.key), new.observation_id),))
    after = replace(after, observations=(("|".join(old.key), old.observation_id),))
    with pytest.raises(ValueError, match="cutoff"):
        build_revision_audit(
            VintageStore((old, new)), before, after, compare_review_states(before, after)
        )


def test_review_observation_identity_cannot_be_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate observation"):
        ReviewState("2025-09-01", (("key", "one"), ("key", "two")))
    pairs = [["key", "id"]]
    state = ReviewState("2025-09-01", pairs)
    pairs[0][1] = "mutated"
    assert state.observations == (("key", "id"),)
