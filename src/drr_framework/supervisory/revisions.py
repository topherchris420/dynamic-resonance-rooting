"""Explain changes to the exact observations held by two review states."""

from __future__ import annotations

import math

from .common import canonical, instant


def build_revision_audit(store, previous, current, delta):
    """Return traceable same-period revisions, never quarter-to-quarter changes.

    The previous state's IDs are authoritative: reconstructing its cutoff from a
    different input store could silently substitute evidence the analyst never saw.
    Missing historical records and comparability breaks remain explicit.
    """
    if previous is None:
        return ()
    if instant(current.as_of) < instant(previous.as_of):
        raise ValueError("Current review precedes previous state")
    before, after = dict(previous.observations), dict(current.observations)
    records = {o.observation_id: o for o in store.known_records(current.as_of)}
    rows = []
    for key in sorted(before.keys() & after.keys()):
        if before[key] == after[key]:
            continue
        old, new = records.get(before[key]), records.get(after[key])
        for record, cutoff in ((old, previous.as_of), (new, current.as_of)):
            if record and (
                "|".join(record.key) != key or instant(record.available_as_of) > instant(cutoff)
            ):
                raise ValueError("Review observation does not match its identity or cutoff")
        previous_fact = dict(observation_id=old.observation_id, **canonical(old)) if old else None
        current_fact = dict(observation_id=new.observation_id, **canonical(new)) if new else None
        changed_fields = (
            tuple(k for k in canonical(old) if getattr(old, k) != getattr(new, k))
            if old and new
            else ()
        )
        raw_change = percent_change = None
        if not old or not new:
            status = "source_unavailable"
            limitation = "An exact observation from the review state is absent from this store."
        elif any(
            k in changed_fields
            for k in ("unit", "definition_version", "perimeter_version", "provenance")
        ):
            status = "incomparable"
            limitation = "Units, definitions, reporting perimeter or provenance changed; no numerical revision is inferred."
        elif old.value is None or new.value is None:
            status = "missing_value"
            limitation = "A reported value is null; no numerical revision is inferred."
        else:
            status = "comparable"
            raw_change = new.value - old.value
            percent_change = raw_change / abs(old.value) * 100 if old.value != 0 else None
            limitation = (
                "Prior value is zero; percent revision is undefined." if old.value == 0 else ""
            )
            if not math.isfinite(raw_change):
                status = "numeric_overflow"
                raw_change = percent_change = None
                limitation = (
                    "The revision exceeds floating-point range; no numerical revision is reported."
                )
            elif percent_change is not None and not math.isfinite(percent_change):
                percent_change = None
                limitation = "The percent revision exceeds floating-point range."
        transitions = []
        record = new or old
        if record:
            for group in (
                "new_signals",
                "strengthened",
                "weakened",
                "changed_evidence",
                "disappeared",
            ):
                for signal in getattr(delta, group):
                    if (
                        signal.institution == record.institution_id
                        and signal.metric == record.metric
                        and signal.key.startswith(f"{record.institution_id}:{record.form}:")
                    ):
                        transitions.append(
                            {"transition": group, "key": signal.key, "claim": signal.claim}
                        )
        rows.append(
            {
                "observation_key": key,
                "previous_observation_id": before[key],
                "current_observation_id": after[key],
                "previous": previous_fact,
                "current": current_fact,
                "changed_fields": changed_fields,
                "comparison_status": status,
                "raw_change": raw_change,
                "percent_change": percent_change,
                "limitation": limitation,
                "concurrent_signal_changes": transitions,
            }
        )
    return tuple(rows)
