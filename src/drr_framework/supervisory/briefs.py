"""Deterministic analyst briefs; all numerical claims come from structured evidence."""

from __future__ import annotations

from .common import DECISION_BOUNDARY


def generate_morning_brief(result):
    delta, attention = result["delta"], result["attention"]
    lines = [
        "# Morning Monitoring Brief",
        "",
        f"As of: {result['as_of']} · {result['scope']}",
        "",
        "## Since Last Review",
        "",
    ]
    lines += [
        f"- {len(delta['new_observations'])} newly available observations; {len(delta['changed_observations'])} revised observations.",
        f"- {len(delta['new_signals'])} new signals; {len(delta['strengthened'])} strengthened; {len(delta['weakened'])+len(delta['disappeared'])} weakened or disappeared.",
        f"- {len(delta['new_exceptions'])} new data-quality exceptions; {len(delta['resolved_exceptions'])} resolved exceptions.",
        f"- {len(delta['new_policy_events'])} new potentially relevant policy/reporting events.",
        f"- {len(result.get('perspectives', {}).get('disagreements', []))} documented perspective divergences remain visible.",
        "",
        "## Review First",
        "",
    ]
    if attention["review_first"]:
        for item in attention["review_first"]:
            s = item["signal"]
            lines += [
                f"- **{s['claim']}** — {item['novelty']}. Materiality {s['materiality']:.0f}/100; data confidence {s['confidence']:.0%}. [Why am I seeing this?](evidence/{s['evidence_id']}.md)"
            ]
            overlay = ((result.get("judgment") or {}).get("overlays") or {}).get(s["evidence_id"])
            if overlay:
                lines.append(
                    "  - Typed judgment policy: "
                    + str(overlay.get("policy_outcome", "JUDGMENT_UNAVAILABLE")).replace("_", " ")
                    + ". This is not the analytical score and it is not an analyst disposition."
                )
    elif any(e["severity"] == "important" for e in result["quality"]):
        lines.append(
            "No material signal cleared the review queue. Data-quality blockers require attention before interpreting this as a quiet period."
        )
    elif attention["deferred"]:
        lines.append("Material items were deferred by the configured attention budget.")
    else:
        lines.append(
            "Nothing material changed among the eligible observations and configured review thresholds."
        )
    if attention["deferred"]:
        lines += [
            "",
            f"{len(attention['deferred'])} additional items are deferred by the attention budget and remain visible in the queue.",
        ]
    lines += ["", "## Filing Revisions", ""]
    for revision in result.get("observation_revisions", []):
        old, new = revision["previous"], revision["current"]
        label = revision["observation_key"]
        if old and new:
            label = f"{new['institution_id']} · {new['form']} · {new['metric']} · {new['reporting_period']}"
            lines.append(
                f"- {label}: {old['value']} {old['unit']} → {new['value']} {new['unit']} "
                f"({old['source_vintage']} → {new['source_vintage']}). "
                f"Comparison: {revision['comparison_status']}. {revision['limitation']}"
            )
        else:
            lines.append(f"- {label}: {revision['limitation']}")
        for transition in revision["concurrent_signal_changes"]:
            lines.append(
                f"  - Concurrent signal change: {transition['transition'].replace('_', ' ')} — {transition['claim']}."
            )
    if not result.get("observation_revisions"):
        lines.append("No same-period filing revisions among the compared observations.")
    lines += ["", "## Deprioritized", ""]
    for s in attention["deprioritized"]:
        revised = any(
            k.startswith(s["institution"] + "|") and ("|" + s["metric"] + "|") in k
            for k in delta["changed_observations"]
        )
        reason = "Evidence weakened, disappeared, or was dispositioned"
        if revised:
            reason += "; the underlying filing was revised"
        lines.append(f"- {s['claim']}: {reason}.")
    if not attention["deprioritized"]:
        lines.append("None.")
    lines += ["", "## Data Quality", ""]
    new = set(delta["new_exceptions"])
    issues = [
        e for e in result["quality"] if e["required_analyst_review"] and e["exception_id"] in new
    ]
    lines += [
        f"- {e['institution']} · {e['metric']} · {e['period']}: {e['issue_type'].replace('_',' ')}. {e['likely_explanation'] or ''}"
        for e in issues
    ] or ["No new exceptions requiring review."]
    lines += ["", "## Policy / Reporting Changes", ""]
    events = [e for e in result["policy"] if e["identifier"]]
    from .common import stable_id

    events = [e for e in events if stable_id(e) in set(delta["new_policy_events"])]
    lines += [
        f"- [{e['title']}]({e['source']}) — {e['status']}; {e['applicability'].replace('_',' ')}."
        for e in events
    ] or ["No new relevant events in the supplied public event inventory."]
    lines += ["", DECISION_BOUNDARY, ""]
    return "\n".join(lines)


def generate_lfbo_monitoring_brief(result, institution):
    analyses = [a for a in result["analyses"] if a["institution"] == institution]
    if not analyses:
        raise ValueError("No eligible institution analysis")
    entries = {oid: p for oid, p in result["evidence"].items() if p["institution"] == institution}
    selected = [
        item
        for item in result["attention"]["review_first"]
        if item["signal"]["institution"] == institution
    ][:5]
    lines = [
        "# LFBO Monitoring Brief",
        "",
        f"Institution: {analyses[0]['name']} ({institution})",
        f"Reporting period: {', '.join(sorted({a['period'] for a in analyses}))}",
        f"As of: {result['as_of']}",
        f"Data vintage: {', '.join(sorted({v for a in analyses for v in a['filing_vintages']}))}",
        f"Peer group: {(result['passport'].get('peer_definition') or {}).get('name','Not configured')}",
        "",
        "## Executive View",
        "",
    ]
    lines += ["- " + i["signal"]["claim"] for i in selected] or [
        "No new material observations selected for this institution."
    ]
    lines += ["", "## Changes Since Previous Review", ""]
    changed = [
        k for k in result["delta"]["changed_observations"] if k.startswith(institution + "|")
    ]
    lines.append(
        f"{len(changed)} revised observations since the prior review. Only new or changed signals enter the morning queue."
    )
    perspectives = result.get("perspectives", {})
    relevant_perspectives = [
        p
        for p in perspectives.get("perspectives", [])
        if not p.get("institution_ids") or institution in p["institution_ids"]
    ]
    relevant_ids = {p["perspective_id"] for p in relevant_perspectives}
    all_disagreements = perspectives.get("disagreements", [])
    disagreements = [
        d for d in all_disagreements if set(d.get("perspective_ids", ())) <= relevant_ids
    ]
    lines += ["", "## Representation Scope", ""]
    lines.append(
        "Perspectives retain their population, scale, horizon, and evidence boundaries. "
        "A difference between perspectives is informative and is not resolved by authority or confidence."
    )
    lines += [
        f"- {p['kind']}: {p['claim']} ({p['population']}; {p['scale']}; {p['dimension']}; support {p['support_assessment']} by {p['support_assessor']})."
        for p in relevant_perspectives
    ]
    if disagreements:
        lines += [f"- Documented divergence: {d['interpretation']}" for d in disagreements]
    elif all_disagreements:
        lines.append(
            "No institution-specific documented divergence; broader system-level records remain visible in the Perspectives panel."
        )
    else:
        lines.append("No documented perspective divergence in the supplied inventory.")
    for title, domains in (
        ("Capital", ("capital",)),
        ("Liquidity / Funding", ("funding", "liquidity")),
        ("Balance Sheet / Exposure", ("balance_sheet",)),
    ):
        lines += ["", "## " + title, ""]
        rows = [(oid, p) for oid, p in entries.items() if p.get("domain") in domains]
        lines += [f"- {p['claim']} ([evidence](evidence/{oid}.md))." for oid, p in rows] or [
            "No selected material evidence in this domain; absence of an alert is not an assessment of safety."
        ]
    lines += [
        "",
        "## Governance / Controls Context",
        "",
        "No governance assessment is inferred from balance-sheet observations. Only sourced public context and analyst-entered interpretation belong here.",
    ]
    for title, field in (
        ("Peer / Horizontal Context", "peer_context"),
        ("Structural / DRR Evidence", "drr_evidence"),
        ("Robustness", "robustness"),
        ("Contradictory Evidence", "contradictory_evidence"),
        ("Policy / Reporting Context", "policy_context"),
        ("Data Quality", "data_limitations"),
    ):
        lines += ["", "## " + title, ""]
        if not entries:
            lines.append("No selected evidence to summarize.")
        for oid, p in entries.items():
            data = p[field]
            if field == "peer_context":
                text = f"{data.get('context','unavailable')}; percentile {data.get('percentile','unavailable')}; {len(data.get('included',[]))} comparable peers"
            elif field == "drr_evidence":
                text = f"DRR {data.get('status','unavailable')}; incremental alert {data.get('incremental_alert',False)}. This does not establish incremental usefulness."
            elif field == "robustness":
                text = f"{data.get('classification','not evaluated')}; {data.get('percentage_surviving')}% of {data.get('planned_count',0)} planned specifications survived"
            else:
                text = (
                    "; ".join(str(v) for v in data)
                    if data
                    else "None in the supplied evidence inventory."
                )
            lines.append(f"- {p['metric']}: {text} ([details](evidence/{oid}.md)).")
    lines += ["", "## Questions for Analyst Review", ""]
    lines += [
        f"- For {p['metric']} at {p['period']}, does the filing narrative explain the observed movement, and are the selected peers still comparable?"
        for p in entries.values()
    ] or ["- Does the current data coverage meet the intended monitoring scope?"]
    lines += ["", "## Evidence", ""] + [f"- [{oid}](evidence/{oid}.md)" for oid in entries]
    lines += ["", "## Analytical Boundary", "", DECISION_BOUNDARY, ""]
    return "\n".join(lines)
