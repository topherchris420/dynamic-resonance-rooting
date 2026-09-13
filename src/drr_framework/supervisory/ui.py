"""Small local workstation: semantic HTML, no external assets, deterministic explanations."""

from __future__ import annotations

import html
import json
import math

from .common import canonical_json

STYLE = """
:root{color-scheme:light;--ink:#1d302c;--muted:#60716a;--paper:#f6f5ef;--line:#d8dfd7;--accent:#276658}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 system-ui,sans-serif}a{color:#185e50}a:focus-visible,button:focus-visible,summary:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #c57932;outline-offset:4px}
header{border-bottom:1px solid var(--line);padding:25px 4vw;display:flex;justify-content:space-between;align-items:center;background:#fff}header strong{letter-spacing:.1em;font-size:12px}header small{display:block;color:var(--muted)}.scope{padding:5px 13px;border:1px solid var(--line);border-radius:20px;font-size:12px}
.layout{display:grid;grid-template-columns:205px minmax(0,1fr);min-height:85vh}nav{padding:30px 18px;border-right:1px solid var(--line)}nav a{display:block;text-decoration:none;padding:10px 13px;margin-bottom:3px;color:var(--muted);border-radius:5px}nav a[aria-current=page]{background:var(--ink);color:white}main{padding:36px 4vw;max-width:1500px;width:100%}.eyebrow{text-transform:uppercase;font-size:11px;font-weight:700;letter-spacing:.15em;color:var(--muted)}h1{font:normal 39px/1.15 Georgia,serif;margin:9px 0 12px}h2{font:normal 26px Georgia,serif;margin:30px 0 16px}h3{font-size:17px;margin:0 0 10px}.lede{color:var(--muted);max-width:800px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}.stat,.card{background:white;border:1px solid var(--line);padding:21px;border-radius:8px}.stat strong{display:block;font:32px Georgia,serif}.stat small{color:var(--muted)}.card{margin:14px 0}.card-top{display:flex;justify-content:space-between;gap:20px}.tag{display:inline-block;background:#edf3ee;border:1px solid #d3e4d8;border-radius:4px;padding:2px 8px;font-size:11px;color:var(--accent)}.muted{color:var(--muted)}.meta{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;margin:13px 0}.explain{border-top:1px solid var(--line);margin-top:16px;padding-top:12px}summary{cursor:pointer;color:var(--accent);font-weight:600}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:11px 10px;vertical-align:top}th{color:var(--muted);font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f1;padding:15px;border-radius:6px;font-size:12px}code{overflow-wrap:anywhere}.empty{border:1px dashed #bdcbbf;padding:28px;border-radius:8px;margin:20px 0}.boundary{font-size:12px;color:var(--muted);margin:35px 0 0;border-top:1px solid var(--line);padding-top:18px}.panel[hidden]{display:none}.review-form{display:grid;gap:10px;max-width:600px;margin-top:16px}label{font-size:13px}input,select,textarea{display:block;width:100%;padding:9px;border:1px solid #acbdb2;border-radius:4px;background:white;color:var(--ink);font:inherit}button,.button{background:var(--ink);color:white;border:0;border-radius:5px;padding:10px 15px;font:inherit;cursor:pointer;text-decoration:none;display:inline-block}button:disabled{opacity:.6;cursor:default}.spark{width:140px;height:34px;color:var(--accent)}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:800px){.layout{display:block}nav{display:flex;overflow:auto;border-bottom:1px solid var(--line);padding:8px}nav a{white-space:nowrap;margin:0}main{padding:25px 18px}.stats{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}header{padding:18px}.card-top{display:block}h1{font-size:32px}}
"""

SCRIPT = """
const links=[...document.querySelectorAll('nav a')];
function show(){const id=location.hash.slice(1)||'today';const valid=links.some(a=>a.hash==='#'+id)?id:'today';document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==valid);links.forEach(a=>{if(a.hash==='#'+valid)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});}
addEventListener('hashchange',show);show();
document.querySelectorAll('.review-form').forEach(form=>form.addEventListener('submit',async event=>{event.preventDefault();const status=form.querySelector('[role=status]');const button=form.querySelector('button');button.disabled=true;try{const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':form.dataset.token},body:JSON.stringify(Object.fromEntries(new FormData(form)))});const data=await response.json();if(!response.ok)throw new Error(data.error||'Review could not be recorded');status.textContent='Disposition recorded. Refresh to update the review queue.';}catch(error){status.textContent=error.message;}finally{button.disabled=false;}}));
"""


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, digits=2):
    return "Unavailable" if value is None else f"{value:,.{digits}f}"


def table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join('<th scope="col">' + esc(h) + "</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + str(cell) + "</td>" for cell in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def sparkline(values, label):
    finite = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return '<span class="muted">Insufficient history</span>'
    lo, hi = min(v for _, v in finite), max(v for _, v in finite)
    paths = []
    prior = None
    for i, value in finite:
        x = 3 + 134 * i / max(len(values) - 1, 1)
        y = 31 - 27 * (value - lo) / (hi - lo) if hi > lo else 17
        paths.append(("M" if prior is None or i != prior + 1 else "L") + f"{x:.1f},{y:.1f}")
        prior = i
    return f'<svg class="spark" viewBox="0 0 140 34" role="img" aria-label="{esc(label)}"><path d="{" ".join(paths)}" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>'


def evidence_details(oid, payload, *, token="", editable=False):
    history = payload.get("historical_context", {})
    peer = payload.get("peer_context", {})
    robust = payload.get("robustness") or {}
    baseline = payload.get("baseline_evidence", [])
    drr = payload.get("drr_evidence", {})
    facts = payload["source_facts"]
    content = '<div class="explain"><details><summary>Why am I seeing this?</summary>'
    questions = [
        ("What changed?", payload["claim"]),
        (
            "Why flagged?",
            "; ".join(history.get("reason_flagged", []))
            or "Multivariate structural diagnostic crossed its configured threshold",
        ),
        (
            "Compared with what?",
            f"Historical percentile: {number(history.get('historical_percentile'))}. Peer percentile: {number(peer.get('percentile'))}.",
        ),
        (
            "What did simple baselines say?",
            ", ".join(b["method"] for b in baseline if b["flagged"]) or "No baseline flagged",
        ),
        (
            "What did DRR add?",
            f"{drr.get('status','unavailable')}; additional alert: {drr.get('incremental_alert',False)}. Incremental usefulness requires evaluation.",
        ),
        (
            "Did it survive alternatives?",
            f"{robust.get('classification','not evaluated')}; {number(robust.get('percentage_surviving'))}% of {robust.get('planned_count',0)} planned specifications.",
        ),
        (
            "Contradictory evidence",
            "; ".join(payload.get("contradictory_evidence", []))
            or "None recorded in tested specifications.",
        ),
        (
            "What was calculated or imputed?",
            f"{payload['calculation']['formula']}. {sum(f['provenance']=='imputed_causal' for f in facts)} imputed source observations.",
        ),
        (
            "Reporting changes",
            "; ".join(e["title"] for e in payload.get("policy_context", []))
            or "No relevant events in the supplied inventory.",
        ),
        ("What does this not establish?", payload["decision_boundary"]),
    ]
    content += table(("Question", "Evidence"), ((esc(k), esc(v)) for k, v in questions))
    content += (
        "<details><summary>Exact observations and filing vintages</summary>"
        + table(
            ("Institution", "Metric", "Period", "Value", "Provenance", "Vintage", "Source"),
            (
                (
                    esc(f["institution_id"]),
                    esc(f["metric"]),
                    esc(f["reporting_period"]),
                    number(f["value"]),
                    esc(f["provenance"]),
                    esc(f["source_vintage"]),
                    (
                        f'<a href="{esc(f["source"])}" rel="noreferrer">Public source</a>'
                        if f["source"].startswith("https://")
                        else esc(f["source"])
                    ),
                )
                for f in facts
            ),
        )
        + "</details>"
    )
    content += f'<details><summary>Calculation, inputs and full evidence packet</summary><pre>{esc(json.dumps(payload,indent=2,ensure_ascii=False))}</pre></details><p class="muted">Evidence ID: <code>{esc(oid)}</code></p>'
    if editable:
        content += (
            f'<form class="review-form" data-token="{esc(token)}"><input type="hidden" name="evidence_id" value="{esc(oid)}"><label>Disposition<select name="disposition">'
            + "".join(
                f'<option value="{s}">{s.title()}</option>'
                for s in ("investigate", "useful", "explained", "noisy", "dismissed", "unresolved")
            )
            + '</select></label><label>Reviewer<input name="reviewer" required maxlength="120" autocomplete="name"></label><label>Rationale<textarea name="rationale" required maxlength="4000" rows="3"></textarea></label><label>Review minutes<input name="review_minutes" type="number" min="0" max="1440" step="0.5" value="0"></label><button type="submit">Record disposition</button><span role="status" aria-live="polite"></span></form>'
        )
    content += "</details></div>"
    return content


def render_workbench(result, *, token="", editable=False):
    nav = (
        ("today", "Today"),
        ("institutions", "Institutions"),
        ("peers", "Peers"),
        ("evidence", "Evidence"),
        ("policy", "Policy"),
        ("quality", "Data quality"),
        ("queue", "Review queue"),
        ("validation", "Validation"),
        ("perspectives", "Perspectives"),
    )
    sections = {}
    attention = result["attention"]
    delta = result["delta"]
    selected = attention["review_first"]
    today = '<p class="eyebrow">Continuous monitoring</p><h1>Since your last review</h1><p class="lede">What changed, what deserves attention, and the evidence behind it.</p><div class="stats">'
    for count, label in (
        (len(selected), "items to review"),
        (len(delta["changed_observations"]), "revised observations"),
        (len(delta["new_exceptions"]), "new data exceptions"),
        (len(attention["deprioritized"]), "deprioritized signals"),
    ):
        today += f'<div class="stat"><strong>{count}</strong><small>{label}</small></div>'
    today += "</div>"
    for item in selected:
        s = item["signal"]
        p = result["evidence"][s["evidence_id"]]
        today += (
            f'<article class="card"><div class="card-top"><div><span class="eyebrow">{esc(s["institution"])} · {esc(s["period"])}</span><h3>{esc(s["claim"])}</h3></div><span class="tag">{esc(item["novelty"])}</span></div><div class="meta"><span>Materiality {s["materiality"]:.0f}/100</span><span>Data confidence {s["confidence"]:.0%}</span><span>Robustness {number(s["robustness"]*100 if s["robustness"] is not None else None)}%</span></div>'
            + evidence_details(s["evidence_id"], p, token=token, editable=editable)
            + "</article>"
        )
    if not selected:
        message = (
            "No material signal cleared the queue. Review the data-quality blockers before treating this as a quiet period."
            if any(e["severity"] == "important" for e in result["quality"])
            else "Nothing material changed among eligible observations and configured review thresholds."
        )
        today += f'<div class="empty">{esc(message)}</div>'
    if attention["deferred"]:
        today += f'<p>{len(attention["deferred"])} additional items are deferred by the attention budget. <a href="#queue">Open review queue</a>.</p>'
    today += "<h2>Deprioritized</h2>" + table(
        ("Observation", "Reason"),
        (
            (
                esc(s["claim"]),
                "Weakened, disappeared, or dispositioned; inspect filing revisions and evidence.",
            )
            for s in attention["deprioritized"]
        ),
    )
    sections["today"] = today
    institutions = (
        '<p class="eyebrow">Institution monitoring</p><h1>Financial context, in view</h1>'
    )
    for a in result["analyses"]:
        institutions += (
            f'<article class="card"><h3>{esc(a["name"])}</h3><p class="muted">{esc(a["form"])} · {esc(a["period"])} · {len(a["filing_vintages"])} source vintages</p>'
            + table(
                ("Metric", "Reported value", "Quarter change", "History"),
                (
                    (
                        esc(c["label"]),
                        number(c["value"]) + " " + esc(c["unit"]),
                        number(c["normalized_change"]) + "%",
                        sparkline(a["history"][c["metric"]], c["label"] + " quarterly history"),
                    )
                    for c in a["changes"]
                ),
            )
            + "</article>"
        )
    sections["institutions"] = institutions
    peer_rows = [p for a in result["analyses"] for p in a["peers"].values()]
    sections["peers"] = (
        '<p class="eyebrow">Horizontal comparison</p><h1>Who moved together?</h1><p class="lede">Cohorts are explicit. The target is excluded from its comparison distribution. Shared movement does not identify a causal source.</p>'
        + table(
            ("Institution", "Metric", "Peers", "Percentile", "Context", "Paired peers moving"),
            (
                (
                    esc(p["institution"]),
                    esc(p["metric"]),
                    str(len(p["included"])),
                    number(p["percentile"]),
                    esc(p["context"].replace("_", " ")),
                    f"{p['moving_together']}/{p['paired_peer_count']}",
                )
                for p in peer_rows
            ),
        )
    )
    evidence = '<p class="eyebrow">Evidence ledger</p><h1>Every claim, traceable</h1>'
    for oid, p in result["evidence"].items():
        evidence += (
            f'<article class="card"><h3>{esc(p["claim"])}</h3>'
            + evidence_details(oid, p, token=token, editable=editable)
            + "</article>"
        )
    sections["evidence"] = evidence
    sections["policy"] = '<p class="eyebrow">Public context</p><h1>Policy & reporting</h1>' + table(
        ("Event", "Published", "Effective", "Status", "Applicability"),
        (
            (
                f'<a href="{esc(e["source"])}" rel="noreferrer">{esc(e["title"])}</a>',
                esc(e["publication_date"]),
                esc(e["effective_date"] or "Not specified"),
                esc(e["status"]),
                esc(e["applicability"].replace("_", " ")),
            )
            for e in result["policy"]
        ),
    )
    sections["quality"] = (
        '<p class="eyebrow">Reconciliation</p><h1>Data needing review</h1>'
        + table(
            ("Institution", "Metric", "Period", "Exception", "Explanation"),
            (
                (
                    esc(e["institution"]),
                    esc(e["metric"]),
                    esc(e["period"]),
                    esc(e["issue_type"].replace("_", " ")),
                    esc(e["likely_explanation"] or "Analyst confirmation required"),
                )
                for e in result["quality"]
                if e["required_analyst_review"]
            ),
        )
    )
    queue = [
        s for s in result["state"]["signals"] if s["disposition"] in ("unresolved", "investigate")
    ]
    sections["queue"] = (
        '<p class="eyebrow">Analyst workflow</p><h1>Review queue</h1><p class="lede">Dispositions record human interpretation separately from the original evidence.</p>'
    )
    if not editable:
        sections[
            "queue"
        ] += "<p>Open the local server with <code>drr-monitor --serve --demo</code> to record dispositions.</p>"
    for s in queue:
        sections["queue"] += (
            f'<article class="card"><h3>{esc(s["claim"])}</h3>'
            + evidence_details(
                s["evidence_id"],
                result["evidence"][s["evidence_id"]],
                token=token,
                editable=editable,
            )
            + "</article>"
        )
    model_risk = {
        "profile": result.get("model_risk_profile", {}),
        "reference_basis": result.get("model_risk_reference_basis", []),
    }
    sections["validation"] = (
        '<p class="eyebrow">Analytical validation</p><h1>Does complexity add value?</h1><p class="lede">Baseline comparisons, sensitivity failures, and human usefulness measures stay visible. Development tests are separate from independent validation.</p>'
        + table(
            ("Institution", "DRR status", "Baseline alerts", "Structural alert"),
            (
                (
                    esc(a["institution"]),
                    esc(a["drr"]["status"]),
                    str(sum(b["flagged"] for b in a["baselines"])),
                    str(a["drr"].get("structural_alert", False)),
                )
                for a in result["analyses"]
            ),
        )
        + f'<details class="card"><summary>Model-risk profile and SR 26-2 reference basis</summary><pre>{esc(json.dumps(model_risk,indent=2))}</pre></details><details class="card"><summary>Analyst feedback metrics</summary><pre>{esc(json.dumps(result["feedback"],indent=2))}</pre></details><details class="card"><summary>Analysis passport and reproducibility</summary><pre>{esc(json.dumps(result["passport"],indent=2))}</pre></details>'
    )
    perspectives = result.get("perspectives", {})
    sections["perspectives"] = (
        '<p class="eyebrow">Representation boundaries</p><h1>What each perspective sees</h1>'
        '<p class="lede">A model can be excellent without being exhaustive. Examine evidence, scope, and documented disagreement together.</p>'
        + table(
            (
                "Perspective",
                "Claim",
                "Population / scale",
                "Dimension / horizon",
                "Support assessment",
                "Outside scope",
            ),
            (
                (
                    esc(p["kind"].replace("_", " ")),
                    esc(p["claim"]),
                    esc(p["population"] + " / " + p["scale"]),
                    esc(p["dimension"] + " / " + p["horizon"]),
                    esc(p["support_assessment"] + " — " + p["support_assessor"]),
                    esc("; ".join(p["outside_scope"])),
                )
                for p in perspectives.get("perspectives", [])
            ),
        )
        + '<details class="card"><summary>Sources, methods, interpretation and documented disagreement</summary><pre>'
        + esc(json.dumps(perspectives, indent=2))
        + "</pre></details>"
    )
    body = "".join(
        f'<section class="panel" id="{key}" aria-label="{label}">{sections[key]}</section>'
        for key, label in nav
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LFBO Monitoring · Vers3Dynamics</title><style>'
        + STYLE
        + '</style></head><body><a href="#today" class="sr-only">Skip to monitoring</a><header><div><strong>VERS3DYNAMICS</strong><small>LFBO Supervisory Research Workbench</small></div><span class="scope">'
        + esc(result["scope"])
        + " · "
        + esc(result["as_of"][:10])
        + '</span></header><div class="layout"><nav aria-label="Workbench">'
        + "".join(f'<a href="#{key}">{label}</a>' for key, label in nav)
        + "</nav><main>"
        + body
        + '<p class="boundary">'
        + esc(result["decision_boundary"])
        + "</p></main></div><script>"
        + SCRIPT
        + "</script></body></html>"
    )
