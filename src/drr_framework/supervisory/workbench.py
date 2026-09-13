"""End-to-end local monitoring orchestration with deterministic, inspectable outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Tuple

from .analyzer import DRRConfig, LFBORegimeAnalyzer
from .baselines import BaselineConfig, run_baselines
from .change_detection import detect_material_changes, attach_peer_context
from .common import canonical, canonical_json, instant, stable_id, DECISION_BOUNDARY
from .evidence_ledger import EvidenceEntry
from .feedback import evaluate_signal_usefulness
from .falsification import falsify_material_change, falsify_drr_signal
from .monitoring import ReviewState, MonitoringSignal, AttentionBudget, compare_review_states
from .passport import AnalysisPassport, software_identity
from .peer_analysis import analyze_peers
from .policy_context import PolicyContext
from .reconciliation import DataQualityException, reconcile_dataset, reconcile_store
from .entity_graph import EntityGraph
from .snc import analyze_public_snc
from .perspectives import PerspectiveInventory
from .model_risk import (
    MODEL_RISK_REFERENCE_BASIS,
    ModelRiskProfile,
    ModelUseClassification,
    ValidationStatus,
)

DEFAULT_MODEL_RISK_PROFILE = ModelRiskProfile(
    model_name="LFBO monitoring workbench with optional DRR diagnostic",
    intended_use="Analyst-directed public-data monitoring, evidence organization, and research diagnostics",
    foreseeable_misuse=(
        "supervisory ratings or findings",
        "MRA/MRIA or enforcement recommendations",
        "causal or governance conclusions from observational signals",
    ),
    use_classification=ModelUseClassification.RESEARCH_DIAGNOSTIC,
    validation_status=ValidationStatus.DEVELOPMENT_TESTED,
    limitations=(
        "DRR is optional and has not been independently validated for supervisory use",
        "Perspective records retain attributed scope and disagreement; they are not a consensus score",
    ),
    use_boundaries=(
        "human interpretation and independent challenge required",
        "institutional expertise and lived/material evidence are preserved without precedence rules",
    ),
    monitoring_plan=(
        "re-run point-in-time backtests and sensitivity checks after material changes",
        "review data-quality exceptions, drift, false positives, and analyst usefulness",
    ),
)


@dataclass(frozen=True)
class WorkbenchConfig:
    allow_synthetic: bool = False
    enable_drr: bool = True
    top_n: int = 5
    baseline: BaselineConfig = BaselineConfig()
    drr: DRRConfig = DRRConfig()
    institution_class: str = "large_foreign_banking_organization"
    model_risk_profile: ModelRiskProfile = DEFAULT_MODEL_RISK_PROFILE


class MonitoringWorkbench:
    def __init__(
        self,
        store,
        registry,
        ledger,
        *,
        cohort=None,
        config=None,
        policy=None,
        entities=None,
        snc=(),
        perspectives=None,
    ):
        self.store = store
        self.registry = registry
        self.ledger = ledger
        self.cohort = cohort
        self.config = config or WorkbenchConfig()
        self.policy = policy or PolicyContext()
        self.entities = entities or EntityGraph()
        self.snc = tuple(snc)
        self.perspectives = perspectives or PerspectiveInventory()
        self._cache = {}

    def run(self, as_of, *, previous=None, institutions=None, forms=None):
        from ..datasets import RegulatoryAnalysisDataset

        as_of = instant(as_of).isoformat()
        if previous and instant(previous.as_of) > instant(as_of):
            raise ValueError("Cannot run before the previous review date")
        cfg = self.config
        identity = software_identity()
        raw = self.store.known_records(as_of)
        scope_records = tuple(
            o
            for o in raw
            if (institutions is None or o.institution_id in institutions)
            and (forms is None or o.form in forms)
        )
        source_hashes = sorted({o.source_hash for o in scope_records if o.source_hash})
        unhashed_records = tuple(o for o in scope_records if not o.source_hash)
        if unhashed_records:
            # Synthetic/analyst-entered inputs may have no upstream file. Preserve a
            # content hash of their complete canonical records as a distinct basis.
            source_hashes.append(stable_id(unhashed_records))
        elif not scope_records:
            source_hashes.append(stable_id(scope_records))
        source_hashes = sorted(set(source_hashes))
        quality = list(reconcile_store(self.store, as_of))
        pairs = sorted({(o.institution_id, o.form) for o in scope_records})
        analyses = []
        signals = []
        facts_by_id = {o.observation_id: o for o in raw}
        current_observations = []
        all_policy = {}
        drr_edges = []
        failed_edges = []
        dispositions = self.ledger.latest_reviews(as_of=as_of)

        def source_facts(ids):
            complete = set(ids)
            pending = list(ids)
            while pending:
                record = facts_by_id[pending.pop()]
                if record.lineage:
                    for parent in record.lineage.input_ids:
                        if parent not in complete:
                            complete.add(parent)
                            pending.append(parent)
            return [
                dict(observation_id=oid, **canonical(facts_by_id[oid])) for oid in sorted(complete)
            ]

        for institution, form in pairs:
            try:
                dataset = RegulatoryAnalysisDataset.from_vintage_store(
                    self.store,
                    self.registry,
                    institution_id=institution,
                    form=form,
                    as_of=as_of,
                    allow_synthetic=cfg.allow_synthetic,
                    peer_group=self.cohort.name if self.cohort else None,
                )
            except ValueError as exc:
                quality.append(
                    DataQualityException(
                        "important",
                        institution,
                        "dataset",
                        as_of[:10],
                        "blocked_dataset",
                        (),
                        str(exc),
                    )
                )
                continue
            current_observations += [
                ("|".join(o.key), o.observation_id) for o in dataset.observations
            ]
            dq = reconcile_dataset(dataset)
            quality.extend(dq)
            events = {
                m: self.policy.relevant(
                    as_of=as_of, form=form, metric=m, institution_class=cfg.institution_class
                )
                for m in dataset.variable_names
            }
            for rows in events.values():
                for event in rows:
                    all_policy[event.event_id] = event
            breakpoints = {
                m: self.policy.breakpoints(
                    as_of=as_of,
                    form=form,
                    metric=m,
                    dates=dataset.dates,
                    institution_class=cfg.institution_class,
                )
                for m in dataset.variable_names
            }
            changes = detect_material_changes(dataset, breakpoints=breakpoints)
            baselines = run_baselines(dataset, cfg.baseline, breakpoints=breakpoints)
            active_breaks = any(
                p in dataset.dates[-cfg.drr.lookback :]
                for points in breakpoints.values()
                for p in points
            )
            input_key = stable_id(
                dict(
                    observations=dataset.observations,
                    semantics=dataset.metric_metadata,
                    config=cfg,
                    breakpoints=breakpoints,
                )
            )
            if input_key in self._cache:
                drr = json.loads(self._cache[input_key])
            elif not cfg.enable_drr:
                drr = {
                    "status": "disabled",
                    "structural_alert": False,
                    "limitation": "DRR disabled by analyst configuration",
                }
            elif active_breaks:
                drr = {
                    "status": "unavailable",
                    "structural_alert": False,
                    "limitation": "Reporting comparability break inside the analysis window",
                }
            else:
                drr = LFBORegimeAnalyzer(cfg.drr).analyze(dataset)
            self._cache[input_key] = canonical_json(drr)
            # Run time is in the passport; identical evidence stays identical across review dates.
            drr_evidence = {k: v for k, v in drr.items() if k != "as_of"}
            peer_rows = {}
            changed = []
            baseline_flag = any(b.flagged for b in baselines) or any(c.flagged for c in changes)
            drr_evidence["incremental_alert"] = (
                bool(drr.get("structural_alert")) and not baseline_flag
            )
            structural_robustness = None
            if drr.get("structural_alert"):
                structural_robustness = falsify_drr_signal(dataset, cfg.drr)
            for change in changes:
                peer = None
                if self.cohort:
                    try:
                        peer = analyze_peers(
                            self.store,
                            self.registry,
                            self.cohort,
                            institution=institution,
                            form=form,
                            metric=change.metric,
                            period=change.period,
                            as_of=as_of,
                            allow_synthetic=cfg.allow_synthetic,
                        )
                        peer_rows[change.metric] = peer
                        change = attach_peer_context(change, peer)
                    except ValueError as exc:
                        change = replace(
                            change,
                            comparison_limitations=change.comparison_limitations
                            + ("Peer comparison unavailable: " + str(exc),),
                        )
                changed.append(change)
                if not change.flagged:
                    continue
                robust = falsify_material_change(dataset, change.metric)
                ids = set(change.evidence_references)
                ids.update(peer.evidence if peer else ())
                # DRR evidence uses all dimensions, whose source facts must be included too.
                ids.update(o.observation_id for o in dataset.observations)
                cbase = [b for b in baselines if b.metric == change.metric]
                peer_payload = (
                    canonical(peer)
                    if peer
                    else {"status": "unavailable", "limitation": "No eligible explicit peer cohort"}
                )
                peer_payload.pop("as_of", None)
                claim = (
                    f"{dataset.institution_name}: {change.label} changed {change.raw_change:+.4g} {change.unit}"
                    if change.raw_change is not None
                    else f"{dataset.institution_name}: {change.label} is unusual relative to comparable history"
                )
                facts = source_facts(ids)
                evidence_available_as_of = max(
                    [instant(f["available_as_of"]) for f in facts]
                    + [instant(event.available_as_of) for event in events[change.metric]]
                ).isoformat()
                packet = EvidenceEntry.create(
                    claim=claim,
                    institution=institution,
                    institution_name=dataset.institution_name,
                    form=form,
                    metric=change.metric,
                    label=change.label,
                    period=change.period,
                    source_facts=facts,
                    evidence_available_as_of=evidence_available_as_of,
                    calculation={
                        "formula": "QoQ = current - prior quarter; percent = 100 * QoQ / abs(prior); robust_z = (current - median(prior))/(1.4826*MAD(prior))",
                        "software_version": identity["version"],
                        "git_commit": identity["commit"],
                        "source_code_sha256": identity["source_code_sha256"],
                        "input_ids": sorted(ids),
                        "parameters": {"lookback": 12, "percent_threshold": 5, "z_threshold": 3},
                        "unit": change.unit,
                    },
                    historical_context=canonical(change),
                    peer_context=peer_payload,
                    baseline_evidence=canonical(cbase),
                    drr_evidence=dict(drr_evidence, robustness=canonical(structural_robustness)),
                    robustness=canonical(robust),
                    contradictory_evidence=list(robust.contradictory_evidence)
                    + (
                        ["Simple statistical baselines did not flag this metric"]
                        if not any(b.flagged for b in cbase)
                        else []
                    ),
                    policy_context=canonical(events[change.metric]),
                    data_limitations=list(change.comparison_limitations)
                    + list(dataset.data_quality_flags),
                    data_quality=canonical([e for e in dq if e.metric == change.metric]),
                    domain=next(
                        m["domain"]
                        for k, m in dataset.metric_metadata.items()
                        if k.startswith(change.metric + ":")
                    ),
                )
                oid = self.ledger.append(packet)
                review = dispositions.get(oid)
                signals.append(
                    MonitoringSignal(
                        f"{institution}:{form}:{change.metric}:material_change",
                        institution,
                        change.metric,
                        change.period,
                        claim,
                        oid,
                        change.materiality_score,
                        change.data_confidence,
                        change.persistence,
                        abs((peer.percentile if peer and peer.percentile is not None else 50) - 50)
                        / 50,
                        (
                            robust.percentage_surviving / 100
                            if robust.percentage_surviving is not None
                            else None
                        ),
                        bool(drr_evidence["incremental_alert"]),
                        review.disposition.value if review else "unresolved",
                    )
                )
            if drr_evidence["incremental_alert"]:
                ids = {o.observation_id for o in dataset.observations}
                claim = f"{dataset.institution_name}: multivariate structural change candidate"
                facts = source_facts(ids)
                packet = EvidenceEntry.create(
                    claim=claim,
                    institution=institution,
                    institution_name=dataset.institution_name,
                    form=form,
                    metric="structural_surprise",
                    period=dataset.reporting_period,
                    source_facts=facts,
                    evidence_available_as_of=max(
                        instant(f["available_as_of"]) for f in facts
                    ).isoformat(),
                    calculation={
                        "formula": "one-step regularized VAR innovation scored against prior innovation distribution",
                        "software_version": identity["version"],
                        "git_commit": identity["commit"],
                        "source_code_sha256": identity["source_code_sha256"],
                        "input_ids": sorted(ids),
                        "parameters": canonical(cfg.drr),
                    },
                    historical_context=drr.get("structural_surprise", {}),
                    peer_context={"status": "not evaluated for multivariate structural claim"},
                    baseline_evidence=canonical(baselines),
                    drr_evidence=drr_evidence,
                    robustness=canonical(structural_robustness),
                    contradictory_evidence=list(structural_robustness.contradictory_evidence),
                    policy_context=[],
                    data_limitations=drr.get("limitations", []),
                    domain="structural",
                )
                oid = self.ledger.append(packet)
                review = dispositions.get(oid)
                signals.append(
                    MonitoringSignal(
                        f"{institution}:{form}:structural_surprise",
                        institution,
                        "structural_surprise",
                        dataset.reporting_period,
                        claim,
                        oid,
                        40,
                        1,
                        robustness=(
                            structural_robustness.percentage_surviving / 100
                            if structural_robustness.percentage_surviving is not None
                            else None
                        ),
                        drr_incremental=True,
                        disposition=review.disposition.value if review else "unresolved",
                    )
                )
            for edge in drr.get("rooting", {}).get("significant_edges", []):
                drr_edges.append(
                    f"{institution}:{form}:{edge['source']}:{edge['target']}:{edge['lag']}"
                )
            if structural_robustness and structural_robustness.classification in {
                "fragile",
                "not supported",
            }:
                failed_edges.append(f"{institution}:{form}:structural_surprise")
            analyses.append(
                dict(
                    institution=institution,
                    name=dataset.institution_name,
                    form=form,
                    period=dataset.reporting_period,
                    as_of=as_of,
                    filing_vintages=dataset.filing_vintage,
                    metrics=dataset.metric_metadata,
                    source_ids=tuple(o.observation_id for o in dataset.observations),
                    source_hashes=dataset.source_hashes,
                    observations=canonical(dataset.observations),
                    changes=canonical(changed),
                    peers=canonical(peer_rows),
                    baselines=canonical(baselines),
                    drr=canonical(drr),
                    robustness=canonical(structural_robustness),
                    quality=canonical(dq),
                    history={m: dataset.frame[m].tolist() for m in dataset.variable_names},
                    dates=dataset.dates,
                )
            )
        state = ReviewState(
            as_of,
            tuple(sorted(current_observations)),
            tuple(sorted(signals, key=lambda s: s.key)),
            tuple(sorted(e.exception_id for e in quality)),
            tuple(sorted(all_policy)),
            tuple(sorted(r.relationship_id for r in self.entities.active(as_of=as_of))),
            tuple(sorted(drr_edges)),
            tuple(sorted(failed_edges)),
        )
        delta = compare_review_states(previous, state)
        attention = AttentionBudget(top_n=cfg.top_n).select(delta)
        perspective_snapshot = self.perspectives.snapshot(
            self.ledger, as_of=as_of, observations=self.store.known_records(as_of)
        )
        result = dict(
            as_of=as_of,
            state=canonical(state),
            state_id=state.state_id,
            delta=canonical(delta),
            attention=canonical(attention),
            analyses=analyses,
            quality=[dict(exception_id=e.exception_id, **canonical(e)) for e in quality],
            policy=canonical(tuple(all_policy.values())),
            entities={
                "entities": canonical(self.entities.entities),
                "relationships": canonical(self.entities.active(as_of=as_of)),
            },
            snc=analyze_public_snc(self.snc, as_of=as_of),
            feedback=canonical(evaluate_signal_usefulness(self.ledger, as_of=as_of)),
            config=canonical(cfg),
            model_risk_profile=canonical(cfg.model_risk_profile),
            model_risk_reference_basis=canonical(MODEL_RISK_REFERENCE_BASIS),
            scope=(
                "synthetic demonstration" if cfg.allow_synthetic else "public regulatory research"
            ),
            decision_boundary=DECISION_BOUNDARY,
            perspectives=perspective_snapshot,
            evidence={s.evidence_id: self.ledger.get(s.evidence_id).payload for s in signals},
        )
        monitoring_output_hash = stable_id(result)
        passport = AnalysisPassport.create(
            software_version=identity["version"],
            git_commit=identity["commit"],
            source_code_sha256=identity["source_code_sha256"],
            working_tree_modified=identity["working_tree_modified"],
            timestamp=as_of,
            as_of=as_of,
            filing_vintages=sorted({v for a in analyses for v in a["filing_vintages"]}),
            mapping_version=self.registry.version,
            institutions=sorted({a["institution"] for a in analyses}),
            peer_definition=canonical(self.cohort),
            variables=sorted({c["metric"] for a in analyses for c in a["changes"]}),
            transformations=[cfg.drr.transformation],
            drr_configuration=canonical(cfg.drr),
            baseline_configuration=canonical(cfg.baseline),
            robustness_configuration={
                "material_change": "8/12/24 quarters; differences; percent changes; historical omissions",
                "drr": "windows, transforms, scaling, lags, methods, correction, variable omissions",
            },
            random_seeds=[cfg.drr.seed],
            source_hashes=source_hashes,
            source_hash_basis={
                "upstream_artifacts": "SHA-256 supplied at ingestion",
                "unhashed_records": (
                    "SHA-256 of canonical source-record snapshot"
                    if unhashed_records
                    else "not applicable"
                ),
            },
            perspective_configuration=perspective_snapshot,
            model_risk_profile=canonical(cfg.model_risk_profile),
            model_risk_reference_basis=canonical(MODEL_RISK_REFERENCE_BASIS),
            output_hashes={"monitoring": monitoring_output_hash},
            output_hash_scope={
                "monitoring": "stable_id of the complete monitoring payload before the passport field is appended"
            },
            previous_state_id=previous.state_id if previous is not None else None,
        )
        result["passport"] = dict(analysis_id=passport.analysis_id, **passport.payload)
        return canonical(result), state, passport


def review_state_from_dict(value):
    if "state" in value:
        payload = value["state"]
        expected = value.get("state_id")
        if expected != stable_id(payload):
            raise ValueError("Previous review-state integrity check failed")
        value = payload
    return ReviewState(
        value["as_of"],
        tuple(tuple(p) for p in value.get("observations", ())),
        tuple(MonitoringSignal(**s) for s in value.get("signals", ())),
        tuple(value.get("exceptions", ())),
        tuple(value.get("policy_events", ())),
        tuple(value.get("entity_relationships", ())),
        tuple(value.get("drr_relationships", ())),
        tuple(value.get("failed_relationships", ())),
    )
