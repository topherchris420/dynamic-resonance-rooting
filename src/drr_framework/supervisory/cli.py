"""Local CLI and loopback-only analyst review server. No telemetry or cloud service."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys

from .briefs import generate_morning_brief, generate_lfbo_monitoring_brief
from .common import canonical, canonical_json, stable_id, write_immutable
from .demo import synthetic_monitoring_lab, synthetic_perspectives, CURRENT_REVIEW, PREVIOUS_REVIEW
from .evidence_ledger import AuditEvent, EvidenceLedger, AnalystReview
from .passport import dependency_inventory
from .peer_analysis import PeerGroupDefinition
from .perspectives import PerspectiveInventory
from .semantics import SemanticRegistry, bundled_registry
from .ui import render_workbench, STYLE, SCRIPT
from .vintage import VintageStore
from .workbench import MonitoringWorkbench, WorkbenchConfig, review_state_from_dict


def export_run(result, state, passport, ledger, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    directory = root / passport.analysis_id[:16]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    write_immutable(directory / "snapshot.json", canonical_json(result) + "\n")
    write_immutable(
        directory / "perspectives.json",
        canonical_json(result.get("perspectives", {})) + "\n",
    )
    write_immutable(
        directory / "review-state.json",
        canonical_json({"state": state, "state_id": state.state_id}) + "\n",
    )
    write_immutable(directory / "morning-brief.md", generate_morning_brief(result))
    write_immutable(directory / "index.html", render_workbench(result))
    for institution in sorted({a["institution"] for a in result["analyses"]}):
        # No analyst-supplied identifier is interpreted as a filesystem path.
        name = "institution-" + stable_id(institution)[:16] + ".md"
        write_immutable(directory / name, generate_lfbo_monitoring_brief(result, institution))
    evidence_ids = set(result.get("evidence", {}))
    for perspective in result.get("perspectives", {}).get("perspectives", []):
        evidence_ids.update(perspective["evidence_ids"])
    ledger.export(
        directory / "evidence",
        as_of=result["as_of"],
        evidence_ids=tuple(sorted(evidence_ids)),
    )
    passport.export(directory)
    write_immutable(
        directory / "dependency-inventory.json", canonical_json(dependency_inventory()) + "\n"
    )
    return directory


def make_review_server(workbench, result, *, previous=None, port=8765, role="analyst"):
    if role not in {"viewer", "analyst"} or not 0 <= port <= 65535:
        raise ValueError("Invalid local server role or port")
    token = secrets.token_urlsafe(32)
    current = [result]
    import base64

    style_hash = base64.b64encode(hashlib.sha256(STYLE.encode()).digest()).decode()
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Audit records are structured in SQLite; never log user text or tokens.
            pass

        def respond(self, status, body, content_type="application/json"):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                f"default-src 'none'; style-src 'sha256-{style_hash}'; script-src 'sha256-{script_hash}'; connect-src 'self'; img-src 'self' data:; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def allowed_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def audit_denial(self, reason, *, outcome="denied"):
            workbench.ledger.record_audit_event(
                AuditEvent(
                    datetime.now(timezone.utc).isoformat(),
                    "review_request",
                    outcome,
                    "loopback-client",
                    details=(("reason", reason),),
                )
            )

        def do_GET(self):
            if not self.allowed_host():
                return self.respond(403, '{"error":"Invalid local host"}')
            if self.path == "/":
                return self.respond(
                    200,
                    render_workbench(current[0], token=token, editable=role == "analyst"),
                    "text/html",
                )
            if self.path == "/api/snapshot":
                return self.respond(200, canonical_json(current[0]))
            if self.path == "/health":
                return self.respond(200, '{"status":"ok","mode":"local"}')
            return self.respond(404, '{"error":"Not found"}')

        def do_POST(self):
            if not self.allowed_host() or role != "analyst":
                self.audit_denial("access_control")
                return self.respond(403, '{"error":"Analyst access required"}')
            if self.path != "/api/review":
                return self.respond(404, '{"error":"Not found"}')
            expected_origins = {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if self.headers.get("Origin") not in expected_origins or not secrets.compare_digest(
                self.headers.get("X-Review-Token", ""), token
            ):
                self.audit_denial("origin_or_token")
                return self.respond(403, '{"error":"Invalid review request origin or token"}')
            if self.headers.get("Content-Type") != "application/json":
                self.audit_denial("content_type", outcome="failed")
                return self.respond(415, '{"error":"JSON required"}')
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.audit_denial("content_length", outcome="failed")
                return self.respond(400, '{"error":"Invalid length"}')
            if not 0 < length <= 16384:
                self.audit_denial("content_length", outcome="failed")
                return self.respond(413, '{"error":"Review exceeds size limit"}')
            try:
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) - {
                    "evidence_id",
                    "disposition",
                    "reviewer",
                    "rationale",
                    "review_minutes",
                }:
                    raise ValueError("Unexpected review fields")
                if len(data.get("reviewer", "")) > 120 or len(data.get("rationale", "")) > 4000:
                    raise ValueError("Review text is too long")
                if data.get("evidence_id") not in current[0]["evidence"]:
                    raise ValueError("Evidence is outside the current review")
                reviewed_at = datetime.now(timezone.utc).isoformat()
                review = AnalystReview(
                    data["evidence_id"],
                    data["disposition"],
                    data["reviewer"],
                    data["rationale"],
                    reviewed_at,
                    float(data.get("review_minutes", 0)),
                )
                oid = workbench.ledger.review(review)
                # Replay calculations at their original as-of; human reviews are displayed
                # as today's activity and never injected into historical analytical inputs.
                for s in current[0]["state"]["signals"]:
                    if s["evidence_id"] == review.evidence_id:
                        s["disposition"] = review.disposition.value
                for group in ("review_first", "deferred"):
                    current[0]["attention"][group] = [
                        item
                        for item in current[0]["attention"][group]
                        if not (
                            item["signal"]["evidence_id"] == review.evidence_id
                            and review.disposition.value in {"explained", "noisy", "dismissed"}
                        )
                    ]
                from .feedback import evaluate_signal_usefulness

                current[0]["feedback"] = canonical(evaluate_signal_usefulness(workbench.ledger))
                return self.respond(200, canonical_json({"review_id": oid, "recorded": True}))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.audit_denial(type(exc).__name__, outcome="failed")
                return self.respond(400, '{"error":"Invalid review payload"}')

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LFBO public-data research and local analyst workbench"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--demo", action="store_true", help="Run the deterministic synthetic monitoring lab"
    )
    source.add_argument(
        "--input", type=Path, help="Normalized public regulatory observations (CSV/JSONL)"
    )
    parser.add_argument("--registry", type=Path, help="Verified semantic registry JSON")
    parser.add_argument("--cohort", type=Path, help="Explicit PeerGroupDefinition JSON")
    parser.add_argument(
        "--perspectives", type=Path, help="Scoped perspective/disagreement JSON inventory"
    )
    parser.add_argument("--as-of", help="UTC data availability cutoff")
    parser.add_argument("--previous", type=Path, help="Previous review-state.json")
    parser.add_argument("--output", type=Path, default=Path("results/lfbo-monitoring"))
    parser.add_argument("--no-drr", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--serve", action="store_true", help="Serve the local review interface at 127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--role",
        choices=("viewer", "analyst"),
        default="analyst",
        help="Local process capability; not institutional identity management",
    )
    args = parser.parse_args(argv)
    if args.input and not args.as_of:
        parser.error("--as-of is required for public regulatory data")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        args.output.chmod(0o700)
    except OSError:
        pass
    ledger = EvidenceLedger(args.output / "ledger.sqlite")
    if args.demo:
        store, registry, cohort, policy, entities = synthetic_monitoring_lab()
        perspectives = synthetic_perspectives(store)
    else:
        store = VintageStore.from_file(args.input)
        registry = (
            SemanticRegistry.from_json(str(args.registry)) if args.registry else bundled_registry()
        )
        cohort = PeerGroupDefinition(**json.loads(args.cohort.read_text())) if args.cohort else None
        policy = entities = None
        perspectives = (
            PerspectiveInventory.from_json(args.perspectives) if args.perspectives else None
        )
    cfg = WorkbenchConfig(allow_synthetic=args.demo, enable_drr=not args.no_drr, top_n=args.top_n)
    workbench = MonitoringWorkbench(
        store,
        registry,
        ledger,
        cohort=cohort,
        config=cfg,
        policy=policy,
        entities=entities,
        perspectives=perspectives,
    )
    previous = (
        review_state_from_dict(json.loads(args.previous.read_text())) if args.previous else None
    )
    if args.demo and previous is None:
        _, previous, _ = workbench.run(PREVIOUS_REVIEW)
    result, state, passport = workbench.run(args.as_of or CURRENT_REVIEW, previous=previous)
    directory = export_run(result, state, passport, ledger, args.output)
    print(generate_morning_brief(result))
    print(f"Workbench: {directory/'index.html'}")
    if args.serve:
        server = make_review_server(
            workbench, result, previous=previous, port=args.port, role=args.role
        )
        print(f"Local review: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
