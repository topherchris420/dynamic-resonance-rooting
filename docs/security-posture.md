# LFBO Workbench Security Posture

## Trust boundary

The workbench is local-first and makes no network request during import, ingestion,
analysis, briefing, or review. Public files must be obtained and approved outside the
runtime, then supplied locally with their authoritative HTTPS citation and SHA-256.
There is no telemetry, cloud dependency, hidden model call, or required LLM.

The optional review server binds to IPv4 loopback only. It validates the `Host` and
`Origin`, requires an unpredictable per-process review token, accepts JSON only, caps
request size, uses a strict Content Security Policy, disables caching and referrers,
and escapes analyst-facing HTML. `viewer` and `analyst` are local process capabilities,
not user authentication or institutional RBAC. Do not expose this server through a
reverse proxy or shared host without adding approved identity, authorization, TLS,
session, rate-limit, and operations controls.

## Integrity and auditability

- Source artifacts carry SHA-256 digests; normalized records, evidence entries,
  reviews, audit events, review states, semantic registries, and passports have stable
  content-derived IDs.
- Evidence is immutable and analyst dispositions are append-only. SQLite deny-update/
  deny-delete triggers protect the evidence, review, and audit tables in the local
  ledger. Security/operation audit events reject token, secret, credential, and
  rationale fields; analyst rationale remains confined to the review table.
- Static exports use exclusive creation and reject divergent overwrites.
- Historical exports are cut off at the run's `as_of` timestamp and include only the
  run's evidence IDs. Review-state exports use a `{state, state_id}` envelope, and
  the loader verifies the content hash before using a prior state.
- Ledger databases and generated artifacts are created with private `0600` files and
  `0700` directories where the workflow owns the path. The loopback server is a
  trusted single-user interface: `/api/snapshot` is intentionally read-only but not
  an identity boundary.
- Hashes detect accidental or unauthorized content change. They are not signatures,
  identity attestations, trusted timestamps, or tamper-proof storage. A production
  deployment should place exports and SQLite ledgers on approved append-only storage,
  sign release and analysis manifests, protect encryption keys, and forward audit
  records to an approved logging service.

## Data classification

The shipped workflow is designed for public regulatory data and deterministic
synthetic examples. It does not create a protected environment for confidential
supervisory information, examination workpapers, borrower-level SNC information,
credentials, or personally identifiable information. Admission of nonpublic data
requires an independent security architecture and authorization.

## Supply chain

CI runs tests across supported Python versions, lint/format checks, a high-confidence
tracked-file secret scan, `pip-audit`, and CycloneDX SBOM generation. Dependabot is
configured for monthly Python and GitHub Actions review. The SBOM is uploaded as a CI
artifact. These checks support review; they do not establish that dependencies are
vulnerability-free.

The project is a reusable Python library and therefore publishes compatible dependency
ranges rather than a single cross-platform lock. A deployment owner should resolve and
pin an environment for the target Python/OS, retain hashes and the generated SBOM, and
re-run the full suite and dependency audit before promotion.

## Operator checklist

1. Verify source URLs, file hashes, semantic registry approval, as-of cutoff, and peer
   definition before analysis.
2. Run from a clean, reviewed commit and retain the analysis passport.
3. Use `--role viewer` when dispositions are not required.
4. Keep the ledger and exports in an access-controlled local directory with backups.
5. Review dependency and secret-scan results; treat scanner outages as a failed gate.
6. Confirm that no reporting-definition or entity-perimeter exception is unresolved.
7. Do not interpret a quiet queue when the Morning Brief reports data blockers.
8. Preserve the explicit analytical boundary in every downstream presentation.

No government authorization, security accreditation, production suitability, or
supervisory methodology approval is claimed.
