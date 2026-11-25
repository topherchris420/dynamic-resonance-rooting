"""Optional typed judgment about an evidence packet.

DRR measures, tests, and prioritizes before this module runs. A judgment provider
characterizes bounded questions about evidence that is already immutable and already
selected by ``AttentionBudget``. It does not compute resonance, baselines, significance,
robustness, materiality, or attention rank, and it does not record an analyst disposition.

Remote judgment is off unless ``WorkbenchConfig.judgment_enabled`` is set. The default
local-only path makes no judgment network call.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from .common import canonical, canonical_json, sha256_hex, stable_id
from .model_risk import ModelRiskProfile

QUESTION_SET_VERSION = "evidence-packet-v1"
JUDGMENT_POLICY_VERSION = "v2"
STATE_SCHEMA_VERSION = "judgment-state-v1"
RESULT_SCHEMA_VERSION = "judgment-result-v1"
OVERLAY_SCHEMA_VERSION = "judgment-overlay-v1"
BUNDLE_SCHEMA_VERSION = "judgment-bundle-v1"
POST_HOC_JUDGMENT_EVALUATION = "POST-HOC JUDGMENT EVALUATION"
NOUL_MATERIAL_THRESHOLD = 0.5
JUDGMENT_STATE_MAX_CHARACTERS = 48000
_SCOPE = (
    "Use only the supplied evidence packet. Do not assess whether an institution is "
    "safe, unsafe, well managed, poorly managed, high risk, or deserving of a supervisory "
    "action. Do not decide whether the analytical method is correct."
)
_READS = (
    "claim",
    "source_facts",
    "calculation",
    "historical_context",
    "peer_context",
    "baseline_evidence",
    "drr_evidence",
    "robustness",
    "contradictory_evidence",
    "data_limitations",
    "policy_context",
    "decision_boundary",
)
_PROVIDER_STATUSES = frozenset(
    {
        "completed",
        "disabled",
        "unavailable",
        "authentication_error",
        "timeout",
        "rate_limit",
        "network_error",
        "service_failure",
        "malformed_response",
        "missing_fields",
        "unsupported_primitive",
        "oversized_state",
        "serialization_failure",
        "missing_credentials",
    }
)
_POLICY_OUTCOMES = {
    "v1": frozenset({"READY", "REVIEW_CAREFULLY", "INSUFFICIENT_EVIDENCE", "JUDGMENT_UNAVAILABLE"}),
    "v2": frozenset(
        {
            "EVIDENCE_REVIEWABLE",
            "REVIEW_CAREFULLY",
            "INSUFFICIENT_EVIDENCE",
            "JUDGMENT_UNAVAILABLE",
        }
    ),
}
_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|bearer|credential|password|secret|token|rationale|reviewer)",
    re.IGNORECASE,
)
_FACT_KEYS = (
    "observation_id",
    "institution_id",
    "institution_name",
    "form",
    "metric",
    "reporting_period",
    "value",
    "unit",
    "source",
    "provenance",
    "source_vintage",
    "available_as_of",
    "source_hash",
    "supersedes",
)
_CALC_KEYS = (
    "formula",
    "software_version",
    "git_commit",
    "source_code_sha256",
    "input_ids",
    "parameters",
    "unit",
)
_TRUNCATE_FIELDS = (
    "drr_evidence",
    "baseline_evidence",
    "historical_context",
    "peer_context",
    "source_facts",
    "policy_context",
    "robustness",
    "contradictory_evidence",
    "data_limitations",
    "calculation",
)
_REPLAY_NOTE = "Replay uses recorded judgment artifacts. Do not re-query the judgment provider."


def _clock_now():
    return datetime.now(timezone.utc).isoformat()


def active_secret_values():
    """Local credential material that must never enter a judgment artifact."""
    found = []
    raw = os.environ.get("TYPESAFE_API_KEY", "")
    if isinstance(raw, str):
        stripped = raw.strip()
        if len(stripped) >= 8:
            found.append(stripped)
    return tuple(found)


def assert_no_secrets(text, secrets=()):
    """Refuse a persisted or transmitted string that contains a local credential."""
    for secret in secrets or active_secret_values():
        if secret and secret in text:
            raise ValueError("Judgment artifact contains a credential")
    return text


def _sensitive_key(key):
    return bool(_SENSITIVE_KEY.search(str(key)))


def _redact_text(value, secrets):
    text = value
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]{8,}", "Bearer [redacted]", text)


def _scrub(value, secrets):
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, Mapping):
        return {
            str(key): _scrub(item, secrets)
            for key, item in value.items()
            if not _sensitive_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(item, secrets) for item in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise TypeError(f"Unsupported judgment state value: {type(value).__name__}")


@dataclass(frozen=True)
class QuestionSpec:
    """One atomic question. The id is for code; the model sees ``instructions``."""

    question_id: str
    primitive: str
    instructions: Mapping
    criteria: object

    def to_api(self):
        body = {"type": self.primitive, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


def _question(question_id, primitive, text, criteria):
    return QuestionSpec(
        question_id,
        primitive,
        {"question": text, "scope": _SCOPE, "read": list(_READS)},
        criteria,
    )


EVIDENCE_PACKET_QUESTIONS: Tuple[QuestionSpec, ...] = (
    _question(
        "evidence_adequacy",
        "choice",
        "Given only the supplied evidence packet, is there enough evidence to meaningfully evaluate the stated claim?",
        {
            "adequate": "The supplied facts, calculation, comparisons, robustness results, and limitations are sufficient for a human analyst to evaluate the stated claim.",
            "limited": "Relevant evidence is present, but gaps or limitations reduce how far the stated claim can be evaluated.",
            "insufficient": "The supplied packet does not contain enough evidence to meaningfully evaluate the stated claim.",
        },
    ),
    _question(
        "scope_overreach",
        "noul",
        "The stated claim materially goes beyond what the supplied evidence establishes.",
        {
            "true": "The stated claim asserts more than the supplied facts, calculations, and limitations establish.",
            "false": "The stated claim stays within what the supplied evidence establishes.",
        },
    ),
    _question(
        "contradiction_material",
        "noul",
        "The supplied contradictory evidence is materially important to interpreting this claim.",
        {
            "true": "Contradictory evidence in the packet is materially important to interpreting the stated claim.",
            "false": "Contradictory evidence is absent or not materially important to interpreting the stated claim.",
        },
    ),
    _question(
        "review_complexity",
        "choice",
        "How complex is this evidence packet for a human analyst to review?",
        {
            "routine": "A trained analyst can review this packet with ordinary effort.",
            "moderate": "The packet needs careful reading across several evidence sections.",
            "complex": "The packet is dense, multi-part, or difficult for a human analyst to review carefully.",
        },
    ),
    _question(
        "limitations_material",
        "noul",
        "The documented limitations materially constrain interpretation of this claim.",
        {
            "true": "The documented limitations materially constrain interpretation of the stated claim.",
            "false": "The documented limitations do not materially constrain interpretation of the stated claim.",
        },
    ),
    _question(
        "additional_review_needed",
        "noul",
        "The evidence contains enough ambiguity, disagreement, limitation, or uncertainty that additional human scrutiny is warranted.",
        {
            "true": "Ambiguity, disagreement, limitation, or uncertainty in the packet warrants additional human scrutiny.",
            "false": "The packet does not show enough ambiguity, disagreement, limitation, or uncertainty to warrant additional human scrutiny beyond ordinary review.",
        },
    ),
)
_QUESTIONS_BY_ID = {question.question_id: question for question in EVIDENCE_PACKET_QUESTIONS}


def question_api_payload(questions=EVIDENCE_PACKET_QUESTIONS):
    """Dict form accepted by ``TypeSafeClient.system_one``."""
    return {question.question_id: question.to_api() for question in questions}


@dataclass(frozen=True)
class JudgmentState:
    """Canonical, bounded packet for one evidence entry."""

    payload_json: str
    state_id: str
    evidence_id: str
    analysis_id: Optional[str]
    truncated: bool
    sendable: bool
    failure: Optional[str] = None

    def __post_init__(self):
        sha256_hex(self.state_id)
        if stable_id(json.loads(self.payload_json)) != self.state_id:
            raise ValueError("Judgment state hash does not match its payload")

    @property
    def payload(self):
        return json.loads(self.payload_json)


@dataclass(frozen=True)
class TypedAnswer:
    """One primitive answer. Noul carries no separate confidence."""

    question_id: str
    primitive: str
    choice: Optional[str] = None
    score: Optional[float] = None
    noul: Optional[float] = None
    probabilities: Optional[Tuple[Tuple[str, float], ...]] = None
    confidence: Optional[float] = None
    legend: Optional[Tuple[Tuple[str, object], ...]] = None

    def __post_init__(self):
        if self.primitive not in {"choice", "score", "noul"}:
            raise ValueError("Unsupported judgment primitive")
        if self.probabilities is not None:
            object.__setattr__(
                self,
                "probabilities",
                tuple((str(key), float(value)) for key, value in self.probabilities),
            )
        if self.legend is not None:
            object.__setattr__(
                self, "legend", tuple((str(key), value) for key, value in self.legend)
            )
        if self.primitive == "noul" and self.confidence is not None:
            raise ValueError("Noul answers do not carry a confidence value")


@dataclass(frozen=True)
class JudgmentResult:
    schema_version: str
    provider: str
    model: str
    provider_model_version: Optional[str]
    question_set_version: str
    evidence_id: str
    analysis_id: Optional[str]
    state_hash: str
    state_truncated: bool
    answers: Tuple[TypedAnswer, ...]
    requested_at: str
    completed_at: str
    latency_ms: float
    provider_status: str
    limitations: Tuple[str, ...]

    def __post_init__(self):
        sha256_hex(self.state_hash)
        if self.provider_status not in _PROVIDER_STATUSES:
            raise ValueError("Invalid judgment provider status")
        if not self.evidence_id or not self.schema_version:
            raise ValueError("Judgment result requires evidence and a schema version")
        object.__setattr__(self, "answers", tuple(self.answers))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))
        if any(not isinstance(answer, TypedAnswer) for answer in self.answers):
            raise ValueError("Judgment answers must be typed")
        try:
            latency = float(self.latency_ms)
        except (TypeError, ValueError):
            raise ValueError("Judgment latency must be finite") from None
        if isinstance(self.latency_ms, bool) or latency < 0 or latency != latency:
            raise ValueError("Judgment latency must be finite")
        object.__setattr__(self, "latency_ms", latency)

    @property
    def judgment_id(self):
        """Content hash. Timestamps and latency are audit fields, not identity."""
        body = canonical(self)
        for key in ("requested_at", "completed_at", "latency_ms"):
            body.pop(key, None)
        return stable_id(body)


@dataclass(frozen=True)
class JudgmentOverlay:
    evidence_id: str
    judgment_result: JudgmentResult
    review_complexity: Optional[str]
    warnings: Tuple[str, ...]
    policy_outcome: str
    policy_version: str

    def __post_init__(self):
        if self.evidence_id != self.judgment_result.evidence_id:
            raise ValueError("Judgment overlay is bound to one evidence record")
        allowed = _POLICY_OUTCOMES.get(self.policy_version)
        if allowed is None or self.policy_outcome not in allowed:
            raise ValueError("Invalid judgment policy outcome")
        if self.review_complexity not in {None, "routine", "moderate", "complex"}:
            raise ValueError("Invalid review complexity")
        object.__setattr__(self, "warnings", tuple(self.warnings))


class JudgmentResponseError(ValueError):
    def __init__(self, status, limitation):
        super().__init__(limitation)
        self.status = status
        self.limitation = limitation


def _pair_map(value):
    if not isinstance(value, Mapping) or not value:
        raise JudgmentResponseError("malformed_response", "Probabilities must be a nonempty object")
    pairs = []
    for key in sorted(value, key=str):
        try:
            number = float(value[key])
        except (TypeError, ValueError):
            raise JudgmentResponseError(
                "malformed_response", "Probability values must be numeric"
            ) from None
        if isinstance(value[key], bool) or not 0.0 <= number <= 1.0:
            raise JudgmentResponseError("malformed_response", "Probability values must be in [0,1]")
        pairs.append((str(key), number))
    return tuple(pairs)


def _confidence(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise JudgmentResponseError(
            "missing_fields", "Choice or score confidence is missing"
        ) from None
    if isinstance(value, bool) or not 0.0 <= number <= 1.0:
        raise JudgmentResponseError("malformed_response", "Confidence must be in [0,1]")
    return number


def _answer_dict(value):
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    kind = getattr(value, "type", None)
    if kind == "choice":
        return {
            "type": "choice",
            "choice": getattr(value, "choice", None),
            "probabilities": getattr(value, "probabilities", None),
            "confidence": getattr(value, "confidence", None),
        }
    if kind == "score":
        return {
            "type": "score",
            "score": getattr(value, "score", None),
            "probabilities": getattr(value, "probabilities", None),
            "confidence": getattr(value, "confidence", None),
            "legend": getattr(value, "legend", None),
        }
    if kind == "noul":
        return {"type": "noul", "noul": getattr(value, "noul", None)}
    raise JudgmentResponseError("unsupported_primitive", "Answer primitive is not supported")


def parse_typed_answer(question_id, value, questions=_QUESTIONS_BY_ID):
    """Parse one official Choice, Score, or Noul answer without inventing fields."""
    body = _answer_dict(value)
    kind = body.get("type")
    expected = questions.get(question_id)
    if kind not in {"choice", "score", "noul"}:
        raise JudgmentResponseError("unsupported_primitive", "Answer primitive is not supported")
    if expected is not None and expected.primitive != kind:
        raise JudgmentResponseError(
            "unsupported_primitive", "Answer primitive does not match the question"
        )
    if kind == "noul":
        if "noul" not in body or body.get("noul") is None:
            raise JudgmentResponseError("missing_fields", "Noul answer is missing noul")
        try:
            probability = float(body["noul"])
        except (TypeError, ValueError):
            raise JudgmentResponseError("malformed_response", "Noul must be numeric") from None
        if isinstance(body["noul"], bool) or not 0.0 <= probability <= 1.0:
            raise JudgmentResponseError("malformed_response", "Noul must be in [0,1]")
        return TypedAnswer(question_id, "noul", noul=probability, confidence=None)
    if kind == "choice":
        if "choice" not in body or "probabilities" not in body or "confidence" not in body:
            raise JudgmentResponseError("missing_fields", "Choice answer is incomplete")
        probabilities = _pair_map(body["probabilities"])
        selected = body["choice"]
        if not isinstance(selected, str) or selected not in dict(probabilities):
            raise JudgmentResponseError("malformed_response", "Choice is outside its probabilities")
        if expected is not None and selected not in expected.criteria:
            raise JudgmentResponseError(
                "malformed_response", "Choice is outside the question criteria"
            )
        return TypedAnswer(
            question_id,
            "choice",
            choice=selected,
            probabilities=probabilities,
            confidence=_confidence(body["confidence"]),
        )
    if "score" not in body or "probabilities" not in body or "confidence" not in body:
        raise JudgmentResponseError("missing_fields", "Score answer is incomplete")
    try:
        score = float(body["score"])
    except (TypeError, ValueError):
        raise JudgmentResponseError("malformed_response", "Score must be numeric") from None
    if isinstance(body["score"], bool) or score != score:
        raise JudgmentResponseError("malformed_response", "Score must be numeric")
    legend = body.get("legend")
    legend_pairs = None
    if legend is not None:
        if not isinstance(legend, Mapping):
            raise JudgmentResponseError("malformed_response", "Score legend must be an object")
        legend_pairs = tuple((str(key), legend[key]) for key in sorted(legend, key=str))
    return TypedAnswer(
        question_id,
        "score",
        score=score,
        probabilities=_pair_map(body["probabilities"]),
        confidence=_confidence(body["confidence"]),
        legend=legend_pairs,
    )


def parse_system_one_response(payload, questions=EVIDENCE_PACKET_QUESTIONS):
    """Parse a System One response object or dict. Question ids stay with our code."""
    if hasattr(payload, "model") and hasattr(payload, "answers"):
        model = payload.model
        raw_answers = payload.answers
    elif isinstance(payload, Mapping):
        model = payload.get("model")
        raw_answers = payload.get("answers")
    else:
        raise JudgmentResponseError("malformed_response", "Response is not a System One object")
    if not isinstance(model, str) or not model.strip():
        raise JudgmentResponseError("missing_fields", "Response model version is missing")
    if not isinstance(raw_answers, Mapping):
        raise JudgmentResponseError("missing_fields", "Response answers are missing")
    expected = tuple(questions)
    missing = [
        question.question_id for question in expected if question.question_id not in raw_answers
    ]
    if missing:
        raise JudgmentResponseError("missing_fields", "Response is missing a requested answer")
    answers = tuple(
        parse_typed_answer(
            question.question_id, raw_answers[question.question_id], _QUESTIONS_BY_ID
        )
        for question in expected
    )
    return model, answers


def display_policy_outcome(outcome):
    """Human label for a policy token. Legacy v1 READY is shown as reviewable evidence."""
    token = "EVIDENCE_REVIEWABLE" if outcome == "READY" else (outcome or "JUDGMENT_UNAVAILABLE")
    return str(token).replace("_", " ")


def apply_judgment_policy(result, *, threshold=NOUL_MATERIAL_THRESHOLD):
    """Map typed answers to a presentation outcome. This does not rank attention."""
    if result.provider_status != "completed":
        return (
            "JUDGMENT_UNAVAILABLE",
            (f"provider_status={result.provider_status}",),
            None,
        )
    by_id = {answer.question_id: answer for answer in result.answers}
    if set(_QUESTIONS_BY_ID) - set(by_id):
        return ("JUDGMENT_UNAVAILABLE", ("incomplete question set",), None)
    adequacy = by_id["evidence_adequacy"]
    complexity = by_id["review_complexity"]
    if (
        adequacy.primitive != "choice"
        or adequacy.choice not in {"adequate", "limited", "insufficient"}
        or complexity.primitive != "choice"
        or complexity.choice not in {"routine", "moderate", "complex"}
    ):
        return ("JUDGMENT_UNAVAILABLE", ("invalid choice answer",), None)
    warnings = []
    for question_id in (
        "scope_overreach",
        "contradiction_material",
        "limitations_material",
        "additional_review_needed",
    ):
        answer = by_id[question_id]
        if answer.primitive != "noul" or answer.noul is None or answer.confidence is not None:
            return ("JUDGMENT_UNAVAILABLE", (f"invalid {question_id}",), None)
        if answer.noul > threshold:
            warnings.append(f"{question_id}={answer.noul:.4f} exceeds {threshold}")
    warning_text = tuple(warnings)
    if adequacy.choice == "insufficient":
        return ("INSUFFICIENT_EVIDENCE", warning_text, complexity.choice)
    if adequacy.choice == "limited" or warning_text:
        if adequacy.choice == "limited" and not warning_text:
            warning_text = ("evidence_adequacy=limited",)
        return ("REVIEW_CAREFULLY", warning_text, complexity.choice)
    return ("EVIDENCE_REVIEWABLE", (), complexity.choice)


def assemble_overlay(result):
    outcome, warnings, complexity = apply_judgment_policy(result)
    return JudgmentOverlay(
        result.evidence_id,
        result,
        complexity,
        warnings,
        outcome,
        JUDGMENT_POLICY_VERSION,
    )


def _safe_limitations(items, secrets):
    cleaned = []
    for item in items:
        text = _redact_text(str(item), secrets)
        if len(text) > 240:
            text = text[:240]
        cleaned.append(text)
    return tuple(cleaned)


def make_result(
    state,
    *,
    provider,
    model,
    provider_model_version,
    status,
    answers=(),
    limitations=(),
    requested_at,
    completed_at,
    latency_ms,
    secrets=(),
):
    return JudgmentResult(
        RESULT_SCHEMA_VERSION,
        provider,
        model,
        provider_model_version,
        QUESTION_SET_VERSION,
        state.evidence_id,
        state.analysis_id,
        state.state_id,
        state.truncated,
        tuple(sorted(answers, key=lambda answer: answer.question_id)),
        requested_at,
        completed_at,
        latency_ms,
        status,
        _safe_limitations(limitations, secrets),
    )


def classify_provider_failure(exc):
    """Map TypeSafe SDK failures by type or HTTP status. Messages are not retained."""
    name = type(exc).__name__
    status = getattr(exc, "status", None)
    if name == "TypeSafeAPITimeoutError" or isinstance(exc, TimeoutError):
        return "timeout"
    if name == "TypeSafeAuthenticationError" or status == 401:
        return "authentication_error"
    if name == "TypeSafeRateLimitError" or status == 429:
        return "rate_limit"
    if name == "TypeSafeAPIConnectionError" or isinstance(exc, ConnectionError):
        return "network_error"
    if name == "TypeSafeAPIResponseValidationError":
        return "missing_fields"
    if name in {"TypeSafeBadRequestError", "TypeSafeUnprocessableEntityError"} or status in {
        400,
        422,
    }:
        return "malformed_response"
    if name == "TypeSafeInternalServerError" or (isinstance(status, int) and 500 <= status <= 599):
        return "service_failure"
    if isinstance(exc, (json.JSONDecodeError, UnicodeError)):
        return "malformed_response"
    if name == "TypeSafeError":
        return "missing_credentials"
    return "service_failure"


_STATUS_LIMITATIONS = {
    "disabled": "Typed judgment provider is disabled.",
    "unavailable": "Typed judgment is unavailable.",
    "authentication_error": "Judgment provider authentication failed.",
    "timeout": "Judgment provider timed out.",
    "rate_limit": "Judgment provider rate limit was reached.",
    "network_error": "Judgment provider network request failed.",
    "service_failure": "Judgment provider failed.",
    "malformed_response": "Judgment provider returned a malformed response.",
    "missing_fields": "Judgment provider response is missing required fields.",
    "unsupported_primitive": "Judgment provider returned an unsupported primitive.",
    "oversized_state": "Judgment state exceeds the bounded packet size and was not sent.",
    "serialization_failure": "Judgment state could not be serialized and was not sent.",
    "missing_credentials": "TYPESAFE_API_KEY is missing or invalid.",
}


class JudgmentProvider:
    """Narrow evaluate interface. Workbench code depends on this, not on a vendor SDK."""

    name = "unspecified"

    def evaluate(self, state, questions):
        raise NotImplementedError


class DisabledJudgmentProvider(JudgmentProvider):
    name = "disabled"

    def __init__(self, clock: Callable[[], str] = _clock_now):
        self.clock = clock

    def evaluate(self, state, questions):
        moment = self.clock()
        return make_result(
            state,
            provider=self.name,
            model="",
            provider_model_version=None,
            status="disabled",
            limitations=(_STATUS_LIMITATIONS["disabled"],),
            requested_at=moment,
            completed_at=moment,
            latency_ms=0.0,
        )


class DeterministicMockJudgmentProvider(JudgmentProvider):
    """Offline answers that are a pure function of the state hash. Not a remote model."""

    name = "mock"

    def __init__(
        self,
        clock: Optional[Callable[[], str]] = None,
        script: Optional[Sequence[TypedAnswer]] = None,
    ):
        self.clock = clock if clock is not None else (lambda: "2026-01-15T00:00:00+00:00")
        self.script = None if script is None else tuple(script)
        self.calls: List[str] = []

    def evaluate(self, state, questions):
        self.calls.append(state.evidence_id)
        moment = self.clock()
        if not state.sendable:
            status = state.failure or "serialization_failure"
            return make_result(
                state,
                provider=self.name,
                model="deterministic-mock",
                provider_model_version="mock-v1",
                status=status,
                limitations=(_STATUS_LIMITATIONS.get(status, _STATUS_LIMITATIONS["unavailable"]),),
                requested_at=moment,
                completed_at=moment,
                latency_ms=0.0,
            )
        answers = self.script if self.script is not None else _mock_answers(state)
        return make_result(
            state,
            provider=self.name,
            model="deterministic-mock",
            provider_model_version="mock-v1",
            status="completed",
            answers=answers,
            limitations=("Deterministic offline stand-in. Not a TypeSafe model response.",),
            requested_at=moment,
            completed_at=moment,
            latency_ms=0.0,
        )


def _choice(question_id, selected, probabilities, confidence):
    ordered = tuple(sorted((str(key), float(value)) for key, value in probabilities.items()))
    return TypedAnswer(
        question_id,
        "choice",
        choice=selected,
        probabilities=ordered,
        confidence=confidence,
    )


def _noul(question_id, probability):
    return TypedAnswer(question_id, "noul", noul=probability, confidence=None)


def _mock_answers(state):
    nudge = int(state.state_id[:2], 16) / 10000.0
    probability = round(0.05 + nudge, 6)
    return (
        _choice(
            "evidence_adequacy",
            "adequate",
            {"adequate": 0.8, "insufficient": 0.05, "limited": 0.15},
            0.7,
        ),
        _noul("scope_overreach", probability),
        _noul("contradiction_material", probability),
        _choice(
            "review_complexity",
            "moderate",
            {"complex": 0.1, "moderate": 0.75, "routine": 0.15},
            0.66,
        ),
        _noul("limitations_material", probability),
        _noul("additional_review_needed", probability),
    )


class TypeSafeJudgmentProvider(JudgmentProvider):
    """Optional Jev client. Constructing it does not open a connection."""

    name = "typesafe"

    def __init__(
        self,
        *,
        model="jev-latest",
        timeout=10.0,
        client=None,
        clock: Callable[[], str] = _clock_now,
    ):
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", model):
            raise ValueError("Invalid TypeSafe model alias")
        self.model = model
        self.timeout = timeout
        self.client = client
        self.clock = clock

    def evaluate(self, state, questions):
        requested = self.clock()
        started = time.perf_counter()
        secrets = active_secret_values()

        def finish(status, answers=(), model_version=None, limitations=None):
            if limitations is None:
                limitations = () if status == "completed" else (_STATUS_LIMITATIONS[status],)
            return make_result(
                state,
                provider=self.name,
                model=self.model,
                provider_model_version=model_version,
                status=status,
                answers=answers,
                limitations=limitations,
                requested_at=requested,
                completed_at=self.clock(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                secrets=secrets,
            )

        if not state.sendable:
            status = (
                state.failure if state.failure in _STATUS_LIMITATIONS else "serialization_failure"
            )
            return finish(status)
        client = self.client
        opened = False
        try:
            if client is None:
                if not secrets and not str(os.environ.get("TYPESAFE_API_KEY", "")).strip():
                    return finish("missing_credentials")
                try:
                    from typesafe_sdk import RetryPolicy, TypeSafeClient
                except ImportError:
                    return finish(
                        "unavailable",
                        limitations=(
                            "typesafe-sdk is not installed; the evidence packet was not sent.",
                        ),
                    )
                try:
                    client = TypeSafeClient(
                        model=self.model, timeout=self.timeout, retry=_no_retry(RetryPolicy)
                    )
                except Exception as exc:
                    return finish(classify_provider_failure(exc))
                opened = True
            response = client.system_one(
                state=state.payload,
                questions=question_api_payload(questions),
                model=self.model,
                timeout=self.timeout,
                retry=_retry_override(client),
            )
            model_version, answers = parse_system_one_response(response, questions)
        except JudgmentResponseError as exc:
            return finish(exc.status, limitations=(exc.limitation,))
        except Exception as exc:
            return finish(classify_provider_failure(exc))
        finally:
            if opened:
                client.close()
        return finish("completed", answers=answers, model_version=model_version, limitations=())


def _no_retry(retry_policy):
    return retry_policy(max_retries=0)


def _retry_override(client):
    """Disable SDK retries when the caller did not inject a client policy."""
    try:
        from typesafe_sdk import RetryPolicy
    except ImportError:
        return None
    if client is None:
        return None
    return RetryPolicy(max_retries=0)


def build_judgment_provider(config, *, client=None, clock=None):
    if config.judgment_provider == "disabled":
        return DisabledJudgmentProvider(clock=clock or _clock_now)
    if config.judgment_provider == "mock":
        return DeterministicMockJudgmentProvider(clock=clock)
    if config.judgment_provider == "typesafe":
        return TypeSafeJudgmentProvider(
            model=config.judgment_model,
            timeout=config.judgment_timeout_seconds,
            client=client,
            clock=clock or _clock_now,
        )
    raise ValueError("Unknown judgment provider")


def _section(payload, key, secrets):
    if key not in payload:
        return None
    return _scrub(payload[key], secrets)


def build_judgment_state(
    entry, *, analysis_id=None, as_of=None, max_characters=JUDGMENT_STATE_MAX_CHARACTERS
):
    """Serialize one evidence entry. Unrelated ledger rows and analyst reviews are not inputs."""
    secrets = active_secret_values()
    try:
        payload = entry.payload
        facts = []
        for fact in payload.get("source_facts", ()):
            row = {}
            for key in _FACT_KEYS:
                if key in fact and not _sensitive_key(key):
                    row[key] = _scrub(fact[key], secrets)
            facts.append(row)
        calculation = payload.get("calculation") or {}
        calculation_body = {
            key: _scrub(calculation[key], secrets)
            for key in _CALC_KEYS
            if key in calculation and not _sensitive_key(key)
        }
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "evidence_id": entry.evidence_id,
            "analysis_id": analysis_id,
            "as_of": as_of or payload.get("evidence_available_as_of"),
            "source_ids": sorted(
                {row["observation_id"] for row in facts if row.get("observation_id")}
            ),
            "source_vintages": sorted(
                {row["source_vintage"] for row in facts if row.get("source_vintage")}
            ),
            "robustness_status": (
                (payload.get("robustness") or {}).get("classification")
                if isinstance(payload.get("robustness"), Mapping)
                else None
            ),
            "claim": _scrub(payload.get("claim"), secrets),
            "source_facts": facts,
            "calculation": calculation_body,
            "historical_context": _section(payload, "historical_context", secrets),
            "peer_context": _section(payload, "peer_context", secrets),
            "baseline_evidence": _section(payload, "baseline_evidence", secrets),
            "drr_evidence": _section(payload, "drr_evidence", secrets),
            "robustness": _section(payload, "robustness", secrets),
            "contradictory_evidence": _section(payload, "contradictory_evidence", secrets),
            "data_limitations": _section(payload, "data_limitations", secrets),
            "policy_context": _section(payload, "policy_context", secrets),
            "decision_boundary": _scrub(payload.get("decision_boundary"), secrets),
            "truncation": {
                "truncated": False,
                "max_characters": max_characters,
                "omitted": [],
            },
        }
        omitted = []
        if len(canonical_json(state)) > max_characters:
            for field in _TRUNCATE_FIELDS:
                if len(canonical_json(state)) <= max_characters:
                    break
                state[field] = {"truncated": True, "field": field}
                omitted.append(field)
            state["truncation"] = {
                "truncated": True,
                "max_characters": max_characters,
                "omitted": omitted,
            }
        encoded = canonical_json(state)
        try:
            assert_no_secrets(encoded, secrets)
        except ValueError:
            return _withheld_state(entry.evidence_id, analysis_id, "serialization_failure")
        sendable = len(encoded) <= max_characters
        failure = None if sendable else "oversized_state"
        return JudgmentState(
            encoded,
            stable_id(json.loads(encoded)),
            entry.evidence_id,
            analysis_id,
            bool(omitted) or not sendable,
            sendable,
            failure,
        )
    except (TypeError, KeyError) as exc:
        marker = {
            "schema_version": STATE_SCHEMA_VERSION,
            "evidence_id": getattr(entry, "evidence_id", ""),
            "analysis_id": analysis_id,
            "serialization": "serialization_failure",
            "reason": type(exc).__name__,
        }
        encoded = canonical_json(marker)
        return JudgmentState(
            encoded,
            stable_id(marker),
            marker["evidence_id"],
            analysis_id,
            False,
            False,
            "serialization_failure",
        )


def _withheld_state(evidence_id, analysis_id, failure):
    marker = {
        "schema_version": STATE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "analysis_id": analysis_id,
        "serialization": failure,
    }
    encoded = canonical_json(marker)
    return JudgmentState(
        encoded,
        stable_id(marker),
        evidence_id,
        analysis_id,
        False,
        False,
        failure,
    )


def judge_selected_evidence(
    provider, attention_items, *, ledger, analysis_id, as_of, questions=EVIDENCE_PACKET_QUESTIONS
):
    """Judge attention selections only. Evidence must already exist in the ledger."""
    overlays = []
    for item in attention_items:
        signal = item.signal if hasattr(item, "signal") else item["signal"]
        evidence_id = (
            signal.evidence_id if hasattr(signal, "evidence_id") else signal["evidence_id"]
        )
        entry = ledger.get(evidence_id)
        state = build_judgment_state(entry, analysis_id=analysis_id, as_of=as_of)
        try:
            result = provider.evaluate(state, questions)
        except Exception:
            moment = _clock_now()
            result = make_result(
                state,
                provider=getattr(provider, "name", "unknown"),
                model="",
                provider_model_version=None,
                status="service_failure",
                limitations=(_STATUS_LIMITATIONS["service_failure"],),
                requested_at=moment,
                completed_at=moment,
                latency_ms=0.0,
            )
        if result.evidence_id != entry.evidence_id or result.state_hash != state.state_id:
            moment = result.completed_at
            result = make_result(
                state,
                provider=result.provider,
                model=result.model,
                provider_model_version=result.provider_model_version,
                status="malformed_response",
                limitations=("Judgment result did not bind to the supplied evidence state.",),
                requested_at=result.requested_at,
                completed_at=moment,
                latency_ms=result.latency_ms,
            )
        overlay = assemble_overlay(result)
        ledger.append_judgment(overlay)
        overlays.append(overlay)
    ordered = tuple(sorted(overlays, key=lambda overlay: overlay.evidence_id))
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "enabled": True,
        "provider": getattr(provider, "name", "unknown"),
        "model": ordered[0].judgment_result.model if ordered else "",
        "question_set_version": QUESTION_SET_VERSION,
        "policy_version": JUDGMENT_POLICY_VERSION,
        "analysis_id": analysis_id,
        "evidence_ids": tuple(overlay.evidence_id for overlay in ordered),
        "artifact_hashes": tuple(overlay.judgment_result.judgment_id for overlay in ordered),
        "replay": _REPLAY_NOTE,
        "historical_use": POST_HOC_JUDGMENT_EVALUATION,
        "overlays": {overlay.evidence_id: canonical(overlay) for overlay in ordered},
    }


def typesafe_model_risk_extension(profile: ModelRiskProfile, *, model: str) -> ModelRiskProfile:
    """Record the optional remote dependency without assigning a risk tier or validation status."""
    from dataclasses import replace

    return replace(
        profile,
        foreseeable_misuse=profile.foreseeable_misuse
        + (
            "treating typed-judgment probabilities as statistical confidence, model validation, or a supervisory conclusion",
        ),
        limitations=profile.limitations
        + (
            "Optional typed judgment characterizes bounded questions about an evidence packet. It does not validate DRR, rate an institution, or authorize supervisory action.",
            "Remote judgment depends on the TypeSafe provider, the requested model alias, and the model version returned with the response.",
        ),
        use_boundaries=profile.use_boundaries
        + (
            "human interpretation remains authoritative over typed judgment",
            "remote judgment is unsupported for confidential supervisory information",
        ),
        monitoring_plan=profile.monitoring_plan
        + (
            "Compare recorded provider model versions when the judgment alias moves; do not treat a newer model as historical evidence.",
        ),
        change_control=profile.change_control
        + (
            "Question set evidence-packet-v1 and judgment policy v2 are explicit. Policy v2 names the adequate, no-material-Noul outcome EVIDENCE_REVIEWABLE. A provider model version change creates a distinct judgment record.",
        ),
        dependencies=profile.dependencies
        + ("typesafe-sdk optional extra; imported only when the TypeSafe provider runs",),
        third_party_products=profile.third_party_products
        + (
            f"TypeSafe Jev ({model}) optional remote typed judgment; not regulatory approval, independent validation, or a supervisory endorsement",
        ),
    )


def judgment_configuration(config):
    if config.judgment_provider == "typesafe":
        model = config.judgment_model
    elif config.judgment_provider == "mock":
        model = "deterministic-mock"
    else:
        model = "disabled"
    return {
        "enabled": True,
        "provider": config.judgment_provider,
        "model": model,
        "question_set_version": QUESTION_SET_VERSION,
        "policy_version": config.judgment_policy_version,
        "evaluated_after": "AttentionBudget.select",
        "artifact_location": (
            "append-only judgments table and result.judgment; excluded from the analytical monitoring hash"
        ),
        "replay": _REPLAY_NOTE,
    }


def replay_recorded_judgments(ledger, *, as_of=None, evidence_ids=None):
    """Return stored judgments. This function does not construct a provider or a network client."""
    return ledger.judgments(as_of=as_of, evidence_ids=evidence_ids)


def describe_post_hoc_judgment():
    return {
        "label": POST_HOC_JUDGMENT_EVALUATION,
        "historical_evidence": False,
        "note": (
            "A judgment produced after the review date is comparison material. It is not "
            "historical evidence unless a recorded judgment artifact existed at that review date."
        ),
    }


def _pairs(value):
    if value is None:
        return None
    return tuple((str(key), item) for key, item in value)


def judgment_result_from_dict(value):
    answers = tuple(
        TypedAnswer(
            answer["question_id"],
            answer["primitive"],
            answer.get("choice"),
            answer.get("score"),
            answer.get("noul"),
            _pairs(answer.get("probabilities")),
            answer.get("confidence"),
            _pairs(answer.get("legend")),
        )
        for answer in value["answers"]
    )
    return JudgmentResult(
        value["schema_version"],
        value["provider"],
        value["model"],
        value.get("provider_model_version"),
        value["question_set_version"],
        value["evidence_id"],
        value.get("analysis_id"),
        value["state_hash"],
        value["state_truncated"],
        answers,
        value["requested_at"],
        value["completed_at"],
        value["latency_ms"],
        value["provider_status"],
        tuple(value.get("limitations") or ()),
    )


def overlay_from_dict(value):
    result = (
        value["judgment_result"]
        if isinstance(value["judgment_result"], JudgmentResult)
        else judgment_result_from_dict(value["judgment_result"])
    )
    return JudgmentOverlay(
        value["evidence_id"],
        result,
        value.get("review_complexity"),
        tuple(value.get("warnings") or ()),
        value["policy_outcome"],
        value["policy_version"],
    )


def judgment_content(payload):
    """Identity comparison that ignores audit timestamps and latency."""
    data = canonical(payload)
    result = data.get("judgment_result", data)
    for key in ("requested_at", "completed_at", "latency_ms"):
        result.pop(key, None)
    return data
