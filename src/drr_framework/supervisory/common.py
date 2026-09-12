"""Canonical serialization and shared research boundaries."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import numpy as np

DECISION_BOUNDARY = (
    "Analytical research for human review. This evidence does not establish causation, "
    "distress, deficient governance, or a supervisory rating, finding, MRA, MRIA, "
    "enforcement recommendation, legal interpretation, or policy decision."
)


def instant(value: Any) -> datetime:
    """Normalize to UTC; date-only inputs mean midnight, never end-of-day."""
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, datetime.min.time())
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (
        result.replace(tzinfo=timezone.utc)
        if result.tzinfo is None
        else result.astimezone(timezone.utc)
    )


def day(value: Any) -> str:
    return instant(value).date().isoformat()


def canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: canonical(getattr(value, f.name))
            for f in fields(value)
            if not f.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(k): canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def stable_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_url(value: str, *, authoritative: bool = False) -> str:
    url = urlsplit(value)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("Public source must be an HTTPS URL without credentials")
    if authoritative and url.hostname not in {
        "www.federalreserve.gov",
        "www.ffiec.gov",
        "cdr.ffiec.gov",
    }:
        raise ValueError(
            "Verified regulatory semantics require an authoritative Federal Reserve/FFIEC source"
        )
    return value


def write_immutable(path: Path, text: str) -> Path:
    """Idempotent creation; refuse to replace an existing, different artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Immutable artifact already exists: {path.name}") from None
    return path
