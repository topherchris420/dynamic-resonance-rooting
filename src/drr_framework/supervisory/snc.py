"""Public, aggregate Shared National Credit series only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

from .common import instant, source_url


@dataclass(frozen=True)
class PublicSNCAggregate:
    year: int
    commitments: float
    non_pass: Optional[float]
    unit: str
    source: str
    publication_date: str
    available_as_of: str
    leveraged_lending: Optional[float] = None
    industry_composition: Tuple[Tuple[str, float], ...] = ()
    lender_composition: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self):
        source_url(self.source)
        if instant(self.available_as_of) < instant(self.publication_date):
            raise ValueError("SNC data cannot precede publication")
        if not math.isfinite(self.commitments) or self.commitments < 0:
            raise ValueError("Invalid public SNC commitments")
        for value in (self.non_pass, self.leveraged_lending):
            if value is not None and (
                not math.isfinite(value) or not 0 <= value <= self.commitments
            ):
                raise ValueError("SNC subsets must be within total commitments in the same units")
        for name in ("industry_composition", "lender_composition"):
            rows = tuple(tuple(row) for row in getattr(self, name))
            if (
                any(not math.isfinite(v) or v < 0 for _, v in rows)
                or sum(v for _, v in rows) > self.commitments
            ):
                raise ValueError("Invalid SNC composition totals")
            object.__setattr__(self, name, rows)


def analyze_public_snc(records, *, as_of):
    rows = sorted(
        (r for r in records if instant(r.available_as_of) <= instant(as_of)), key=lambda r: r.year
    )
    if len({r.year for r in rows}) != len(rows):
        raise ValueError("Resolve duplicate SNC report vintages explicitly")
    result = []
    for i, row in enumerate(rows):
        old = rows[i - 1] if i else None
        comparable = old and row.year - old.year == 1 and row.unit == old.unit
        result.append(
            dict(
                year=row.year,
                commitments=row.commitments,
                non_pass=row.non_pass,
                unit=row.unit,
                non_pass_share=(
                    row.non_pass / row.commitments
                    if row.non_pass is not None and row.commitments
                    else None
                ),
                annual_commitment_change=row.commitments - old.commitments if comparable else None,
                leveraged_lending=row.leveraged_lending,
                industry_composition=row.industry_composition,
                lender_composition=row.lender_composition,
                source=row.source,
                publication_date=row.publication_date,
            )
        )
    return {
        "rows": result,
        "scope": "public aggregate SNC analytics",
        "boundary": "No confidential credit-level or borrower-level SNC information is represented.",
    }
