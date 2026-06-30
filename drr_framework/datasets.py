"""Dataset adapters for DRR audience-specific workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyResonanceDataset:
    """A tabular policy-analysis dataset prepared for DRR.

    The adapter keeps the analyst-facing DataFrame and the numeric DRR matrix
    together so workflows can move between economic labels and numerical
    diagnostics without losing provenance.
    """

    frame: pd.DataFrame
    values: np.ndarray
    variable_names: Tuple[str, ...]
    dates: Optional[Tuple[str, ...]]
    sampling_rate: float
    metadata: Dict[str, object]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        date_column: Optional[str] = None,
        value_columns: Optional[Sequence[str]] = None,
        sampling_rate: float = 1.0,
        transform: Optional[str] = None,
        interpolate: bool = True,
        standardize: bool = True,
    ) -> "PolicyResonanceDataset":
        frame = pd.read_csv(path)
        return cls.from_frame(
            frame,
            date_column=date_column,
            value_columns=value_columns,
            sampling_rate=sampling_rate,
            transform=transform,
            interpolate=interpolate,
            standardize=standardize,
            source=str(path),
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        date_column: Optional[str] = None,
        value_columns: Optional[Sequence[str]] = None,
        sampling_rate: float = 1.0,
        transform: Optional[str] = None,
        interpolate: bool = True,
        standardize: bool = True,
        source: str = "dataframe",
    ) -> "PolicyResonanceDataset":
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if frame.empty:
            raise ValueError("frame must not be empty")

        prepared = frame.copy()
        dates = None
        if date_column is not None:
            if date_column not in prepared.columns:
                raise ValueError(f"date_column {date_column!r} not found")
            prepared[date_column] = pd.to_datetime(prepared[date_column], errors="raise")
            prepared = prepared.sort_values(date_column).reset_index(drop=True)
            dates = tuple(prepared[date_column].dt.strftime("%Y-%m-%d"))

        if value_columns is None:
            value_columns = tuple(
                column
                for column in prepared.select_dtypes(include=[np.number]).columns
                if column != date_column
            )
        else:
            value_columns = tuple(value_columns)

        if not value_columns:
            raise ValueError("at least one numeric value column is required")
        missing = [column for column in value_columns if column not in prepared.columns]
        if missing:
            raise ValueError(f"value_columns not found: {missing}")

        numeric = prepared.loc[:, value_columns].apply(pd.to_numeric, errors="coerce")
        numeric = _transform_policy_values(numeric, transform)
        rows_before_cleaning = len(numeric)

        if interpolate:
            numeric = numeric.interpolate(method="linear", limit_direction="both")
        numeric = numeric.dropna(axis=0, how="any")

        if len(numeric) < 3:
            raise ValueError("policy dataset must contain at least three complete observations")

        if dates is not None:
            dates = tuple(np.asarray(dates, dtype=object)[numeric.index.to_numpy()])

        if standardize:
            numeric = _standardize(numeric)

        prepared = prepared.loc[numeric.index].reset_index(drop=True)
        numeric = numeric.reset_index(drop=True)

        return cls(
            frame=prepared,
            values=numeric.to_numpy(dtype=float),
            variable_names=tuple(value_columns),
            dates=dates,
            sampling_rate=float(sampling_rate),
            metadata={
                "source": source,
                "date_column": date_column,
                "transform": transform or "level",
                "interpolate": interpolate,
                "standardize": standardize,
                "rows_before_cleaning": rows_before_cleaning,
                "rows_after_cleaning": len(numeric),
            },
        )

    def with_sampling_rate(self, sampling_rate: float) -> "PolicyResonanceDataset":
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        return replace(self, sampling_rate=float(sampling_rate))

    def to_drr_input(self) -> np.ndarray:
        return self.values.copy()


def load_policy_dataset(
    path: str | Path,
    *,
    date_column: Optional[str] = None,
    value_columns: Optional[Sequence[str]] = None,
    sampling_rate: float = 1.0,
    transform: Optional[str] = None,
    interpolate: bool = True,
    standardize: bool = True,
) -> PolicyResonanceDataset:
    """Load a CSV of policy observables into a DRR-ready dataset."""

    return PolicyResonanceDataset.from_csv(
        path,
        date_column=date_column,
        value_columns=value_columns,
        sampling_rate=sampling_rate,
        transform=transform,
        interpolate=interpolate,
        standardize=standardize,
    )


def _transform_policy_values(frame: pd.DataFrame, transform: Optional[str]) -> pd.DataFrame:
    if transform is None or transform == "level":
        return frame
    if transform == "diff":
        return frame.diff()
    if transform == "pct_change":
        return frame.pct_change() * 100.0
    if transform == "log_diff":
        if (frame <= 0).any().any():
            raise ValueError("log_diff requires strictly positive values")
        return np.log(frame).diff()
    raise ValueError("transform must be one of: level, diff, pct_change, log_diff")


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    means = frame.mean(axis=0)
    stds = frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (frame - means) / stds
