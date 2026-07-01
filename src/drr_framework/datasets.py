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
    def from_sql(
        cls,
        query: str,
        connection,
        *,
        date_column: Optional[str] = None,
        value_columns: Optional[Sequence[str]] = None,
        sampling_rate: float = 1.0,
        transform: Optional[str] = None,
        interpolate: bool = True,
        standardize: bool = True,
    ) -> "PolicyResonanceDataset":
        frame = pd.read_sql_query(query, connection)
        return cls.from_frame(
            frame,
            date_column=date_column,
            value_columns=value_columns,
            sampling_rate=sampling_rate,
            transform=transform,
            interpolate=interpolate,
            standardize=standardize,
            source="sql_query",
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


@dataclass(frozen=True)
class SupervisoryPanelDataset:
    """Panel data adapter for institution-level supervisory analytics.

    Input should be a wide panel with one row per institution/date and one
    column per metric, such as liquidity ratio, capital ratio, funding spread,
    asset growth, market stress proxy, or controls-risk indicator.
    """

    frame: pd.DataFrame
    institution_column: str
    date_column: str
    metric_columns: Tuple[str, ...]
    peer_group_column: Optional[str]
    sampling_rate: float
    metadata: Dict[str, object]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        institution_column: str,
        date_column: str,
        metric_columns: Sequence[str],
        peer_group_column: Optional[str] = None,
        sampling_rate: float = 1.0,
        transform: Optional[str] = None,
        interpolate: bool = True,
        standardize: bool = True,
    ) -> "SupervisoryPanelDataset":
        frame = pd.read_csv(path)
        return cls.from_frame(
            frame,
            institution_column=institution_column,
            date_column=date_column,
            metric_columns=metric_columns,
            peer_group_column=peer_group_column,
            sampling_rate=sampling_rate,
            transform=transform,
            interpolate=interpolate,
            standardize=standardize,
            source=str(path),
        )

    @classmethod
    def from_sql(
        cls,
        query: str,
        connection,
        *,
        institution_column: str,
        date_column: str,
        metric_columns: Sequence[str],
        peer_group_column: Optional[str] = None,
        sampling_rate: float = 1.0,
        transform: Optional[str] = None,
        interpolate: bool = True,
        standardize: bool = True,
    ) -> "SupervisoryPanelDataset":
        frame = pd.read_sql_query(query, connection)
        return cls.from_frame(
            frame,
            institution_column=institution_column,
            date_column=date_column,
            metric_columns=metric_columns,
            peer_group_column=peer_group_column,
            sampling_rate=sampling_rate,
            transform=transform,
            interpolate=interpolate,
            standardize=standardize,
            source="sql_query",
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        institution_column: str,
        date_column: str,
        metric_columns: Sequence[str],
        peer_group_column: Optional[str] = None,
        sampling_rate: float = 1.0,
        transform: Optional[str] = None,
        interpolate: bool = True,
        standardize: bool = True,
        source: str = "dataframe",
    ) -> "SupervisoryPanelDataset":
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if frame.empty:
            raise ValueError("frame must not be empty")

        metric_columns = tuple(metric_columns)
        if not metric_columns:
            raise ValueError("metric_columns must not be empty")

        required = [institution_column, date_column, *metric_columns]
        if peer_group_column is not None:
            required.append(peer_group_column)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"required panel columns not found: {missing}")

        prepared = frame.loc[:, required].copy()
        prepared[date_column] = pd.to_datetime(prepared[date_column], errors="raise")
        prepared = prepared.reset_index(drop=True)

        cleaned_groups = []
        for institution_id, group in prepared.groupby(institution_column, sort=False):
            policy_dataset = PolicyResonanceDataset.from_frame(
                group,
                date_column=date_column,
                value_columns=metric_columns,
                sampling_rate=sampling_rate,
                transform=transform,
                interpolate=interpolate,
                standardize=standardize,
                source=f"institution:{institution_id}",
            )
            cleaned = policy_dataset.frame.copy().reset_index(drop=True)
            cleaned.loc[:, list(metric_columns)] = policy_dataset.values
            cleaned_groups.append(cleaned)

        if not cleaned_groups:
            raise ValueError("panel dataset contains no analyzable institutions")

        normalized = pd.concat(cleaned_groups, ignore_index=True)
        return cls(
            frame=normalized,
            institution_column=institution_column,
            date_column=date_column,
            metric_columns=metric_columns,
            peer_group_column=peer_group_column,
            sampling_rate=float(sampling_rate),
            metadata={
                "source": source,
                "transform": transform or "level",
                "interpolate": interpolate,
                "standardize": standardize,
                "institution_count": int(normalized[institution_column].nunique()),
                "rows_after_cleaning": len(normalized),
            },
        )

    @property
    def institutions(self) -> Tuple[str, ...]:
        values = self.frame[self.institution_column].astype(str).unique()
        return tuple(values)

    @property
    def peer_groups(self) -> Tuple[str, ...]:
        if self.peer_group_column is None:
            return tuple()
        values = self.frame[self.peer_group_column].astype(str).unique()
        return tuple(values)

    def dataset_for_institution(self, institution_id: str) -> PolicyResonanceDataset:
        group = self.frame[self.frame[self.institution_column].astype(str) == str(institution_id)]
        if group.empty:
            raise ValueError(f"institution_id {institution_id!r} not found")
        return PolicyResonanceDataset.from_frame(
            group,
            date_column=self.date_column,
            value_columns=self.metric_columns,
            sampling_rate=self.sampling_rate,
            transform=None,
            interpolate=False,
            standardize=False,
            source=f"institution:{institution_id}",
        )

    def dataset_for_peer_group(self, peer_group: str) -> PolicyResonanceDataset:
        if self.peer_group_column is None:
            raise ValueError("peer_group_column was not configured")
        group = self.frame[self.frame[self.peer_group_column].astype(str) == str(peer_group)]
        if group.empty:
            raise ValueError(f"peer_group {peer_group!r} not found")
        aggregate = (
            group.groupby(self.date_column, as_index=False)[list(self.metric_columns)]
            .mean()
            .sort_values(self.date_column)
        )
        return PolicyResonanceDataset.from_frame(
            aggregate,
            date_column=self.date_column,
            value_columns=self.metric_columns,
            sampling_rate=self.sampling_rate,
            transform=None,
            interpolate=False,
            standardize=False,
            source=f"peer_group:{peer_group}",
        )

    def to_tableau_frame(self) -> pd.DataFrame:
        """Return a tidy long table suitable for Tableau dashboards."""

        id_vars = [self.institution_column, self.date_column]
        if self.peer_group_column is not None:
            id_vars.append(self.peer_group_column)
        return self.frame.melt(
            id_vars=id_vars,
            value_vars=list(self.metric_columns),
            var_name="metric",
            value_name="value",
        )


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


def load_policy_dataset_from_sql(
    query: str,
    connection,
    *,
    date_column: Optional[str] = None,
    value_columns: Optional[Sequence[str]] = None,
    sampling_rate: float = 1.0,
    transform: Optional[str] = None,
    interpolate: bool = True,
    standardize: bool = True,
) -> PolicyResonanceDataset:
    """Load policy observables from a SQL query into a DRR-ready dataset."""

    return PolicyResonanceDataset.from_sql(
        query,
        connection,
        date_column=date_column,
        value_columns=value_columns,
        sampling_rate=sampling_rate,
        transform=transform,
        interpolate=interpolate,
        standardize=standardize,
    )


def load_supervisory_panel(
    path: str | Path,
    *,
    institution_column: str,
    date_column: str,
    metric_columns: Sequence[str],
    peer_group_column: Optional[str] = None,
    sampling_rate: float = 1.0,
    transform: Optional[str] = None,
    interpolate: bool = True,
    standardize: bool = True,
) -> SupervisoryPanelDataset:
    """Load institution-level supervisory panel data from CSV."""

    return SupervisoryPanelDataset.from_csv(
        path,
        institution_column=institution_column,
        date_column=date_column,
        metric_columns=metric_columns,
        peer_group_column=peer_group_column,
        sampling_rate=sampling_rate,
        transform=transform,
        interpolate=interpolate,
        standardize=standardize,
    )


def load_supervisory_panel_from_sql(
    query: str,
    connection,
    *,
    institution_column: str,
    date_column: str,
    metric_columns: Sequence[str],
    peer_group_column: Optional[str] = None,
    sampling_rate: float = 1.0,
    transform: Optional[str] = None,
    interpolate: bool = True,
    standardize: bool = True,
) -> SupervisoryPanelDataset:
    """Load institution-level supervisory panel data from a SQL query."""

    return SupervisoryPanelDataset.from_sql(
        query,
        connection,
        institution_column=institution_column,
        date_column=date_column,
        metric_columns=metric_columns,
        peer_group_column=peer_group_column,
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
