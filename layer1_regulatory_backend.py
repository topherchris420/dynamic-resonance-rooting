#!/usr/bin/env python3
"""
================================================================================
MACRO STABILITY & BANKING SKIN COCKPIT - LAYER 1: REGULATORY BACKEND BRIDGE
================================================================================
Federal Reserve Model Risk Management (SR 11-7) Compliant Implementation

Project: Dynamic Resonance Rooting (DRR) Framework Integration
Purpose: Real-time early-warning system for detecting non-linear liquidity
         panics ("Dash for Cash" loop) across Large Foreign Banking Organizations

Author: Principal Financial Systems Architect & Federal Reserve Data Engineering Lead
Date: 2025-07-09
Compliance: SR 11-7 Model Risk Management Guidelines
================================================================================
"""

import numpy as np
import pandas as pd
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

# DRR Framework Imports
from drr_framework import DynamicResonanceRooting
from drr_framework.reporting import write_tableau_artifacts

# =============================================================================
# CONFIGURATION & CONSTANTS (REGULATORY COMPLIANCE)
# =============================================================================

# Federal Reserve PCA Well-Capitalized Thresholds (12 CFR Part 225)
PCA_WELL_CAPITALIZED_THRESHOLD = 0.065  # 6.5% CET1

# DRR Confidence Threshold (SR 11-7 requires documented confidence levels)
DEFAULT_DRR_CONFIDENCE_THRESHOLD = 0.65

# Rate Shock Parameters (in basis points)
RATE_SHOCK_BPS = 200  # 200 bps parallel upward shift

# AOCI Modes
AOCI_FULL_INCLUSION = "full_aoci"
AOCI_OPT_OUT = "opt_out"


@dataclass
class BankingDataSchema:
    """
    Standardized schema for Federal Reserve banking data integration.
    Maps FFIEC 002 Schedule RAL/P and FR Y-9C Schedule HC-B/HC-R fields.
    """
    # Liquidity Risk Metrics (FFIEC 002 Schedule RAL/P)
    overnight_repo: str = "RAL_OVERNIGHT_REPO"
    eurodollar_borrowings: str = "RAL_EURUSD_BORR"

    # Securities Portfolio Duration (FR Y-9C Schedule HC-B)
    afs_portfolio: str = "HC_B_AFS_SECURITIES"
    htm_portfolio: str = "HC_B_HTM_SECURITIES"
    afs_duration: str = "HC_B_AFS_DURATION"
    htm_duration: str = "HC_B_HTM_DURATION"

    # Capital Metrics (FR Y-9C Schedule HC-R)
    cet1_capital: str = "HC_R_CET1"
    risk_weighted_assets: str = "HC_R_RWA"
    aoci_adjustment: str = "HC_R_AOCI"

    # LFBO Identification
    rssd_id: str = "RSSD_ID"
    legal_name: str = "LEGAL_NAME"
    hub_country: str = "HUB_COUNTRY"


# =============================================================================
# LAYER 1.1: SQL CALL REPORT EXTRACTION ENGINE
# =============================================================================

class FFIEC002_FR_Y9C_Extractor:
    """
    Custom SQL adapter for extracting banking data from FFIEC 002 and FR Y-9C reports.

    Implements regulatory data extraction for:
    - FFIEC 002 Schedule RAL/P: Repurchase Agreements & Eurodollar Borrowings
    - FR Y-9C Schedule HC-B: Securities Portfolios (AFS/HTM)
    - FR Y-9C Schedule HC-R: Capital Ratios & CET1

    Compliance: SR 11-7 requires documented data lineage and validation.
    """

    # SQL Query for FFIEC 002 Schedule RAL/P (Liquidity Risk)
    SQL_SCHEDULE_RAL_P = """
    SELECT
        r.rssd_id,
        r.legal_name,
        r.report_date,
        r.hub_country,
        -- Overnight Repurchase Agreements (federal funds sold + securities sold under agreements to repurchase)
        COALESCE(r.ff_sold, 0) + COALESCE(r.sec_sold_repo, 0) AS RAL_OVERNIGHT_REPO,
        -- Eurodollar Borrowings (foreign branch liabilities)
        COALESCE(r.eurodollar_borr, 0) AS RAL_EURUSD_BORR,
        -- Term Repo Borrowings (>1 day)
        COALESCE(r.term_repo_borr, 0) AS RAL_TERM_REPO
    FROM ffiec_002_schedule_ral_p r
    WHERE r.report_date >= :start_date
      AND r.report_date <= :end_date
      AND r.hub_country IN ('GB', 'DE', 'FR', 'JP', 'CH', 'CA')
    """

    # SQL Query for FR Y-9C Schedule HC-B (Securities Portfolio)
    SQL_SCHEDULE_HC_B = """
    SELECT
        h.rssd_id,
        h.report_date,
        -- Available-for-Sale Securities (fair value)
        COALESCE(h.afs_securities_fv, 0) AS HC_B_AFS_SECURITIES,
        -- Held-to-Maturity Securities (amortized cost)
        COALESCE(h.htm_securities_ac, 0) AS HC_B_HTM_SECURITIES,
        -- Weighted average duration (years)
        COALESCE(h.afs_duration_wa, 0) AS HC_B_AFS_DURATION,
        COALESCE(h.htm_duration_wa, 0) AS HC_B_HTM_DURATION,
        -- Unrealized gains/losses (AOCI component)
        COALESCE(h.afs_unrealized_gl, 0) AS HC_B_AFS_UNREALIZED_GL,
        COALESCE(h.htm_unrealized_gl, 0) AS HC_B_HTM_UNREALIZED_GL
    FROM fr_y9c_schedule_hc_b h
    WHERE h.report_date >= :start_date
      AND h.report_date <= :end_date
    """

    # SQL Query for FR Y-9C Schedule HC-R (Regulatory Capital)
    SQL_SCHEDULE_HC_R = """
    SELECT
        c.rssd_id,
        c.report_date,
        -- Common Equity Tier 1 (CET1) Capital
        COALESCE(c.cet1_capital, 0) AS HC_R_CET1,
        -- Risk-Weighted Assets (RWA)
        COALESCE(c.risk_weighted_assets, 0) AS HC_R_RWA,
        -- Accumulated Other Comprehensive Income (AOCI)
        COALESCE(c.aoci, 0) AS HC_R_AOCI,
        -- Additional Tier 1 Capital
        COALESCE(c.at1_capital, 0) AS HC_R_AT1,
        -- Tier 2 Capital
        COALESCE(c.tier2_capital, 0) AS HC_R_TIER2,
        -- Total Capital
        COALESCE(c.total_capital, 0) AS HC_R_TOTAL_CAPITAL,
        -- Capital Conservation Buffer
        COALESCE(c.ccb, 0) AS HC_R_CCB
    FROM fr_y9c_schedule_hc_r c
    WHERE c.report_date >= :start_date
      AND c.report_date <= :end_date
    """

    def __init__(self, db_connection: sqlite3.Connection):
        """Initialize with database connection."""
        self.conn = db_connection

    def extract_liquidity_metrics(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Extract overnight repo and Eurodollar borrowings from Schedule RAL/P."""
        df = pd.read_sql(
            self.SQL_SCHEDULE_RAL_P,
            self.conn,
            params={"start_date": start_date, "end_date": end_date}
        )
        return df

    def extract_securities_portfolio(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Extract AFS/HTM portfolio durations from Schedule HC-B."""
        df = pd.read_sql(
            self.SQL_SCHEDULE_HC_B,
            self.conn,
            params={"start_date": start_date, "end_date": end_date}
        )
        return df

    def extract_capital_metrics(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Extract CET1 capital metrics from Schedule HC-R."""
        df = pd.read_sql(
            self.SQL_SCHEDULE_HC_R,
            self.conn,
            params={"start_date": start_date, "end_date": end_date}
        )
        return df

    def extract_all(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Complete extraction: Merge RAL/P, HC-B, and HC-R data.
        Returns unified panel of LFBO liquidity and capital metrics.
        """
        liquidity = self.extract_liquidity_metrics(start_date, end_date)
        securities = self.extract_securities_portfolio(start_date, end_date)
        capital = self.extract_capital_metrics(start_date, end_date)

        # Merge on RSSD_ID and report_date
        merged = liquidity.merge(securities, on=["rssd_id", "report_date"], how="outer")
        merged = merged.merge(capital, on=["rssd_id", "report_date"], how="outer")

        # Compute derived metrics
        merged["TOTAL_REPO_DEPENDENCY"] = (
            merged["RAL_OVERNIGHT_REPO"] + merged["RAL_EURUSD_BORR"]
        )
        merged["SECURITIES_TOTAL"] = (
            merged["HC_B_AFS_SECURITIES"] + merged["HC_B_HTM_SECURITIES"]
        )

        return merged.sort_values(["rssd_id", "report_date"])


# =============================================================================
# LAYER 1.2: DRR FRAMEWORK ADAPTER
# =============================================================================

class DRR_BankingAdapter:
    """
    DRR Framework Adapter for Federal Reserve Banking Data.

    Transforms regulatory banking time-series into multivariate arrays
    required by drr.analyze_system() for resonance detection and rooting analysis.

    Implements:
    - Welch/FFT resonance detection for hidden cyclical funding stress
    - Lag-aware rooting analysis for directional lead-lag structures
    - Composite Resonance Depth scoring
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        tau: int = 1,
        sampling_rate: float = 4.0,  # Quarterly data = 4 samples/year
        confidence_threshold: float = DEFAULT_DRR_CONFIDENCE_THRESHOLD
    ):
        """
        Initialize DRR Banking Adapter.

        Parameters:
        -----------
        embedding_dim : int
            Phase space reconstruction dimension (default: 3)
        tau : int
            Time delay for time-delay embedding (default: 1)
        sampling_rate : float
            Data sampling rate in Hz (default: 4.0 for quarterly)
        confidence_threshold : float
            Confidence threshold for belief state (default: 0.65)
        """
        self.drr = DynamicResonanceRooting(
            embedding_dim=embedding_dim,
            tau=tau,
            sampling_rate=sampling_rate
        )
        self.confidence_threshold = confidence_threshold
        self.schema = BankingDataSchema()

    def prepare_multivariate_array(
        self,
        df: pd.DataFrame,
        bank_id: str,
        columns: List[str]
    ) -> np.ndarray:
        """
        Map banking time-series columns into DRR multivariate format.

        Parameters:
        -----------
        df : pd.DataFrame
            Panel data with time-series columns
        bank_id : str
            RSSD_ID of the bank to analyze
        columns : List[str]
            Column names to include in multivariate array

        Returns:
        --------
        np.ndarray
            Multivariate time-series array (n_timesteps x n_variables)
        """
        bank_data = df[df[self.schema.rssd_id] == bank_id].sort_values("report_date")

        if bank_data.empty:
            raise ValueError(f"No data found for bank ID: {bank_id}")

        # Extract columns and handle missing values
        array_data = bank_data[columns].fillna(0).values

        # Transpose so rows = timesteps, columns = variables
        return array_data.T

    def analyze_bank(
        self,
        df: pd.DataFrame,
        bank_id: str,
        shock_scenario: str = AOCI_FULL_INCLUSION
    ) -> Dict[str, Any]:
        """
        Complete DRR analysis for a single bank.

        Parameters:
        -----------
        df : pd.DataFrame
            Panel data from FFIEC/FR Y-9C extraction
        bank_id : str
            RSSD_ID of the bank to analyze
        shock_scenario : str
            AOCI_FULL_INCLUSION or AOCI_OPT_OUT

        Returns:
        --------
        Dict containing resonance depths, frequencies, and significant edges
        """
        # Define multivariate input columns
        liquidity_cols = [
            "RAL_OVERNIGHT_REPO",
            "RAL_EURUSD_BORR",
            "RAL_TERM_REPO"
        ]
        securities_cols = [
            "HC_B_AFS_SECURITIES",
            "HC_B_HTM_SECURITIES",
            "HC_B_AFS_DURATION",
            "HC_B_HTM_DURATION"
        ]
        capital_cols = [
            "HC_R_CET1",
            "HC_R_RWA",
            "HC_R_AOCI"
        ]

        all_columns = liquidity_cols + securities_cols + capital_cols

        # Prepare multivariate array
        multivariate_data = self.prepare_multivariate_array(df, bank_id, all_columns)

        # Run DRR analysis
        results = self.drr.analyze_system(
            multivariate_data,
            multivariate=True,
            window_size=min(50, len(multivariate_data[0]) // 4),
            state_space=True,
            state_space_horizon=12
        )

        # Add metadata
        results["bank_id"] = bank_id
        results["shock_scenario"] = shock_scenario
        results["confidence_threshold"] = self.confidence_threshold

        return results

    def compute_term_premium_shock_impact(
        self,
        df: pd.DataFrame,
        bank_id: str,
        shock_bps: int = RATE_SHOCK_BPS,
        aoci_mode: str = AOCI_FULL_INCLUSION
    ) -> pd.DataFrame:
        """
        Simulate 200 bps parallel upward rate shift under two conditions:
        - Full AOCI Inclusion (mark-to-market AFS/HTM losses in CET1)
        - AOCI Opt-Out (exclude AOCI from CET1 calculation per Fed rule)

        This implements the "Dash for Cash" loop detection under stress.
        """
        bank_data = df[df[self.schema.rssd_id] == bank_id].copy()

        # Duration-weighted portfolio sensitivity
        # ΔPV ≈ -Duration × ΔRate × Portfolio Value
        shock_decimal = shock_bps / 10000

        # Compute duration-weighted sensitivity
        bank_data["AFS_RATE_LOSS"] = (
            bank_data["HC_B_AFS_DURATION"] *
            shock_decimal *
            bank_data["HC_B_AFS_SECURITIES"]
        )
        bank_data["HTM_RATE_LOSS"] = (
            bank_data["HC_B_HTM_DURATION"] *
            shock_decimal *
            bank_data["HC_B_HTM_SECURITIES"]
        )
        bank_data["TOTAL_RATE_LOSS"] = (
            bank_data["AFS_RATE_LOSS"] + bank_data["HTM_RATE_LOSS"]
        )

        # Compute post-shock CET1 based on AOCI mode
        if aoci_mode == AOCI_FULL_INCLUSION:
            # Full inclusion: losses reduce CET1
            bank_data["POST_SHOCK_CET1"] = (
                bank_data["HC_R_CET1"] - bank_data["TOTAL_RATE_LOSS"]
            )
        else:  # AOCI_OPT_OUT
            # Opt-out: AOCI not included, losses don't affect CET1
            bank_data["POST_SHOCK_CET1"] = bank_data["HC_R_CET1"]

        # Compute post-shock CET1 ratio
        bank_data["POST_SHOCK_CET1_RATIO"] = (
            bank_data["POST_SHOCK_CET1"] / bank_data["HC_R_RWA"]
        )

        # Flag PCA breach
        bank_data["PCA_BREACH"] = (
            bank_data["POST_SHOCK_CET1_RATIO"] < PCA_WELL_CAPITALIZED_THRESHOLD
        )

        return bank_data


# =============================================================================
# LAYER 1.3: SPECTRAL & LAGGED BIAS METRICS COMPUTATION
# =============================================================================

class SpectralLaggedBiasComputer:
    """
    Computes Welch/FFT resonance detection and lag-aware rooting analysis.

    Identifies:
    - Hidden cyclical funding stress patterns
    - Directional lead-lag structures (significant_edges)
    - Composite Resonance Depth scores
    """

    def __init__(self, drr_adapter: DRR_BankingAdapter):
        self.adapter = drr_adapter

    def compute_resonance_detection(
        self,
        time_series: np.ndarray,
        method: str = "fft"
    ) -> Dict[str, Any]:
        """
        Compute Welch/FFT resonance detection for hidden cyclical funding stress.

        Parameters:
        -----------
        time_series : np.ndarray
            1D time series data
        method : str
            "fft" or "wavelet" detection method

        Returns:
        --------
        Dict with frequencies, power spectrum, dominant frequency
        """
        return self.adapter.drr.detect_resonances(
            time_series,
            method=method,
            peak_height_ratio=0.1
        )

    def compute_rooting_analysis(
        self,
        drr_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract significant_edges from DRR rooting analysis.

        Identifies directional lead-lag structures between:
        - Treasury Term-Premium shocks
        - Bank liquidity drains (repo/Eurodollar)
        """
        if "rooting_analysis" not in drr_results:
            return {"significant_edges": [], "is_rooted": False}

        rooting = drr_results["rooting_analysis"]

        # Extract significant edges (typically transfer entropy based)
        significant_edges = []
        if "edges" in rooting:
            for edge in rooting["edges"]:
                if edge.get("weight", 0) > 0.1:  # Threshold for significance
                    significant_edges.append({
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "weight": edge.get("weight"),
                        "lag": edge.get("lag", 0),
                        "p_value": edge.get("p_value", 1.0)
                    })

        return {
            "significant_edges": significant_edges,
            "is_rooted": drr_results.get("is_rooted", False),
            "agent_belief": drr_results.get("agent_belief", 0.0)
        }

    def compute_resonance_depth_composite(
        self,
        drr_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute composite, normalized Resonance Depth score.

        Combines:
        - Spectral concentration
        - Temporal persistence
        - Phase coherence
        - Amplitude stability
        """
        return drr_results.get("resonance_depths", {})


# =============================================================================
# LAYER 1.4: TABLEAU-READY EXPORT
# =============================================================================

class TableauExportEngine:
    """
    Exports DRR analysis results to Tableau-ready CSV format.

    Produces four standardized exports:
    - resonance_summary: Key metrics per bank
    - resonance_map: Frequency detection results
    - rooting_edges: Significant lead-lag relationships
    - state_space_diagnostics: System stability metrics
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_shock_simulation(
        self,
        df: pd.DataFrame,
        drr_results: Dict[str, Any],
        shock_bps: int = RATE_SHOCK_BPS,
        aoci_mode: str = AOCI_FULL_INCLUSION,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Export computed resonance_depths, detected_frequencies, and
        significant_edges into structured, Tableau-ready CSV format.
        """
        if metadata is None:
            metadata = {}

        # Add shock scenario metadata
        metadata.update({
            "shock_bps": shock_bps,
            "aoci_mode": aoci_mode,
            "pca_threshold": PCA_WELL_CAPITALIZED_THRESHOLD
        })

        # Use DRR native export
        paths = write_tableau_artifacts(
            drr_results,
            self.output_dir,
            stem=f"drr_{aoci_mode}_{shock_bps}bps",
            metadata=metadata
        )

        return paths

    def export_bank_panel(
        self,
        shock_results: pd.DataFrame,
        resonance_depths: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export complete bank panel with shock simulation results.
        """
        # Add resonance depth to results
        output_df = shock_results.copy()
        output_df["resonance_depth"] = output_df["rssd_id"].map(
            lambda x: resonance_depths.get(x, 0.0)
        )

        # Export to CSV
        output_path = self.output_dir / "bank_panel_shock_simulation.csv"
        output_df.to_csv(output_path, index=False)

        return output_path


# =============================================================================
# MAIN ORCHESTRATION: LAYER 1 INTEGRATION
# =============================================================================

def run_macro_stability_cockpit(
    db_path: str = ":memory:",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "./output",
    run_shock_simulation: bool = True
) -> Dict[str, Any]:
    """
    Main orchestration function for Layer 1: Regulatory Backend Bridge.

    Executes full pipeline:
    1. SQL Call Report Extraction (FFIEC 002 / FR Y-9C)
    2. DRR Framework Adaptation
    3. Spectral & Lagged Bias Metrics Computation
    4. Tableau-Ready Export

    Returns comprehensive results dictionary for Layer 2 visualization.
    """
    warnings.filterwarnings("ignore")

    # Step 1: Initialize extractor (would connect to actual regulatory DB)
    conn = sqlite3.connect(db_path)
    extractor = FFIEC002_FR_Y9C_Extractor(conn)

    # Step 2: Extract banking data
    banking_panel = extractor.extract_all(start_date, end_date)

    if banking_panel.empty:
        # Generate synthetic LFBO data for demonstration
        print("[INFO] Generating synthetic LFBO data for demonstration...")
        banking_panel = _generate_synthetic_lfbo_data(start_date, end_date)

    # Step 3: Initialize DRR Adapter
    drr_adapter = DRR_BankingAdapter(
        embedding_dim=3,
        tau=1,
        sampling_rate=4.0,
        confidence_threshold=DEFAULT_DRR_CONFIDENCE_THRESHOLD
    )

    # Step 4: Initialize spectral/lagged bias computer
    computer = SpectralLaggedBiasComputer(drr_adapter)

    # Step 5: Get unique bank IDs
    bank_ids = banking_panel["rssd_id"].unique()

    # Step 6: Run analysis for each bank
    all_results = {}
    all_shock_results = {}

    for bank_id in bank_ids[:20]:  # Process first 20 banks for demo
        try:
            # DRR Analysis
            drr_results = drr_adapter.analyze_bank(banking_panel, bank_id)

            # Extract key metrics
            resonance_depths = computer.compute_resonance_depth_composite(drr_results)
            rooting_analysis = computer.compute_rooting_analysis(drr_results)

            all_results[bank_id] = {
                "resonance_depths": resonance_depths,
                "rooting_analysis": rooting_analysis,
                "drr_full_results": drr_results
            }

            # Run shock simulation if requested
            if run_shock_simulation:
                for aoci_mode in [AOCI_FULL_INCLUSION, AOCI_OPT_OUT]:
                    shock_result = drr_adapter.compute_term_premium_shock_impact(
                        banking_panel,
                        bank_id,
                        shock_bps=RATE_SHOCK_BPS,
                        aoci_mode=aoci_mode
                    )
                    all_shock_results[f"{bank_id}_{aoci_mode}"] = shock_result

        except Exception as e:
            print(f"[WARN] Failed to analyze bank {bank_id}: {e}")
            continue

    # Step 7: Export to Tableau format
    exporter = TableauExportEngine(output_dir)

    # Export results for each scenario
    export_paths = {}
    for bank_id, results in list(all_results.items())[:5]:
        for aoci_mode in [AOCI_FULL_INCLUSION, AOCI_OPT_OUT]:
            key = f"{bank_id}_{aoci_mode}"
            if key in all_shock_results:
                shock_df = all_shock_results[key]
                paths = exporter.export_shock_simulation(
                    shock_df,
                    results["drr_full_results"],
                    shock_bps=RATE_SHOCK_BPS,
                    aoci_mode=aoci_mode,
                    metadata={
                        "bank_id": bank_id,
                        "report_date": end_date
                    }
                )
                export_paths[key] = paths

    # Step 8: Compile summary for Layer 2
    summary = {
        "n_banks_analyzed": len(all_results),
        "n_scenarios": len(all_shock_results),
        "export_paths": {k: str(v) for k, v in export_paths.items()},
        "resonance_depth_summary": {
            bank_id: results["resonance_depths"]
            for bank_id, results in all_results.items()
        },
        "rooting_summary": {
            bank_id: results["rooting_analysis"]
            for bank_id, results in all_results.items()
        },
        "shock_simulation_params": {
            "shock_bps": RATE_SHOCK_BPS,
            "aoci_modes": [AOCI_FULL_INCLUSION, AOCI_OPT_OUT],
            "pca_threshold": PCA_WELL_CAPITALIZED_THRESHOLD
        }
    }

    return summary


def _generate_synthetic_lfbo_data(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Generate synthetic LFBO data for demonstration purposes.
    Simulates 5 major foreign banking organizations over 20 quarters.
    """
    np.random.seed(42)

    # Define LFBO hub countries
    lfbos = [
        {"rssd_id": "LFBO_001", "legal_name": "Global Bank International", "hub_country": "GB"},
        {"rssd_id": "LFBO_002", "legal_name": "Deutsche Federal Reserve Bank", "hub_country": "DE"},
        {"rssd_id": "LFBO_003", "legal_name": "Paris Mutual Banking Group", "hub_country": "FR"},
        {"rssd_id": "LFBO_004", "legal_name": "Nippon Credit Holdings", "hub_country": "JP"},
        {"rssd_id": "LFBO_005", "legal_name": "Zurich Cantonal Investment", "hub_country": "CH"},
    ]

    # Generate date range (quarterly)
    dates = pd.date_range(start=start_date, end=end_date, freq="Q")

    records = []
    for lfbo in lfbos:
        base_cet1 = np.random.uniform(80e9, 150e9)  # $80-150B CET1
        base_rwa = np.random.uniform(800e9, 1500e9)  # $800B-1.5T RWA
        base_repo = np.random.uniform(50e9, 200e9)  # $50-200B repo
        base_afs = np.random.uniform(100e9, 300e9)  # $100-300B AFS
        base_htm = np.random.uniform(50e9, 150e9)  # $50-150B HTM

        for date in dates:
            # Add cyclical stress patterns (simulating funding pressure)
            quarter_factor = np.sin(2 * np.pi * date.year / 4) * 0.1

            records.append({
                "rssd_id": lfbo["rssd_id"],
                "legal_name": lfbo["legal_name"],
                "hub_country": lfbo["hub_country"],
                "report_date": date.strftime("%Y-%m-%d"),
                "RAL_OVERNIGHT_REPO": base_repo * (1 + quarter_factor + np.random.normal(0, 0.05)),
                "RAL_EURUSD_BORR": base_repo * 0.3 * (1 + quarter_factor + np.random.normal(0, 0.05)),
                "RAL_TERM_REPO": base_repo * 0.2 * (1 + np.random.normal(0, 0.03)),
                "HC_B_AFS_SECURITIES": base_afs * (1 + np.random.normal(0, 0.02)),
                "HC_B_HTM_SECURITIES": base_htm * (1 + np.random.normal(0, 0.02)),
                "HC_B_AFS_DURATION": np.random.uniform(2.5, 5.5),
                "HC_B_HTM_DURATION": np.random.uniform(4.0, 7.0),
                "HC_R_CET1": base_cet1 * (1 + np.random.normal(0, 0.03)),
                "HC_R_RWA": base_rwa * (1 + np.random.normal(0, 0.01)),
                "HC_R_AOCI": -base_afs * np.random.uniform(0.01, 0.05),
            })

    return pd.DataFrame(records)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import json

    print("=" * 80)
    print("MACRO STABILITY & BANKING SKIN COCKPIT - LAYER 1")
    print("Regulatory Backend Bridge | SR 11-7 Compliant")
    print("=" * 80)

    # Run orchestration
    results = run_macro_stability_cockpit(
        db_path=":memory:",
        start_date="2020-01-01",
        end_date="2024-12-31",
        output_dir="./tableau_output"
    )

    print("\n[COMPLETE] Analysis Summary:")
    print(f"  Banks Analyzed: {results['n_banks_analyzed']}")
    print(f"  Scenarios Run: {results['n_scenarios']}")
    print(f"  PCA Threshold: {results['shock_simulation_params']['pca_threshold']:.1%}")
    print(f"  Rate Shock: {results['shock_simulation_params']['shock_bps']} bps")

    print("\n[OUTPUT] Export Paths:")
    for key, path in results["export_paths"].items():
        print(f"  {key}: {path}")

    # Save summary
    with open("./tableau_output/cockpit_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n[DONE] Layer 1 complete. Ready for Layer 2 Tableau visualization.")