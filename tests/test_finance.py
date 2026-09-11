"""
Comprehensive Unit Tests for DRR Quant Lab (`drr_framework.finance`).
"""

import os
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from drr_framework.finance import (
    DEFAULT_OBSERVATION_UNIVERSE,
    DEFAULT_PORTFOLIO_UNIVERSE,
    MarketData,
    MarketResonanceState,
    PortfolioRegimePolicy,
    QuantMacroConfig,
    WalkForwardBacktester,
    analyze_market_regime,
    annual_to_daily_rf,
    calculate_performance_metrics,
    calculate_returns,
    generate_synthetic_market_data,
    optimize_portfolio,
    validate_drr_predictive_signal,
    validate_market_data,
)


class TestFinanceFeatures(unittest.TestCase):
    """Test market data loading, validation, synthetic generation, and return calculation."""

    def test_synthetic_data_generation(self):
        mdata = generate_synthetic_market_data(
            symbols=["SPY", "TLT", "GLD", "HYG", "VIXY"],
            start_date="2020-01-01",
            end_date="2021-12-31",
            seed=42,
        )
        self.assertIsInstance(mdata, MarketData)
        self.assertEqual(len(mdata.prices), len(mdata.returns) + 1)
        self.assertListEqual(mdata.observation_symbols, ["SPY", "TLT", "GLD", "HYG", "VIXY"])
        self.assertListEqual(mdata.portfolio_symbols, ["SPY", "TLT", "GLD", "HYG"])

    def test_calculate_returns_explicit(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"SPY": [100.0, 102.0, 101.0, 103.0, 105.0]}, index=dates)
        rets = calculate_returns(prices)
        self.assertEqual(len(rets), 4)
        expected_first = (102.0 - 100.0) / 100.0
        self.assertAlmostEqual(rets["SPY"].iloc[0], expected_first)

    def test_data_validation_failures(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"SPY": np.ones(10)}, index=dates)

        # Missing symbol
        with self.assertRaises(ValueError):
            validate_market_data(prices, ["SPY", "TLT"])

        # Insufficient history
        with self.assertRaises(ValueError):
            validate_market_data(prices, ["SPY"], min_history_length=20)

        # NaNs
        prices_nan = prices.copy()
        prices_nan.iloc[2, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_market_data(prices_nan, ["SPY"], min_history_length=5)


class TestFinanceRegimes(unittest.TestCase):
    """Test DRR market regime adapter and no-lookahead regime policy thresholding."""

    def setUp(self):
        self.mdata = generate_synthetic_market_data(
            symbols=["SPY", "TLT", "GLD"],
            start_date="2020-01-01",
            end_date="2020-08-01",
            seed=123,
        )

    def test_analyze_market_regime(self):
        window = self.mdata.returns.iloc[:100]
        state = analyze_market_regime(
            returns_window=window,
            spectral_method="welch",
            rooting_method="lagged_correlation",
            rooting_n_surrogates=20,
            rooting_random_state=42,
        )
        self.assertIsInstance(state, MarketResonanceState)
        self.assertGreaterEqual(state.mean_depth, 0.0)
        self.assertGreaterEqual(state.max_depth, state.mean_depth)
        self.assertGreaterEqual(state.depth_dispersion, 0.0)
        self.assertGreaterEqual(state.network_density, 0.0)

    def test_policy_no_lookahead_leakage(self):
        policy = PortfolioRegimePolicy(
            threshold_type="expanding_percentile",
            fixed_threshold=0.50,
            percentile=80.0,
        )

        dummy_states = [
            MarketResonanceState(
                timestamp=None,
                mean_depth=0.1 * i,
                max_depth=0.1 * i,
                min_depth=0.01,
                depth_dispersion=0.01,
                spectral_concentration=0.5,
                temporal_persistence=0.5,
                phase_coherence=0.5,
                amplitude_stability=0.5,
                network_density=0.1,
                significant_edge_count=1,
                effective_rooting_method="lagged_correlation",
                agent_belief=0.5,
                is_rooted=False,
            )
            for i in range(1, 20)
        ]

        decisions = []
        for state in dummy_states:
            mode = policy.choose_policy(state)
            decisions.append(mode)

        # Verify initial warm-up and expanding percentile decisions
        self.assertEqual(len(decisions), 19)
        self.assertIn("standard", decisions)
        self.assertIn("high_resonance", decisions)


class TestPortfolioOptimization(unittest.TestCase):
    """Test annual to daily risk-free rate conversion and portfolio optimization."""

    def test_annual_to_daily_rf(self):
        annual_rf = 0.04  # 4%
        daily_rf = annual_to_daily_rf(annual_rf, trading_days=252)
        compounded = (1.0 + daily_rf) ** 252 - 1.0
        self.assertAlmostEqual(annual_rf, compounded, places=6)

    def test_optimize_portfolio_bounds_and_sum(self):
        mdata = generate_synthetic_market_data(
            symbols=["SPY", "TLT", "GLD", "HYG"],
            start_date="2020-01-01",
            end_date="2020-06-01",
            seed=42,
        )
        weights_std = optimize_portfolio(
            returns=mdata.returns,
            policy_mode="standard",
            annual_rf=0.04,
            min_weight=0.0,
            max_weight=0.50,
        )
        self.assertAlmostEqual(weights_std.sum(), 1.0, places=5)
        self.assertTrue((weights_std >= -1e-5).all())
        self.assertTrue((weights_std <= 0.50 + 1e-5).all())

        weights_cvar = optimize_portfolio(
            returns=mdata.returns,
            policy_mode="high_resonance",
            annual_rf=0.04,
            min_weight=0.0,
            max_weight=0.50,
        )
        self.assertAlmostEqual(weights_cvar.sum(), 1.0, places=5)


class TestMetricsAndBacktest(unittest.TestCase):
    """Test financial performance metrics, predictive signal analysis, and walk-forward backtester."""

    def setUp(self):
        self.mdata = generate_synthetic_market_data(
            symbols=["SPY", "TLT", "GLD", "HYG", "VIXY"],
            start_date="2018-01-01",
            end_date="2020-12-31",
            seed=42,
            portfolio_symbols=["SPY", "TLT", "GLD", "HYG"],
        )

    def test_performance_metrics_calculation(self):
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        turnover = pd.Series([0.1, 0.0, 0.0, 0.2] * 63)
        metrics = calculate_performance_metrics(
            returns=returns,
            annual_rf=0.04,
            trading_days=252,
            turnover=turnover,
        )
        self.assertIn("cagr", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("historical_cvar_95", metrics)

    def test_validate_drr_predictive_signal_constant_inputs(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        # Constant feature
        drr_states = pd.DataFrame(
            {
                "mean_depth": np.ones(50),
                "network_density": np.ones(50),
                "depth_dispersion": np.ones(50),
            },
            index=dates,
        )
        # Non-constant market returns
        market_returns = pd.DataFrame(
            {"SPY": np.random.normal(0.001, 0.01, 50)},
            index=dates,
        )

        results = validate_drr_predictive_signal(drr_states, market_returns, horizons=(5,))
        self.assertIn("horizon_5d", results)
        res_5d = results["horizon_5d"]["mean_depth"]
        self.assertEqual(res_5d["fwd_vol_pearson_corr"], 0.0)
        self.assertEqual(res_5d["fwd_vol_pearson_pvalue"], 1.0)
        self.assertEqual(res_5d["fwd_drawdown_pearson_corr"], 0.0)
        self.assertEqual(res_5d["fwd_drawdown_pvalue"], 1.0)

    def test_validate_drr_predictive_signal_small_scale_inputs(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        # Extremely small scale (1e-15 magnitude) zero-mean varying inputs
        feature = np.linspace(-1e-15, 1e-15, 50)
        drr_states = pd.DataFrame(
            {
                "mean_depth": feature,
                "network_density": feature,
                "depth_dispersion": feature,
            },
            index=dates,
        )
        # Varying market returns (quadratic to ensure non-constant rolling std)
        market_returns = pd.DataFrame(
            {"SPY": np.linspace(0.01, 0.5, 50) ** 2},
            index=dates,
        )

        results = validate_drr_predictive_signal(drr_states, market_returns, horizons=(5,))
        self.assertIn("horizon_5d", results)
        res_5d = results["horizon_5d"]["mean_depth"]
        # Small scale varying series should correctly compute non-zero correlation
        self.assertAlmostEqual(res_5d["fwd_vol_pearson_corr"], 1.0, places=4)

    def test_walk_forward_backtest_execution_and_export(self):
        config = QuantMacroConfig(
            lookback=126,
            depth_window=63,
            rebalance_frequency="monthly",
            transaction_cost_bps=5.0,
            rooting_method="lagged_correlation",
            rooting_surrogates=10,
            random_state=42,
        )
        backtester = WalkForwardBacktester(config=config)
        res = backtester.run(self.mdata)

        self.assertFalse(res.returns.empty)
        self.assertFalse(res.weights.empty)
        self.assertIn("sharpe_ratio", res.metrics)

        # Test export capability
        with tempfile.TemporaryDirectory() as tmpdir:
            res.export(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "results.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "weights.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "drr_states.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "summary.json")))


class TestAdaptersAndAntiLeakage(unittest.TestCase):
    """Test Qlib, Riskfolio, VectorBT adapters, negative controls, and anti-leakage invariant."""

    def setUp(self):
        self.mdata = generate_synthetic_market_data(
            symbols=["SPY", "TLT", "GLD", "HYG", "VIXY"],
            start_date="2018-01-01",
            end_date="2021-06-30",
            seed=42,
            portfolio_symbols=["SPY", "TLT", "GLD", "HYG"],
        )

    def test_anti_leakage_invariant(self):
        from drr_framework.finance.validation import assert_no_lookahead_leakage

        def run_pipeline(prices_df):
            rets = calculate_returns(prices_df)
            mdata = MarketData(
                prices=prices_df,
                returns=rets,
                observation_symbols=list(prices_df.columns),
                portfolio_symbols=list(prices_df.columns[:4]),
            )
            cfg = QuantMacroConfig(lookback=126, depth_window=63, rooting_surrogates=10)
            backtester = WalkForwardBacktester(config=cfg)
            res = backtester.run(mdata)
            return res.regimes

        # Verify running pipeline on truncated vs full data produces identical OOS signals
        assert_no_lookahead_leakage(
            run_pipeline_fn=run_pipeline,
            full_dataset=self.mdata.prices,
            cutoff_date="2020-06-30",
        )

    def test_qlib_matched_experiment(self):
        from drr_framework.finance.qlib import QlibDRRMatchedExperiment
        from drr_framework.finance.config import QuantResearchConfig

        cfg = QuantResearchConfig(depth_window=63, rooting_surrogates=10)
        exp = QlibDRRMatchedExperiment(config=cfg, target_symbol="SPY")
        res = exp.run_matched_experiment(self.mdata, model_family="linear")

        self.assertIn("control", res)
        self.assertIn("experiment", res)
        self.assertIn("ic", res["control"]["metrics"])
        self.assertIn("ic", res["experiment"]["metrics"])
        self.assertIn("ic_delta", res["summary_deltas"])

        ablation = exp.run_feature_ablation_study(self.mdata, model_family="linear")
        self.assertIn("Baseline", ablation)
        self.assertIn("Baseline + All DRR", ablation)

    def test_vectorbt_adapter_and_parameter_sweep(self):
        from drr_framework.finance.validation import VectorBTAdapter

        adapter = VectorBTAdapter(transaction_cost_bps=5.0)
        sweep_df = adapter.run_parameter_robustness_sweep(
            self.mdata,
            lookbacks=(126,),
            percentiles=(80.0,),
            depth_windows=(63,),
        )
        self.assertEqual(len(sweep_df), 1)
        self.assertIn("sharpe", sweep_df.columns)

    def test_negative_controls_and_hac(self):
        from drr_framework.finance.validation import (
            calculate_hac_standard_errors,
            benjamini_hochberg_fdr,
            run_negative_controls,
        )

        # HAC SE
        x = pd.Series(np.random.normal(0, 1, 100))
        y = pd.Series(np.random.normal(0, 1, 100))
        beta, se, t_stat = calculate_hac_standard_errors(x, y)
        self.assertIsInstance(beta, float)
        self.assertGreater(se, 0.0)

        # FDR
        pvals = [0.001, 0.01, 0.04, 0.20, 0.50]
        sig, adj = benjamini_hochberg_fdr(pvals)
        self.assertEqual(len(sig), 5)
        self.assertTrue(sig[0])

        # Negative Controls
        states = pd.DataFrame(
            {"mean_depth": np.random.uniform(0.2, 0.8, 100)}, index=self.mdata.returns.index[:100]
        )
        res = run_negative_controls(states, self.mdata.returns.iloc[:100], n_shuffles=20)
        self.assertIn("empirical_p_value", res)


if __name__ == "__main__":
    unittest.main()
