"""Behavior contracts for the numerical/presentation seam."""

import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from drr_framework import DynamicResonanceRooting


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("network", [None, nx.DiGraph(), nx.DiGraph([("dim_0", "dim_1")])])
def test_plot_preserves_panels_and_display(monkeypatch, network):
    analyzer = DynamicResonanceRooting(sampling_rate=64.0)
    data = np.sin(2 * np.pi * 4 * np.arange(128) / 64)
    analyzer.detect_resonances(data)
    results = {
        "resonances": analyzer.resonances,
        "resonance_depths": {"dim_0": 0.75},
    }
    if network is not None:
        results["influence_network"] = network
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))

    assert analyzer.plot_results(results, data) is None

    assert len(shown) == 1
    axes = shown[0].axes
    assert len(axes) == (4 if network is None else 6)
    assert axes[0].get_title() == "Original Time Series"
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), data)
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), np.arange(128) / 64)
    assert axes[1].get_title() == "Phase Space Reconstruction"
    assert axes[2].get_title() == "Power Spectrum with Detected Resonances"
    assert axes[3].patches[0].get_height() == 0.75
    if network is not None:
        assert axes[4].get_title() == "Influence Network"
        assert f"Network Nodes: {len(network)}" in axes[5].texts[0].get_text()


def test_plot_fallback_and_headless_save(monkeypatch, tmp_path):
    analyzer = DynamicResonanceRooting()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Headless plot displayed"))
    saved = []
    original_savefig = plt.savefig

    def savefig(*args, **kwargs):
        saved.append(plt.gcf())
        original_savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", savefig)
    analyzer.plot_results(
        {"resonances": {"dim_0": {"frequencies": []}}},
        np.zeros((32, 4)),
        save_plots=True,
        show=False,
    )
    assert (tmp_path / "drr_analysis_results.png").read_bytes().startswith(b"\x89PNG")
    assert saved[0].axes[0].get_title() == "Original Time Series (First 3 Dimensions)"
    assert len(saved[0].axes[0].lines) == 3
    assert saved[0].axes[1].texts[0].get_text() == "Phase space\nnot available"
    assert saved[0].axes[2].texts[0].get_text() == "No resonances\ndetected"
    assert not plt.get_fignums()


def test_empty_results_do_not_create_figure(caplog):
    DynamicResonanceRooting().plot_results({}, np.zeros(32), show=False)
    assert "No results to plot" in caplog.text
    assert not plt.get_fignums()


def test_renderer_accepts_results_without_analyzer(monkeypatch):
    from drr_framework.visualizations import plot_analysis_results

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    data = np.arange(32)
    plot_analysis_results(
        {"resonance_depths": {"dim_0": 0.5}},
        data,
        sampling_rate=8.0,
        embedding_dim=2,
        tau=1,
    )
    np.testing.assert_allclose(shown[0].axes[0].lines[0].get_xdata(), data / 8)
    assert shown[0].axes[3].patches[0].get_height() == 0.5


def test_compact_notebook_helper_preserves_input_trace(monkeypatch):
    from drr_framework.visualizations import plot_results

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    data = np.arange(32)
    plot_results({}, data)
    assert len(shown[0].axes) == 1
    assert shown[0].axes[0].get_title() == "DRR Analysis Input Trace"
    np.testing.assert_array_equal(shown[0].axes[0].lines[0].get_ydata(), data)


def test_numerical_analysis_does_not_load_matplotlib():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockMatplotlib(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                    raise ModuleNotFoundError("Matplotlib blocked for numerical analysis")
        sys.meta_path.insert(0, BlockMatplotlib())
        sys.path.insert(0, "src")
        import numpy as np
        from drr_framework import DynamicResonanceRooting
        t = np.arange(128) / 64
        analyzer = DynamicResonanceRooting(sampling_rate=64)
        result = analyzer.analyze_system(
            np.column_stack([np.sin(2*np.pi*4*t), np.cos(2*np.pi*4*t)]),
            multivariate=True, window_size=32, rooting_n_surrogates=0,
        )
        assert len(result["resonance_depths"]) == 2
        assert "error" not in result["state_space_analysis"]
        assert "error" not in result["rooting_analysis"]
        assert "matplotlib" not in sys.modules
        assert "drr_framework.visualizations" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
