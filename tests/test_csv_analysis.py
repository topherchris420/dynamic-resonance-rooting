"""End-to-end CSV workflows and data-integrity rejection tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from drr_framework.csv_analysis import analyze_csv, main, write_bundle

SAMPLE = Path(__file__).resolve().parents[1] / "data/raw/coupled_oscillator_sample.csv"
OPTIONS = {
    "columns": ["dim_0", "dim_1", "dim_2"],
    "sampling_rate": 200.0,
    "time_column": "time_seconds",
}


def test_known_signal_reproduces_and_exports(tmp_path):
    first = analyze_csv(SAMPLE, **OPTIONS, n_surrogates=25)
    second = analyze_csv(SAMPLE, **OPTIONS, n_surrogates=25)
    assert first == second
    assert first["provenance"]["input_sha256"] == hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    assert first["channels"]["dim_0"]["dominant_frequency_hz"] == pytest.approx(12.5, abs=0.5)
    edge = next(
        e
        for e in first["rooting"]["significant_edges"]
        if e["source"] == "dim_0" and e["target"] == "dim_1"
    )
    assert edge["lag"] == 2
    assert edge["lag_seconds"] == 0.01
    assert edge["adjusted_p_value"] <= 0.05
    destination = tmp_path / "run"
    write_bundle(first, destination)
    text = (destination / "analysis.json").read_text()
    assert json.loads(text) == first
    assert "NaN" not in text and "Infinity" not in text
    report = (destination / "report.md").read_text()
    assert "12.5" in report and "do not establish causation" in report
    with pytest.raises(FileExistsError):
        write_bundle(second, destination)
    assert (destination / "analysis.json").read_text() == text


def test_inference_disabled_and_resolution_warning():
    payload = analyze_csv(SAMPLE, **OPTIONS, n_surrogates=0)
    assert payload["rooting"]["inference_available"] is False
    assert payload["rooting"]["significant_edges"] == []
    assert payload["rooting"]["p_values"][0][1] is None
    assert any("Inference is disabled" in warning for warning in payload["warnings"])
    low_resolution = analyze_csv(SAMPLE, **OPTIONS, n_surrogates=5)
    assert any("Too few surrogates" in warning for warning in low_resolution["warnings"])


def test_named_columns_and_preprocessing_provenance(tmp_path):
    path = tmp_path / "named.csv"
    path.write_text(SAMPLE.read_text().replace("dim_0,dim_1,dim_2", "source,response,distractor"))
    options = dict(OPTIONS, columns=["response", "source"])
    payload = analyze_csv(path, **options, n_surrogates=25)
    assert payload["rooting"]["matrix_columns"] == ["response", "source"]
    assert any(
        e["source"] == "source" and e["target"] == "response"
        for e in payload["rooting"]["significant_edges"]
    )
    changed = analyze_csv(path, **options, n_surrogates=25, difference=True)
    assert changed["provenance"]["analyzed_rows"] == payload["provenance"]["input_rows"] - 1
    assert changed["run_id"] != payload["run_id"]
    assert changed["provenance"]["parameters"]["difference"] is True


@pytest.mark.parametrize(
    "override,match",
    [
        ({"columns": ["dim_0"]}, "two distinct"),
        ({"columns": ["dim_0", "dim_0"]}, "two distinct"),
        ({"columns": ["dim_0", "missing"]}, "not found"),
        ({"columns": ["dim_0", "time_seconds"]}, "cannot also"),
        ({"sampling_rate": float("nan")}, "finite and positive"),
        ({"sampling_rate": float("inf")}, "finite and positive"),
        ({"sampling_rate": 0}, "finite and positive"),
        ({"sampling_rate": 100}, "match --sampling-rate"),
        ({"window_size": 99999}, "exceeds"),
        ({"window_size": 1}, "integer >= 8"),
        ({"max_lag": 99999}, "three paired"),
        ({"n_surrogates": -1}, "integer >= 0"),
        ({"n_surrogates": 1.5}, "integer >= 0"),
        ({"random_state": -1}, "integer >= 0"),
        ({"alpha": float("nan")}, "between 0 and 1"),
        ({"surrogate_method": "invalid"}, "surrogate_method"),
    ],
)
def test_invalid_configuration(override, match):
    with pytest.raises(ValueError, match=match):
        analyze_csv(SAMPLE, **dict(OPTIONS, **override))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda lines: [lines[0].replace("dim_1", "dim_0")] + lines[1:], "unique column"),
        (lambda lines: [lines[0].replace("dim_1", "")] + lines[1:], "unique column"),
        (lambda lines: lines[:3], "32 observations"),
        (lambda lines: lines[:1] + ["0,NaN,1,2"] + lines[2:], "non-finite"),
        (lambda lines: lines[:1] + ["0,inf,1,2"] + lines[2:], "non-finite"),
        (lambda lines: lines[:1] + ["0,,1,2"] + lines[2:], "finite number"),
        (lambda lines: lines[:1] + ["0,1,2"] + lines[2:], "number of fields"),
        (lambda lines: lines[:1] + ["0,1,2,3,4"] + lines[2:], "number of fields"),
        (lambda lines: lines[:1] + [""] + lines[2:], "number of fields"),
        (lambda lines: lines[:2] + [lines[1]] + lines[3:], "increase uniformly"),
        (lambda lines: lines[:8] + lines[9:], "increase uniformly"),
        (lambda lines: lines[:1] + list(reversed(lines[1:])), "increase uniformly"),
    ],
)
def test_rejects_corrupt_data(tmp_path, mutation, match):
    path = tmp_path / "bad.csv"
    path.write_text("\n".join(mutation(SAMPLE.read_text().splitlines())) + "\n")
    with pytest.raises(ValueError, match=match):
        analyze_csv(path, **OPTIONS)


def test_constant_signal_rejected(tmp_path):
    path = tmp_path / "constant.csv"
    np.savetxt(
        path,
        np.column_stack([np.ones(300), np.arange(300)]),
        delimiter=",",
        header="a,b",
        comments="",
    )
    with pytest.raises(ValueError, match="constant"):
        analyze_csv(path, columns=["a", "b"], sampling_rate=1)


def test_cli_success_and_failure_leave_no_partial_output(tmp_path, capsys):
    destination = tmp_path / "result"
    args = [
        str(SAMPLE),
        "--columns",
        "dim_0",
        "dim_1",
        "dim_2",
        "--sampling-rate",
        "200",
        "--time-column",
        "time_seconds",
        "--output-dir",
        str(destination),
    ]
    assert main(args) == 0
    assert "report.md" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
    bad_destination = tmp_path / "invalid"
    args[-1] = str(bad_destination)
    with pytest.raises(SystemExit) as error:
        main(args + ["--window-size", "99999"])
    assert error.value.code == 2
    assert not bad_destination.exists()
