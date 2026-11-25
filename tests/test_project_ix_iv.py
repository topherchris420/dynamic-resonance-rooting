import numpy as np
import pytest

from drr_framework.project_ix_iv import (
    benchmark,
    gram_action,
    rectangular_rank_bound,
    strassen_applies,
    strassen_step,
)


def test_rank_bound():
    assert rectangular_rank_bound(2, 2, 2) == 5
    assert rectangular_rank_bound(16, 16, 16) == 512
    assert rectangular_rank_bound(1, 7, 11) == 77
    assert rectangular_rank_bound(10, 11, 12) == rectangular_rank_bound(12, 10, 11)
    with pytest.raises(ValueError):
        rectangular_rank_bound(0, 1, 1)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 17, 32])
def test_strassen_step(n):
    rng = np.random.default_rng(15)
    a, b = rng.standard_normal((2, n, n))
    np.testing.assert_allclose(strassen_step(a, b), a @ b, atol=1e-10, rtol=1e-9)


def test_gram_action():
    rng = np.random.default_rng(42)
    f = rng.standard_normal((33, 4))
    x = rng.standard_normal((33, 3))
    np.testing.assert_allclose(gram_action(f, x), (f @ f.T) @ x, atol=1e-10)
    with pytest.raises(ValueError):
        gram_action(f, x[:2])


def test_report_schema_and_accuracy():
    report = benchmark(sizes=(16,), seeds=(19,), repeats=3)
    assert report["all_accurate"]
    assert {row["case"] for row in report["rows"]} == {
        "dense",
        "gram_action",
        "sparse_transition",
    }
    for row in report["rows"]:
        assert row["accurate"]
        assert len(row["raw_seconds"]["baseline"]) == 3
        assert row["relative_frobenius_error"] <= 1e-10
    with pytest.raises(ValueError):
        benchmark(sizes=(16,), repeats=1)


def test_odd_size_dense_rows_are_fallbacks_without_speed_gates():
    assert [strassen_applies(n) for n in (1, 2, 3, 16, 17)] == [False, True, False, True, False]
    report = benchmark(sizes=(3, 4), seeds=(19,), repeats=3)
    assert report["all_accurate"]
    rows = {(row["case"], row["n"]): row for row in report["rows"]}
    odd = rows[("dense", 3)]
    assert odd["candidate_method"] == "numpy_fallback_odd_size"
    assert odd["speed_gates_applicable"] is False
    assert odd["passes_one_shot_gate"] is None
    assert odd["passes_reused_operation_gate"] is None
    even = rows[("dense", 4)]
    assert even["candidate_method"] == "strassen_one_level"
    assert even["speed_gates_applicable"] is True
    assert isinstance(even["passes_one_shot_gate"], bool)
    assert isinstance(even["passes_reused_operation_gate"], bool)
    for case in ("gram_action", "sparse_transition"):
        assert rows[(case, 3)]["speed_gates_applicable"] is True
        assert isinstance(rows[(case, 3)]["passes_one_shot_gate"], bool)
