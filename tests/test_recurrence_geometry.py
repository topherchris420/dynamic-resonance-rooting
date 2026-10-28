"""Recurrence quantification, linear recurrences, and the oriented-area identity."""

from __future__ import annotations

import numpy as np
import pytest

from drr_framework import DynamicResonanceRooting
from drr_framework.recurrence import (
    delay_embedding,
    recurrence_quantification,
    second_order_recurrence,
)
from drr_framework.trajectory_geometry import (
    cyclic_order,
    lag_one_antisymmetric_covariance,
    oriented_area_matrix,
)

# ------------------------------------------------------------------ embedding


def test_delay_embedding_is_the_reconstruction_drr_already_uses():
    data = np.random.default_rng(0).normal(size=200)
    drr = DynamicResonanceRooting(embedding_dim=4, tau=3)
    assert np.array_equal(drr.time_delay_embedding(data), delay_embedding(data, 4, 3))
    assert delay_embedding(np.arange(6.0), 2, 2).tolist() == [[0, 2], [1, 3], [2, 4], [3, 5]]
    with pytest.raises(ValueError, match="Data too short"):
        drr.time_delay_embedding(np.arange(9.0))
    with pytest.raises(ValueError, match="1-dimensional"):
        drr.time_delay_embedding(np.ones((10, 2)))
    with pytest.raises(ValueError, match="tau"):
        delay_embedding(np.arange(10.0), 2, 0)


# ------------------------------------------------------------------ recurrence


def test_a_sine_is_fully_deterministic_and_recurs_at_its_period():
    fs = 100.0
    t = np.arange(1024) / fs
    result = recurrence_quantification(np.sin(2 * np.pi * 4.0 * t), sampling_rate=fs, frequency=4.0)
    assert result["determinism"] > 0.9
    assert result["recurrence_period_seconds"] == pytest.approx(0.25, abs=0.02)
    assert result["recurrence_rate"] == pytest.approx(0.05, abs=0.01)
    assert result["delay"] == 6


def test_white_noise_has_low_determinism_and_no_period():
    noise = np.random.default_rng(1).normal(size=1024)
    result = recurrence_quantification(noise)
    assert result["delay"] == 2
    assert result["determinism"] < 0.2
    assert result["recurrence_period_seconds"] is None
    assert 0.0 <= result["laminarity"] <= 1.0
    # A one-sample delay shares coordinates between consecutive vectors and
    # draws spurious diagonal lines; the default refuses it.
    assert recurrence_quantification(noise, delay=1)["determinism"] > 0.4


def test_determinism_falls_as_noise_rises():
    rng = np.random.default_rng(2)
    t = np.arange(800)
    tone = np.sin(2 * np.pi * t / 25.0)
    values = [
        recurrence_quantification(tone + level * rng.normal(size=t.size), frequency=0.04)[
            "determinism"
        ]
        for level in (0.0, 0.3, 1.0, 3.0)
    ]
    assert values == sorted(values, reverse=True)


def test_recurrence_validates_its_arguments():
    data = np.random.default_rng(3).normal(size=300)
    with pytest.raises(ValueError, match="constant"):
        recurrence_quantification(np.ones(100))
    with pytest.raises(ValueError, match="max_points"):
        recurrence_quantification(data, max_points=100)
    with pytest.raises(ValueError, match="recurrence_rate"):
        recurrence_quantification(data, recurrence_rate=1.5)
    with pytest.raises(ValueError, match="Theiler"):
        recurrence_quantification(data, theiler_window=294)


# ------------------------------------------------------------------ linear recurrences


def test_second_order_recurrence_recovers_a_known_resonator():
    rho, omega = 0.97, 2 * np.pi * 0.08
    a1, a2 = 2 * rho * np.cos(omega), -(rho**2)
    rng = np.random.default_rng(4)
    x = np.zeros(30_000)
    shocks = rng.normal(size=x.size)
    for k in range(2, x.size):
        x[k] = a1 * x[k - 1] + a2 * x[k - 2] + shocks[k]
    result = second_order_recurrence(x, sampling_rate=10.0)
    assert result["oscillatory"] and result["stationary"]
    assert result["frequency_hz"] == pytest.approx(0.8, abs=0.01)
    assert result["damping_time_seconds"] == pytest.approx(-1 / (10.0 * np.log(rho)), rel=0.1)
    assert result["coefficients"] == pytest.approx([a1, a2], abs=0.02)


def test_fibonacci_recurrence_is_explosive_and_not_oscillatory():
    fibonacci = [1.0, 1.0]
    for _ in range(30):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    result = second_order_recurrence(np.array(fibonacci))
    assert not result["oscillatory"]
    assert not result["stationary"]
    assert result["root_modulus"] == pytest.approx((1 + np.sqrt(5)) / 2, rel=1e-3)
    assert result["damping_time_seconds"] is None


def test_additive_noise_makes_a_tone_look_damped_and_then_not_oscillate():
    """The documented errors-in-variables bias, measured on a unit tone."""
    rng = np.random.default_rng(5)
    t = np.arange(4000)
    tone = np.sin(2 * np.pi * 0.05 * t)
    clean = second_order_recurrence(tone + 0.01 * rng.normal(size=t.size))
    mild = second_order_recurrence(tone + 0.1 * rng.normal(size=t.size))
    heavy = second_order_recurrence(tone + 0.5 * rng.normal(size=t.size))
    assert clean["frequency_hz"] == pytest.approx(0.05, abs=0.001)
    assert mild["oscillatory"] and mild["root_modulus"] < clean["root_modulus"] - 0.1
    assert abs(mild["frequency_hz"] - 0.05) > 0.005
    assert not heavy["oscillatory"]


# ------------------------------------------------------------------ geometry


def test_oriented_area_is_the_antisymmetric_lag_one_covariance():
    data = np.random.default_rng(6).normal(size=(500, 4)).cumsum(axis=0)
    area = oriented_area_matrix(data, normalize=False)
    assert area == pytest.approx(lag_one_antisymmetric_covariance(data), abs=1e-9)
    assert area == pytest.approx(-area.T)


def test_normalized_area_of_two_cosines_is_sin_omega_sin_theta():
    t = np.arange(5000)
    omega, theta = 2 * np.pi * 0.03, 0.9
    pair = np.column_stack((np.cos(omega * t), np.cos(omega * t - theta)))
    assert oriented_area_matrix(pair)[0, 1] == pytest.approx(
        np.sin(omega) * np.sin(theta), rel=1e-3
    )
    in_phase = np.column_stack((np.cos(omega * t), 2 * np.cos(omega * t)))
    assert oriented_area_matrix(in_phase)[0, 1] == pytest.approx(0.0, abs=1e-6)
    rescaled = oriented_area_matrix(pair * np.array([3.0, 0.2]) + 5.0)
    assert rescaled == pytest.approx(oriented_area_matrix(pair))


def test_area_lead_is_defined_only_modulo_a_period():
    t = np.arange(4000)
    omega = 2 * np.pi / 20.0
    lag = np.column_stack((np.cos(omega * t), np.cos(omega * (t - 5))))
    long_lag = np.column_stack((np.cos(omega * t), np.cos(omega * (t - 15))))
    assert oriented_area_matrix(lag)[0, 1] > 0
    assert oriented_area_matrix(long_lag)[0, 1] < 0


def test_cyclic_order_lists_the_leading_channel_first():
    t = np.arange(4000)
    omega = 2 * np.pi / 40.0
    data = np.column_stack([np.cos(omega * t - k * 0.6) for k in range(3)])
    order = cyclic_order(oriented_area_matrix(data))
    assert order["order"] == [0, 1, 2]
    assert order["leading_share"] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="antisymmetric"):
        cyclic_order(np.ones((3, 3)))
    with pytest.raises(ValueError, match="channels"):
        oriented_area_matrix(np.ones((10, 1)))
