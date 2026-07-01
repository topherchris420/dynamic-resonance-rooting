"""Synthetic benchmark systems used by DRR examples and tests."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkSystems:
    """Factory methods for canonical dynamical-system benchmark data."""

    @staticmethod
    def generate_lorenz_data(
        duration: float = 30,
        dt: float = 0.01,
        initial_state: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Lorenz-system data with deterministic Euler integration."""

        logger.info("Generating Lorenz benchmark data")
        state = np.asarray(initial_state or [1.0, 1.0, 1.0], dtype=float)
        n_steps = int(duration / dt)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = state
        sigma, rho, beta = 10, 28, 8 / 3
        for i in range(n_steps - 1):
            x, y, z = xyz[i]
            xyz[i + 1] = [
                x + sigma * (y - x) * dt,
                y + (x * (rho - z) - y) * dt,
                z + (x * y - beta * z) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xyz

    @staticmethod
    def generate_heston_data(
        duration: float = 252,
        dt: float = 1 / 252,
        initial_state: Optional[dict] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stochastic-volatility data from a compact Heston model.

        Args:
            duration: Number of model years or periods represented by the run.
            dt: Step size.
            initial_state: Optional mapping with ``s0`` and ``v0`` keys.
            random_state: Optional seed for reproducible stochastic draws.
        """

        logger.info("Generating Heston benchmark data")
        state = initial_state or {"s0": 100, "v0": 0.04}
        rng = np.random.default_rng(random_state) if random_state is not None else None
        n_steps = int(duration * (1 / dt))
        s = np.zeros(n_steps)
        v = np.zeros(n_steps)
        s[0] = state["s0"]
        v[0] = state["v0"]

        kappa, theta, sigma, rho = 2.0, 0.04, 0.2, -0.7

        for i in range(1, n_steps):
            if rng is None:
                w_s = np.random.normal()
                w_v = rho * w_s + np.sqrt(1 - rho**2) * np.random.normal()
            else:
                w_s = rng.normal()
                w_v = rho * w_s + np.sqrt(1 - rho**2) * rng.normal()

            s[i] = s[i - 1] * np.exp((0.05 - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * w_s)
            v[i] = np.maximum(
                0,
                v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * w_v,
            )

        t = np.linspace(0, duration, n_steps)
        return t, np.vstack((s, v)).T

    @staticmethod
    def generate_fitzhugh_nagumo_data(
        duration: float = 500,
        dt: float = 0.1,
        initial_state: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate FitzHugh-Nagumo excitable-system benchmark data."""

        logger.info("Generating FitzHugh-Nagumo benchmark data")
        state = np.asarray(initial_state or [0.1, 0.1], dtype=float)
        n_steps = int(duration / dt)
        xy = np.zeros((n_steps, 2))
        xy[0] = state
        a, b, c = 0.7, 0.8, 0.08

        for i in range(n_steps - 1):
            x, y = xy[i]
            xy[i + 1] = [
                x + (x - x**3 / 3 - y) * dt,
                y + c * (x + a - b * y) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xy

    @staticmethod
    def generate_rossler_data(
        duration: float = 30,
        dt: float = 0.01,
        initial_state: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Roessler-system data with deterministic Euler integration."""

        logger.info("Generating Roessler benchmark data")
        state = np.asarray(initial_state or [1.0, 1.0, 1.0], dtype=float)
        n_steps = int(duration / dt)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = state
        a, b, c = 0.2, 0.2, 5.7
        for i in range(n_steps - 1):
            x, y, z = xyz[i]
            xyz[i + 1] = [
                x + (-y - z) * dt,
                y + (x + a * y) * dt,
                z + (b + z * (x - c)) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xyz
