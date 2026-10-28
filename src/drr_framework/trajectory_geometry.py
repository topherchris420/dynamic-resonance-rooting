"""Oriented area of a reconstructed trajectory, and the limit of what it can add.

Two channels trace a curve ``(x(t), y(t))``. Its oriented area
``A_xy = (1/2) oint (x dy - y dx)`` is positive when the curve turns
counterclockwise; for two oscillations that means ``x`` leads ``y`` by less than
half a period. The matrix of pairwise areas is the lead matrix of cyclicity
analysis (Baryshnikov and Schlafly 2016, IEEE Conference on Decision and
Control) and the
antisymmetric second-level term of the path signature (the Levy area).

**Identity.** For centred samples ``x_t, y_t``, ``t = 0..N-1``, the discrete area

    A_xy = (1/2) sum_{t=0}^{N-2} [x_t (y_{t+1} - y_t) - y_t (x_{t+1} - x_t)]

equals ``((N - 1) / 2) (C_xy(1) - C_yx(1))`` with the lag-one cross-covariance
``C_xy(1) = (1 / (N - 1)) sum_t x_t y_{t+1}``. *Proof.* Expanding the bracket,
``x_t y_t`` and ``y_t x_t`` cancel, which leaves
``(1/2) sum_t (x_t y_{t+1} - y_t x_{t+1})``. QED.

The oriented area is therefore a function of second-order statistics. It cannot
carry information that the lag-one cross-covariance, and hence the cross
spectrum, does not. For ``x_t = cos(omega t)`` and ``y_t = cos(omega t - theta)``
the normalized area is ``sin(omega) sin(theta)`` (``omega`` in radians per
sample): zero for in-phase and anti-phase coupling, and a mixture of frequency
and phase lag otherwise. Its sign says which channel leads only modulo a
period, because a lag of ``d`` samples looks exactly like a lead of ``P - d``.

These are geometric descriptions. A curve that looks like a coupled system's
is not evidence that the channels interact, and a lead in area is not a cause.

The geometric framing draws on the *Geometry* volume of Discovering the Art of
Mathematics (https://www.artofmathematics.org/books); the identity and its proof
are elementary and given in full above.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

#: Identifier of this module's definitions.
GEOMETRY_METHOD_VERSION = "drr_trajectory_geometry_v1"


def _as_matrix(data: Any) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("data must have shape (time >= 3, channels >= 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("data contains non-finite values")
    return values - values.mean(axis=0)


def oriented_area_matrix(data: Any, normalize: bool = True) -> np.ndarray:
    """Pairwise oriented areas ``A_ij`` of the centred trajectory (antisymmetric).

    With ``normalize`` the areas are divided by ``(N - 1) sigma_i sigma_j``, which
    makes them dimensionless, invariant to gain, and comparable across record
    lengths. Constant channels get zero area.
    """
    x = _as_matrix(data)
    steps = np.diff(x, axis=0)
    head = x[:-1]
    area = 0.5 * (head.T @ steps - steps.T @ head)
    if normalize:
        sigma = x.std(axis=0)
        scale = (x.shape[0] - 1) * np.outer(sigma, sigma)
        area = np.divide(area, scale, out=np.zeros_like(area), where=scale > 0)
    return area


def lag_one_antisymmetric_covariance(data: Any) -> np.ndarray:
    """``((N - 1) / 2) (C(1) - C(1)^T)``, equal to the unnormalized oriented area."""
    x = _as_matrix(data)
    lagged = x[:-1].T @ x[1:] / (x.shape[0] - 1)
    return 0.5 * (x.shape[0] - 1) * (lagged - lagged.T)


def cyclic_order(area: Any) -> Dict[str, Any]:
    """Cyclic lead order from the leading eigenvector of an antisymmetric area matrix.

    ``i A`` is Hermitian; the phases of the eigenvector of its largest
    eigenvalue order the channels around a cycle (Baryshnikov and Schlafly
    2016). Phases are measured from the channel with the largest loading, and
    ``order`` lists channels by decreasing phase, so each channel tends to lead
    the next. ``leading_share`` is that eigenvalue's share of the positive
    spectrum, near one when a single cycle dominates. The order is cyclic, so
    its first entry is only the reference, and it describes second-order
    statistics, not causes.
    """
    matrix = np.asarray(area, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("area must be a square matrix with at least two channels")
    if not np.allclose(matrix, -matrix.T, atol=1e-12 * max(1.0, float(np.abs(matrix).max()))):
        raise ValueError("area must be antisymmetric")
    values, vectors = np.linalg.eigh(1j * matrix)
    leading = vectors[:, -1]
    reference = int(np.argmax(np.abs(leading)))
    phases = np.angle(leading * np.conj(leading[reference]) / max(abs(leading[reference]), 1e-300))
    positive = values[values > 0]
    share = float(values[-1] / positive.sum()) if positive.size and positive.sum() > 0 else 0.0
    return {
        "order": [int(i) for i in np.argsort(-phases, kind="mergesort")],
        "phases_rad": [float(p) for p in phases],
        "leading_share": share,
    }


__all__ = [
    "GEOMETRY_METHOD_VERSION",
    "cyclic_order",
    "lag_one_antisymmetric_covariance",
    "oriented_area_matrix",
]
