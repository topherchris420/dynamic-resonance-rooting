"""DSGE-inspired state-space tools for Dynamic Resonance Rooting.

The New York Fed DSGE.jl package organizes dynamic models around explicit
transition and measurement equations. This module brings that discipline into
DRR without depending on Julia: resonance systems can be represented as
linear-Gaussian state-space models, filtered with a Kalman filter, and probed
with impulse responses and stability diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

_EPS = np.finfo(float).eps


def _as_2d_data(data: np.ndarray, name: str = "data", allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D time series")
    if array.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three observations")
    if allow_nan:
        finite_or_missing = np.isfinite(array) | np.isnan(array)
        if not np.all(finite_or_missing):
            raise ValueError(f"{name} contains invalid values")
    elif not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_square(matrix: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_matrix(matrix: np.ndarray, rows: int, cols: int, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (rows, cols):
        raise ValueError(f"{name} must have shape ({rows}, {cols})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_vector(vector: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _symmetrize_psd(matrix: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    return matrix + np.eye(matrix.shape[0]) * jitter


def _safe_inverse_and_logdet(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = _symmetrize_psd(matrix)
    sign, logdet = np.linalg.slogdet(matrix)
    if sign > 0 and np.isfinite(logdet):
        return np.linalg.inv(matrix), float(logdet)

    jittered = _symmetrize_psd(matrix, jitter=1e-6)
    sign, logdet = np.linalg.slogdet(jittered)
    if sign > 0 and np.isfinite(logdet):
        return np.linalg.inv(jittered), float(logdet)

    return np.linalg.pinv(jittered), float(np.log(max(np.linalg.det(jittered), _EPS)))


@dataclass(frozen=True)
class Transition:
    """State transition equation: ``s_t = TTT s_{t-1} + RRR eps_t + CCC``."""

    TTT: np.ndarray
    RRR: np.ndarray
    CCC: np.ndarray

    def __post_init__(self) -> None:
        n_states = np.asarray(self.TTT).shape[0]
        object.__setattr__(self, "TTT", _as_square(self.TTT, n_states, "TTT"))
        object.__setattr__(
            self, "RRR", _as_matrix(self.RRR, n_states, np.asarray(self.RRR).shape[1], "RRR")
        )
        object.__setattr__(self, "CCC", _as_vector(self.CCC, n_states, "CCC"))


@dataclass(frozen=True)
class Measurement:
    """Measurement equation: ``y_t = ZZ s_t + DD + u_t``."""

    ZZ: np.ndarray
    DD: np.ndarray
    QQ: np.ndarray
    EE: np.ndarray

    def __post_init__(self) -> None:
        n_observables, n_states = np.asarray(self.ZZ).shape
        n_shocks = np.asarray(self.QQ).shape[0]
        object.__setattr__(self, "ZZ", _as_matrix(self.ZZ, n_observables, n_states, "ZZ"))
        object.__setattr__(self, "DD", _as_vector(self.DD, n_observables, "DD"))
        object.__setattr__(self, "QQ", _as_square(self.QQ, n_shocks, "QQ"))
        object.__setattr__(self, "EE", _as_square(self.EE, n_observables, "EE"))


@dataclass(frozen=True)
class StateSpaceSystem:
    """A DRR state-space system with DSGE-style transition and measurement blocks."""

    transition: Transition
    measurement: Measurement
    state_names: Sequence[str]
    observable_names: Sequence[str]
    shock_names: Sequence[str]

    def __post_init__(self) -> None:
        n_states = self.transition.TTT.shape[0]
        n_observables = self.measurement.ZZ.shape[0]
        n_shocks = self.transition.RRR.shape[1]

        if self.measurement.ZZ.shape[1] != n_states:
            raise ValueError("ZZ column count must match number of states")
        if self.measurement.QQ.shape != (n_shocks, n_shocks):
            raise ValueError("QQ shape must match number of shocks")
        if len(self.state_names) != n_states:
            raise ValueError("state_names length must match number of states")
        if len(self.observable_names) != n_observables:
            raise ValueError("observable_names length must match number of observables")
        if len(self.shock_names) != n_shocks:
            raise ValueError("shock_names length must match number of shocks")

        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "observable_names", tuple(self.observable_names))
        object.__setattr__(self, "shock_names", tuple(self.shock_names))

    @property
    def n_states(self) -> int:
        return self.transition.TTT.shape[0]

    @property
    def n_observables(self) -> int:
        return self.measurement.ZZ.shape[0]

    @property
    def n_shocks(self) -> int:
        return self.transition.RRR.shape[1]

    def __getitem__(self, key: str) -> np.ndarray:
        if key in {"TTT", "RRR", "CCC"}:
            return getattr(self.transition, key)
        if key in {"ZZ", "DD", "QQ", "EE"}:
            return getattr(self.measurement, key)
        raise KeyError(key)

    def diagnostics(self) -> Dict[str, float | bool]:
        eigenvalues = np.linalg.eigvals(self.transition.TTT)
        spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        return {
            "spectral_radius": spectral_radius,
            "is_stable": spectral_radius < 1.0,
            "transition_condition_number": float(np.linalg.cond(self.transition.TTT)),
            "measurement_condition_number": float(np.linalg.cond(self.measurement.ZZ)),
            "state_count": self.n_states,
            "observable_count": self.n_observables,
            "shock_count": self.n_shocks,
        }


@dataclass(frozen=True)
class KalmanResult:
    """Output of the DRR Kalman filter."""

    log_likelihood: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    final_state: np.ndarray
    final_covariance: np.ndarray
    innovations: np.ndarray

    @property
    def total_log_likelihood(self) -> float:
        return float(np.sum(self.log_likelihood))

    def summary(self) -> Dict[str, float | list[float]]:
        rmse = np.sqrt(np.nanmean(self.innovations**2, axis=0))
        rmse = np.where(np.isfinite(rmse), rmse, 0.0)
        return {
            "total_log_likelihood": self.total_log_likelihood,
            "mean_log_likelihood": float(np.mean(self.log_likelihood)),
            "innovation_rmse": [float(value) for value in rmse],
            "final_state_norm": float(np.linalg.norm(self.final_state)),
            "final_uncertainty_trace": float(np.trace(self.final_covariance)),
        }


def fit_resonance_state_space(
    data: np.ndarray,
    ridge: float = 1e-6,
    measurement_error_scale: float = 1e-3,
    state_names: Optional[Sequence[str]] = None,
    observable_names: Optional[Sequence[str]] = None,
) -> StateSpaceSystem:
    """Estimate a linear DRR state-space system from observed trajectories.

    The estimator is intentionally conservative: it fits a VAR(1)-style
    transition with ridge regularization, uses identity measurement loading, and
    derives process/measurement covariances from residual energy. That gives DRR
    a DSGE-like model object while keeping the method deterministic and testable.
    """
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if measurement_error_scale < 0:
        raise ValueError("measurement_error_scale must be non-negative")

    observations = _as_2d_data(data)
    n_obs, n_variables = observations.shape

    x = observations[:-1]
    y = observations[1:]
    design = np.column_stack([x, np.ones(n_obs - 1)])

    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    transition_matrix = coef[:-1].T
    transition_intercept = coef[-1]

    residuals = y - design @ coef
    if residuals.shape[0] > 1:
        shock_covariance = np.cov(residuals, rowvar=False)
    else:
        shock_covariance = np.eye(n_variables) * _EPS
    shock_covariance = np.atleast_2d(shock_covariance)
    shock_covariance = _symmetrize_psd(shock_covariance)

    variable_variance = np.var(observations, axis=0)
    measurement_covariance = np.diag(np.maximum(variable_variance * measurement_error_scale, 1e-9))

    names = tuple(state_names or [f"state_{i}" for i in range(n_variables)])
    observables = tuple(observable_names or [f"obs_{i}" for i in range(n_variables)])
    shocks = tuple(f"shock_{i}" for i in range(n_variables))

    transition = Transition(
        TTT=transition_matrix,
        RRR=np.eye(n_variables),
        CCC=transition_intercept,
    )
    measurement = Measurement(
        ZZ=np.eye(n_variables),
        DD=np.zeros(n_variables),
        QQ=shock_covariance,
        EE=measurement_covariance,
    )
    return StateSpaceSystem(
        transition=transition,
        measurement=measurement,
        state_names=names,
        observable_names=observables,
        shock_names=shocks,
    )


def kalman_filter(
    system: StateSpaceSystem,
    observations: np.ndarray,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
) -> KalmanResult:
    """Filter observations through a DRR state-space system."""
    y = _as_2d_data(observations, name="observations", allow_nan=True)
    if y.shape[1] != system.n_observables:
        raise ValueError("observations column count must match system observables")

    n_obs = y.shape[0]
    n_states = system.n_states
    n_observables = system.n_observables

    state = (
        _as_vector(initial_state, n_states, "initial_state")
        if initial_state is not None
        else np.zeros(n_states)
    )
    covariance = (
        _as_square(initial_covariance, n_states, "initial_covariance")
        if initial_covariance is not None
        else np.eye(n_states)
    )

    TTT = system["TTT"]
    RRR = system["RRR"]
    CCC = system["CCC"]
    ZZ = system["ZZ"]
    DD = system["DD"]
    QQ = system["QQ"]
    EE = system["EE"]
    process_covariance = _symmetrize_psd(RRR @ QQ @ RRR.T)

    predicted_state = np.zeros((n_obs, n_states))
    predicted_covariance = np.zeros((n_obs, n_states, n_states))
    filtered_state = np.zeros((n_obs, n_states))
    filtered_covariance = np.zeros((n_obs, n_states, n_states))
    innovations = np.full((n_obs, n_observables), np.nan)
    log_likelihood = np.zeros(n_obs)

    for t in range(n_obs):
        state_pred = TTT @ state + CCC
        covariance_pred = _symmetrize_psd(TTT @ covariance @ TTT.T + process_covariance)

        predicted_state[t] = state_pred
        predicted_covariance[t] = covariance_pred

        observed = np.isfinite(y[t])
        if not np.any(observed):
            state = state_pred
            covariance = covariance_pred
            filtered_state[t] = state
            filtered_covariance[t] = covariance
            continue

        H = ZZ[observed]
        d = DD[observed]
        R = EE[np.ix_(observed, observed)]
        innovation = y[t, observed] - (H @ state_pred + d)
        innovation_covariance = _symmetrize_psd(H @ covariance_pred @ H.T + R)
        inv_innovation_covariance, logdet = _safe_inverse_and_logdet(innovation_covariance)

        gain = covariance_pred @ H.T @ inv_innovation_covariance
        state = state_pred + gain @ innovation
        identity = np.eye(n_states)
        covariance = _symmetrize_psd((identity - gain @ H) @ covariance_pred)

        filtered_state[t] = state
        filtered_covariance[t] = covariance
        innovations[t, observed] = innovation

        dim = int(np.sum(observed))
        quadratic = float(innovation.T @ inv_innovation_covariance @ innovation)
        log_likelihood[t] = -0.5 * (dim * np.log(2 * np.pi) + logdet + quadratic)

    return KalmanResult(
        log_likelihood=log_likelihood,
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        filtered_state=filtered_state,
        filtered_covariance=filtered_covariance,
        initial_state=(
            np.asarray(initial_state, dtype=float).copy()
            if initial_state is not None
            else np.zeros(n_states)
        ),
        initial_covariance=(
            np.asarray(initial_covariance, dtype=float).copy()
            if initial_covariance is not None
            else np.eye(n_states)
        ),
        final_state=filtered_state[-1].copy(),
        final_covariance=filtered_covariance[-1].copy(),
        innovations=innovations,
    )


@dataclass(frozen=True)
class ChandrasekharResult:
    """Output of the Chandrasekhar recursion likelihood evaluation."""

    log_likelihood: np.ndarray
    innovations: np.ndarray

    @property
    def total_log_likelihood(self) -> float:
        return float(np.sum(self.log_likelihood))


def stationary_initialization(system: StateSpaceSystem) -> tuple[np.ndarray, np.ndarray]:
    """Unconditional (stationary) state mean and covariance of the system.

    Returns ``(mu, Sigma)`` where ``mu = (I - TTT)^{-1} CCC`` and ``Sigma`` solves
    the discrete Lyapunov equation ``Sigma = TTT Sigma TTT' + RRR QQ RRR'``. This
    matches ``init_stationary_states`` in ``StateSpaceRoutines.jl`` and is the
    initialization required by :func:`chandrasekhar_recursion`.
    """
    TTT = system["TTT"]
    RRR = system["RRR"]
    CCC = system["CCC"]
    QQ = system["QQ"]
    n_states = system.n_states

    eigenvalues = np.linalg.eigvals(TTT)
    if np.max(np.abs(eigenvalues)) >= 1.0 - 1e-9:
        raise ValueError(
            "stationary initialization requires a stable transition (spectral radius < 1)"
        )

    mean = np.linalg.solve(np.eye(n_states) - TTT, CCC)
    process_covariance = _symmetrize_psd(RRR @ QQ @ RRR.T)
    # vec(Sigma) = (I - TTT (x) TTT)^{-1} vec(RQR')
    kron = np.kron(TTT, TTT)
    vec_sigma = np.linalg.solve(np.eye(n_states * n_states) - kron, process_covariance.reshape(-1))
    covariance = _symmetrize_psd(vec_sigma.reshape(n_states, n_states))
    return mean, covariance


def chandrasekhar_recursion(
    system: StateSpaceSystem,
    observations: np.ndarray,
) -> ChandrasekharResult:
    """Evaluate the log-likelihood via the Chandrasekhar recursions.

    This is a dependency-free port of ``chand_recursion`` from the New York Fed's
    ``StateSpaceRoutines.jl``. For a *stable, time-invariant* system with complete
    data it returns exactly the Kalman log-likelihood (under stationary
    initialization), but propagates two small, fixed-size matrices (``W`` of shape
    ``(n_states, n_observables)`` and ``M`` of shape
    ``(n_observables, n_observables)``) instead of the full ``n_states`` by
    ``n_states`` covariance, which is materially faster when the state dimension is
    large.

    The recursion is only valid from the stationary prediction covariance, so the
    filter is initialized via :func:`stationary_initialization`. Missing
    observations are not supported; use :func:`kalman_filter` for those. To
    reproduce this value with :func:`kalman_filter`, pass the same stationary
    ``initial_state`` and ``initial_covariance``.

    See Herbst (2015), "Using the 'Chandrasekhar Recursions' for Likelihood
    Evaluation of DSGE Models".
    """
    y = _as_2d_data(observations, name="observations", allow_nan=False)
    if y.shape[1] != system.n_observables:
        raise ValueError("observations column count must match system observables")

    TTT = system["TTT"]
    CCC = system["CCC"]
    ZZ = system["ZZ"]
    DD = system["DD"]

    n_observables = system.n_observables

    predicted_state, predicted_covariance = stationary_initialization(system)

    V = _symmetrize_psd(ZZ @ predicted_covariance @ ZZ.T + system["EE"])
    W = TTT @ predicted_covariance @ ZZ.T
    V_inverse, _ = _safe_inverse_and_logdet(V)
    M = -V_inverse
    K = W @ V_inverse

    n_obs = y.shape[0]
    log_likelihood = np.zeros(n_obs)
    innovations = np.zeros((n_obs, n_observables))

    for t in range(n_obs):
        innovation = y[t] - (ZZ @ predicted_state + DD)
        current_inverse, logdet = _safe_inverse_and_logdet(V)
        quadratic = float(innovation.T @ current_inverse @ innovation)
        log_likelihood[t] = -0.5 * (n_observables * np.log(2 * np.pi) + logdet + quadratic)
        innovations[t] = innovation

        predicted_state = CCC + TTT @ predicted_state + K @ innovation

        ZW = ZZ @ W
        V_next = _symmetrize_psd(V + ZW @ M @ ZW.T)
        inverse_next, _ = _safe_inverse_and_logdet(V_next)
        K_next = (K @ V + TTT @ W @ M @ ZW.T) @ inverse_next
        W_next = TTT @ W - K_next @ ZW
        M_next = M + M @ ZW.T @ current_inverse @ ZW @ M

        V, K, W, M = V_next, K_next, W_next, M_next

    return ChandrasekharResult(log_likelihood=log_likelihood, innovations=innovations)


def impulse_response(
    system: StateSpaceSystem,
    horizon: int = 20,
    shock_index: int = 0,
    shock_size: float = 1.0,
) -> Dict[str, object]:
    """Compute state and observable impulse responses for one structural shock."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not 0 <= shock_index < system.n_shocks:
        raise ValueError("shock_index out of range")

    shock = np.zeros(system.n_shocks)
    shock[shock_index] = shock_size

    state_response = np.zeros((horizon + 1, system.n_states))
    observable_response = np.zeros((horizon + 1, system.n_observables))
    state_response[0] = system["RRR"] @ shock
    observable_response[0] = system["ZZ"] @ state_response[0]

    for step in range(1, horizon + 1):
        state_response[step] = system["TTT"] @ state_response[step - 1]
        observable_response[step] = system["ZZ"] @ state_response[step]

    return {
        "shock_name": system.shock_names[shock_index],
        "state_response": state_response,
        "observable_response": observable_response,
    }


def analyze_resonance_state_space(
    data: np.ndarray,
    impulse_horizon: int = 12,
    ridge: float = 1e-6,
    state_names: Optional[Sequence[str]] = None,
    observable_names: Optional[Sequence[str]] = None,
    smooth: bool = True,
    smoother_method: str = "koopman",
) -> Dict[str, object]:
    """Fit, filter, smooth, and diagnose a DRR state-space representation.

    Parameters
    ----------
    data:
        ``(T, n_variables)`` observed trajectory.
    impulse_horizon:
        Horizon for the impulse-response diagnostics.
    ridge:
        Ridge penalty for the VAR(1) transition estimate.
    state_names, observable_names:
        Optional labels.
    smooth:
        When ``True`` (default) a fixed-interval Kalman smoother is run to
        recover retrospective state and structural-shock estimates.
    smoother_method:
        ``"koopman"`` (disturbance smoother, default) or ``"hamilton"`` (RTS).
    """
    observations = _as_2d_data(data)
    system = fit_resonance_state_space(
        observations,
        ridge=ridge,
        state_names=state_names,
        observable_names=observable_names,
    )
    kalman = kalman_filter(system, observations)
    responses = [
        impulse_response(system, horizon=impulse_horizon, shock_index=index)
        for index in range(system.n_shocks)
    ]
    diagnostics: Dict[str, object] = dict(system.diagnostics())
    diagnostics.update(kalman.summary())

    result: Dict[str, object] = {
        "system": system,
        "kalman": kalman,
        "diagnostics": diagnostics,
        "impulse_responses": responses,
    }

    if smooth:
        # Local import avoids a module-level cycle (smoothers imports state_space).
        from .smoothers import kalman_smoother

        smoothed = kalman_smoother(system, observations, method=smoother_method, kalman=kalman)
        result["smoother"] = smoothed
        diagnostics.update(smoothed.summary())

    return result
