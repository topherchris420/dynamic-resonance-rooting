import logging
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, hilbert, welch
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)

try:
    from pyinform import transfer_entropy

    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    logger.info("pyinform not available; transfer entropy analysis will use fallback mode")


_EPS = np.finfo(float).eps


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _peak_confidence(values: np.ndarray, noise_floor: float) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    span = max(float(np.max(values) - noise_floor), _EPS)
    return np.clip((values - noise_floor) / span, 0.0, 1.0)


class MarkovChain:
    """A first-order Markov chain over discrete states."""

    def __init__(self, states: np.ndarray):
        self.states = states
        self.unique_states = np.unique(states)
        self.n_states = len(self.unique_states)
        self.state_map = {state: i for i, state in enumerate(self.unique_states)}

        self.transition_matrix = self._calculate_transition_matrix()
        self.stationary_distribution = self._calculate_stationary_distribution()

    def _calculate_transition_matrix(self) -> np.ndarray:
        mapped = np.array([self.state_map[s] for s in self.states])
        current_indices = mapped[:-1]
        next_indices = mapped[1:]
        matrix = np.zeros((self.n_states, self.n_states))
        np.add.at(matrix, (current_indices, next_indices), 1)

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums

    def _calculate_stationary_distribution(self) -> np.ndarray:
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            one_eigenvalue_idx = np.isclose(eigenvalues, 1)

            if np.any(one_eigenvalue_idx):
                stationary_vector = eigenvectors[:, one_eigenvalue_idx].flatten().real
                if np.sum(stationary_vector) < 0:
                    stationary_vector *= -1
                stationary_vector = np.maximum(stationary_vector, 0)
                total = stationary_vector.sum()
                if total > 0:
                    return stationary_vector / total
            return np.ones(self.n_states) / self.n_states
        except np.linalg.LinAlgError:
            return np.ones(self.n_states) / self.n_states


class ResonanceDetector:
    """Detect resonances in a time series."""

    def detect(
        self,
        data: np.ndarray,
        method: str = "fft",
        sampling_rate: float = 100,
        n_clusters: int = 4,
        peak_height_ratio: float = 0.1,
    ) -> dict:
        """Detect dominant resonance structure in one time series.

        Supported methods are ``fft``, ``welch``, and ``markov``. ``wavelet`` is
        reserved as an explicit not-yet-implemented research surface.
        """

        data = np.asarray(data, dtype=float)
        if data.size == 0:
            raise ValueError("data must not be empty")
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains non-finite values")

        if method == "fft":
            return self._detect_with_fft(data, sampling_rate, peak_height_ratio)
        if method == "welch":
            return self._detect_with_welch(data, sampling_rate, peak_height_ratio)
        if method == "wavelet":
            return self._detect_with_wavelet(data)
        if method == "markov":
            return self._detect_with_markov(data, n_clusters)
        raise ValueError(f"Unknown resonance detection method: {method}")

    def _detect_with_fft(
        self, data: np.ndarray, sampling_rate: float, peak_height_ratio: float
    ) -> dict:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if not 0 < peak_height_ratio <= 1:
            raise ValueError("peak_height_ratio must be in the interval (0, 1]")

        centered = data - np.mean(data)
        n = len(centered)
        yf = rfft(centered)
        xf = rfftfreq(n, 1 / sampling_rate)
        magnitudes = np.abs(yf)

        max_magnitude = float(np.max(magnitudes)) if magnitudes.size else 0.0
        peaks, _ = find_peaks(magnitudes, height=max_magnitude * peak_height_ratio)
        dominant_freq = xf[peaks]
        peak_magnitudes = magnitudes[peaks]

        if len(peak_magnitudes) > 0:
            order = np.argsort(peak_magnitudes)[::-1]
            dominant_freq = dominant_freq[order]
            peak_magnitudes = peak_magnitudes[order]

        positive_magnitudes = magnitudes[xf > 0]
        noise_floor = float(np.median(positive_magnitudes)) if positive_magnitudes.size else 0.0

        return {
            "method": "fft",
            "dominant_freq": dominant_freq,
            "peak_magnitude": peak_magnitudes,
            "confidence": _peak_confidence(peak_magnitudes, noise_floor),
            "noise_floor": noise_floor,
            "frequency_axis": xf,
            "spectrum": magnitudes,
        }

    def _detect_with_welch(
        self, data: np.ndarray, sampling_rate: float, peak_height_ratio: float
    ) -> dict:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if not 0 < peak_height_ratio <= 1:
            raise ValueError("peak_height_ratio must be in the interval (0, 1]")

        centered = data - np.mean(data)
        nperseg = min(256, len(centered))
        if nperseg < 8:
            return self._detect_with_fft(data, sampling_rate, peak_height_ratio)

        freqs, psd = welch(
            centered,
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
        )
        max_power = float(np.max(psd)) if psd.size else 0.0
        peaks, _ = find_peaks(psd, height=max_power * peak_height_ratio)
        peaks = np.array([idx for idx in peaks if freqs[idx] > 0], dtype=int)

        dominant_freq = freqs[peaks]
        peak_power = psd[peaks]
        if len(peak_power) > 0:
            order = np.argsort(peak_power)[::-1]
            dominant_freq = dominant_freq[order]
            peak_power = peak_power[order]

        positive_psd = psd[freqs > 0]
        noise_floor = float(np.median(positive_psd)) if positive_psd.size else 0.0

        return {
            "method": "welch",
            "dominant_freq": dominant_freq,
            "peak_magnitude": peak_power,
            "confidence": _peak_confidence(peak_power, noise_floor),
            "noise_floor": noise_floor,
            "frequency_axis": freqs,
            "spectrum": psd,
        }

    def _detect_with_wavelet(self, data: np.ndarray) -> dict:
        raise NotImplementedError(
            "Wavelet resonance detection is not implemented yet. "
            "Use method='fft', method='welch', or method='markov'."
        )

    def _detect_with_markov(self, data: np.ndarray, n_clusters: int) -> dict:
        if n_clusters <= 1:
            raise ValueError("n_clusters must be greater than 1")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if len(data) < n_clusters:
            raise ValueError("n_clusters cannot exceed number of observations")

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        states = kmeans.fit_predict(data)
        mc = MarkovChain(states)
        resonant_states = np.where(np.diag(mc.transition_matrix) > 0.9)[0]

        return {
            "method": "markov",
            "transition_matrix": mc.transition_matrix,
            "stationary_distribution": mc.stationary_distribution,
            "resonant_states": resonant_states,
            "clusters": states,
        }


class RootingAnalyzer:
    """Analyze directed dependencies among system components."""

    def analyze(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        n_surrogates: int = 0,
        random_state: Optional[int] = None,
        alpha: float = 0.05,
        method: str = "lagged_correlation",
    ) -> dict:
        """Estimate directed lead-lag rooting scores among variables.

        The default path uses deterministic lagged correlation. If ``method`` is
        ``transfer_entropy`` and ``pyinform`` is available, the analyzer uses that
        backend and otherwise reports the fallback method explicitly.
        """

        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Data must be a multivariate time series with at least two variables.")
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1")
        if n_surrogates < 0:
            raise ValueError("n_surrogates must be non-negative")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in the interval (0, 1)")

        data = np.asarray(data, dtype=float)
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains non-finite values")

        if method == "transfer_entropy" and PYINFORM_AVAILABLE:
            scores, effective_lag = self._transfer_entropy_scores(data, max_lag)
            method_used = "transfer_entropy"
        else:
            if method == "transfer_entropy" and not PYINFORM_AVAILABLE:
                logger.info("pyinform unavailable; using lagged correlation rooting")
            scores, effective_lag = self._lagged_correlation_scores(data, max_lag)
            method_used = "lagged_correlation"

        p_values = self._surrogate_p_values(
            data=data,
            observed=scores,
            max_lag=max_lag,
            n_surrogates=n_surrogates,
            random_state=random_state,
            method_used=method_used,
        )
        edge_threshold = float(np.mean(scores) + np.std(scores))
        significant_edges = self._significant_edges(
            scores, p_values, effective_lag, alpha, edge_threshold
        )

        return {
            "method": method_used,
            "transfer_entropy": scores,
            "effective_lag": effective_lag,
            "p_values": p_values,
            "alpha": alpha,
            "edge_threshold": edge_threshold,
            "significant_edges": significant_edges,
            "correction": "uncorrected_surrogate_p_value",
        }

    def _lagged_correlation_scores(
        self, data: np.ndarray, max_lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_variables = data.shape[1]
        scores = np.zeros((n_variables, n_variables), dtype=float)
        effective_lag = np.zeros((n_variables, n_variables), dtype=int)

        for source in range(n_variables):
            for target in range(n_variables):
                if source == target:
                    continue
                best_score = 0.0
                best_lag = 1
                for lag in range(1, max_lag + 1):
                    x = data[:-lag, source]
                    y = data[lag:, target]
                    if np.std(x) <= _EPS or np.std(y) <= _EPS:
                        score = 0.0
                    else:
                        score = abs(float(np.corrcoef(x, y)[0, 1]))
                    if score > best_score:
                        best_score = score
                        best_lag = lag
                scores[source, target] = best_score
                effective_lag[source, target] = best_lag
        return scores, effective_lag

    def _transfer_entropy_scores(
        self, data: np.ndarray, max_lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_variables = data.shape[1]
        scores = np.zeros((n_variables, n_variables), dtype=float)
        effective_lag = np.ones((n_variables, n_variables), dtype=int)
        data_discrete = self._discretize_data(data)

        for source in range(n_variables):
            for target in range(n_variables):
                if source == target:
                    continue
                best_score = 0.0
                best_lag = 1
                for lag in range(1, max_lag + 1):
                    try:
                        shifted_source = data_discrete[:-lag, source]
                        shifted_target = data_discrete[lag:, target]
                        score = float(transfer_entropy(shifted_source, shifted_target, k=1))
                    except Exception as exc:
                        logger.warning(
                            "Transfer entropy failed for pair (%s, %s): %s", source, target, exc
                        )
                        score = 0.0
                    if score > best_score:
                        best_score = score
                        best_lag = lag
                scores[source, target] = best_score
                effective_lag[source, target] = best_lag
        return scores, effective_lag

    def _surrogate_p_values(
        self,
        data: np.ndarray,
        observed: np.ndarray,
        max_lag: int,
        n_surrogates: int,
        random_state: Optional[int],
        method_used: str,
    ) -> np.ndarray:
        p_values = np.ones_like(observed, dtype=float)
        if n_surrogates == 0:
            off_diag = observed[~np.eye(observed.shape[0], dtype=bool)]
            threshold = float(np.mean(off_diag) + np.std(off_diag)) if off_diag.size else 0.0
            p_values[(observed > threshold) & (observed > 0)] = 0.0
            return p_values

        rng = np.random.default_rng(random_state)
        n_variables = data.shape[1]
        exceedances = np.zeros_like(observed, dtype=float)

        for _ in range(n_surrogates):
            surrogate = data.copy()
            for source in range(n_variables):
                surrogate[:, source] = rng.permutation(surrogate[:, source])
            if method_used == "transfer_entropy" and PYINFORM_AVAILABLE:
                surrogate_scores, _ = self._transfer_entropy_scores(surrogate, max_lag)
            else:
                surrogate_scores, _ = self._lagged_correlation_scores(surrogate, max_lag)
            exceedances += surrogate_scores >= observed

        p_values = (exceedances + 1.0) / (n_surrogates + 1.0)
        np.fill_diagonal(p_values, 1.0)
        return p_values

    def _significant_edges(
        self,
        scores: np.ndarray,
        p_values: np.ndarray,
        effective_lag: np.ndarray,
        alpha: float,
        edge_threshold: float,
    ) -> List[Dict[str, object]]:
        edges: List[Dict[str, object]] = []
        n_variables = scores.shape[0]
        for source in range(n_variables):
            for target in range(n_variables):
                if source == target:
                    continue
                score = float(scores[source, target])
                p_value = float(p_values[source, target])
                if score > edge_threshold and p_value <= alpha:
                    edges.append(
                        {
                            "source": f"dim_{source}",
                            "target": f"dim_{target}",
                            "weight": score,
                            "p_value": p_value,
                            "lag": int(effective_lag[source, target]),
                        }
                    )
        edges.sort(key=lambda edge: float(cast(float, edge["weight"])), reverse=True)
        return edges

    def _discretize_data(self, data: np.ndarray, n_bins: int = 10) -> np.ndarray:
        data_discrete = np.zeros_like(data, dtype=int)

        for i in range(data.shape[1]):
            _, bin_edges = np.histogram(data[:, i], bins=n_bins)
            data_discrete[:, i] = np.digitize(data[:, i], bin_edges[1:-1])

        return data_discrete


class DepthCalculator:
    """Calculate the normalized DRR resonance depth metric."""

    def calculate(
        self,
        data: np.ndarray,
        window_size: int,
        sampling_rate: float = 1.0,
        resonance_frequencies: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute the composite normalized Resonance Depth metric.

        The return payload includes the scalar depth, component scores, a compact
        confidence interval, and the target frequency used for scoring.
        """

        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        data = np.asarray(data, dtype=float)
        finite_mask = np.isfinite(data)
        if not np.all(finite_mask):
            logger.warning(
                "DepthCalculator: %d non-finite values filtered from input", np.sum(~finite_mask)
            )
            data = data[finite_mask]

        if len(data) < window_size:
            return self._empty_result()

        window_data = data[-window_size:]
        rolling_std = float(np.std(window_data, ddof=1)) if len(window_data) > 1 else 0.0
        target_frequency = self._select_target_frequency(
            window_data, sampling_rate, resonance_frequencies
        )

        spectral_concentration = self._spectral_concentration(
            window_data, sampling_rate, target_frequency
        )
        temporal_persistence = self._temporal_persistence(
            data, window_size, sampling_rate, target_frequency
        )
        phase_coherence = self._phase_coherence(window_data, sampling_rate, target_frequency)
        amplitude_stability = self._amplitude_stability(window_data)

        components = {
            "spectral_concentration": spectral_concentration,
            "temporal_persistence": temporal_persistence,
            "phase_coherence": phase_coherence,
            "amplitude_stability": amplitude_stability,
        }
        resonance_depth = _clip01(
            0.35 * spectral_concentration
            + 0.25 * temporal_persistence
            + 0.25 * phase_coherence
            + 0.15 * amplitude_stability
        )
        confidence_interval = self._confidence_interval(resonance_depth, len(window_data))

        return {
            "method": "drr_composite_v1",
            "resonance_depth": resonance_depth,
            "components": components,
            "confidence_interval": confidence_interval,
            "target_frequency_hz": target_frequency,
            "legacy_rolling_std": rolling_std,
        }

    def _empty_result(self) -> dict:
        return {
            "method": "drr_composite_v1",
            "resonance_depth": 0.0,
            "components": {
                "spectral_concentration": 0.0,
                "temporal_persistence": 0.0,
                "phase_coherence": 0.0,
                "amplitude_stability": 0.0,
            },
            "confidence_interval": (0.0, 0.0),
            "target_frequency_hz": 0.0,
            "legacy_rolling_std": 0.0,
        }

    def _select_target_frequency(
        self,
        data: np.ndarray,
        sampling_rate: float,
        resonance_frequencies: Optional[np.ndarray],
    ) -> float:
        if resonance_frequencies is not None:
            freqs = np.asarray(resonance_frequencies, dtype=float)
            freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
            if freqs.size:
                return float(freqs[0])

        nperseg = min(256, len(data))
        if nperseg < 8:
            return 0.0
        freqs, psd = welch(data - np.mean(data), fs=sampling_rate, nperseg=nperseg)
        positive = freqs > 0
        if not np.any(positive):
            return 0.0
        positive_freqs = freqs[positive]
        positive_psd = psd[positive]
        return float(positive_freqs[np.argmax(positive_psd)])

    def _spectral_concentration(
        self, data: np.ndarray, sampling_rate: float, target_frequency: float
    ) -> float:
        nperseg = min(256, len(data))
        if nperseg < 8:
            return 0.0
        freqs, psd = welch(data - np.mean(data), fs=sampling_rate, nperseg=nperseg)
        positive = freqs > 0
        total_power = float(np.sum(psd[positive]))
        if total_power <= _EPS:
            return 0.0

        if target_frequency <= 0:
            return _clip01(float(np.max(psd[positive]) / total_power))

        nearest = int(np.argmin(np.abs(freqs - target_frequency)))
        band = np.arange(max(0, nearest - 1), min(len(psd), nearest + 2))
        resonant_power = float(np.sum(psd[band]))
        return _clip01(resonant_power / total_power)

    def _temporal_persistence(
        self, data: np.ndarray, window_size: int, sampling_rate: float, target_frequency: float
    ) -> float:
        if target_frequency <= 0:
            return 0.0

        segment_size = min(window_size, max(32, len(data) // 4))
        if len(data) < segment_size:
            return 0.0

        scores = []
        tolerance = max(sampling_rate / max(segment_size, 1), 0.5)
        for start in range(0, len(data) - segment_size + 1, segment_size):
            segment = data[start : start + segment_size]
            freq = self._select_target_frequency(segment, sampling_rate, None)
            if freq <= 0:
                continue
            scores.append(np.exp(-(((freq - target_frequency) / tolerance) ** 2)))
        if not scores:
            return 0.0
        return _clip01(float(np.mean(scores)))

    def _phase_coherence(
        self, data: np.ndarray, sampling_rate: float, target_frequency: float
    ) -> float:
        if target_frequency <= 0 or len(data) < 4:
            return 0.0
        analytic = hilbert(data - np.mean(data))
        phase = np.unwrap(np.angle(analytic))
        expected = 2 * np.pi * target_frequency * np.arange(len(data)) / sampling_rate
        residual = phase - expected
        coherence = abs(np.mean(np.exp(1j * residual)))
        return _clip01(float(coherence))

    def _amplitude_stability(self, data: np.ndarray) -> float:
        if len(data) < 4:
            return 0.0
        amplitude = np.abs(hilbert(data - np.mean(data)))
        mean_amp = float(np.mean(amplitude))
        if mean_amp <= _EPS:
            return 0.0
        return _clip01(1.0 - float(np.std(amplitude) / mean_amp))

    def _confidence_interval(self, resonance_depth: float, n_samples: int) -> Tuple[float, float]:
        effective_n = max(4, n_samples // 32)
        half_width = 1.96 * np.sqrt(
            max(resonance_depth * (1.0 - resonance_depth), 0.0) / effective_n
        )
        half_width = float(np.clip(half_width, 0.02, 0.25))
        return (_clip01(resonance_depth - half_width), _clip01(resonance_depth + half_width))


class AnomalyDetector:
    """Detect anomalies based on resonance depth."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def detect(self, resonance_depth: float) -> bool:
        return resonance_depth > self.threshold

    def reset_history(self) -> None:
        """Reset detector history.

        The current threshold detector is stateless, so this method is a stable
        compatibility hook for future streaming anomaly detectors.
        """

        return None
