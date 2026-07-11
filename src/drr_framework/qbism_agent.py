import numpy as np


class QBistAgent:
    """
    QBist Agent for Dynamic Resonance Rooting.
    Maintains subjective beliefs about resonance outcomes
    and updates them via Bayesian coherence rules.
    """

    # Logistic slope chosen so a resonance depth of 0 maps to a likelihood of
    # 0.1, a depth of 0.5 stays neutral at 0.5, and a depth of 1 maps to 0.9.
    _LIKELIHOOD_SLOPE = 2.0 * float(np.log(9.0))

    def __init__(self, prior: float = 0.5, learning_rate: float = 0.15):
        """
        Args:
            prior (float): Initial belief that a signal is meaningfully rooted.
            learning_rate (float): Controls belief update strength.
        """
        self.prior: float = float(np.clip(prior, 0.01, 0.99))
        self.belief: float = self.prior
        self.learning_rate = learning_rate
        self.history: list[float] = []

    def update_belief(self, resonance_depth: float) -> float:
        """
        Update belief based on experienced resonance depth.
        This is NOT an objective truth update, but belief revision.
        """
        likelihood = self._likelihood(resonance_depth)
        posterior = self.belief + self.learning_rate * (likelihood - self.belief)

        self.belief = float(np.clip(posterior, 0.01, 0.99))
        self.history.append(self.belief)

        return self.belief

    def _likelihood(self, resonance_depth: float) -> float:
        """
        Maps resonance depth to subjective likelihood.
        This replaces 'collapse' with Bayesian coherence.

        The logistic is centered at a depth of 0.5 so shallow resonances pull
        the belief down while deep resonances pull it up.
        """
        depth = float(np.clip(resonance_depth, 0.0, 1.0))
        return float(1.0 / (1.0 + np.exp(-self._LIKELIHOOD_SLOPE * (depth - 0.5))))

    def reset(self) -> None:
        """Restore the initial prior belief and clear the update history."""
        self.belief = self.prior
        self.history = []
