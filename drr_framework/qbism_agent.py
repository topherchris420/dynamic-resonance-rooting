import numpy as np

class QBistAgent:
    """
    QBist Agent for Dynamic Resonance Rooting.
    Maintains subjective beliefs about resonance outcomes
    and updates them via Bayesian coherence rules.
    """

    def __init__(self, prior: float = 0.5, learning_rate: float = 0.15):
        """
        Args:
            prior (float): Initial belief that a signal is meaningfully rooted.
            learning_rate (float): Controls belief update strength.
        """
        self.belief = np.clip(prior, 0.01, 0.99)
        self.learning_rate = learning_rate
        self.history = []

    def update_belief(self, resonance_depth: float):
        """
        Update belief based on experienced resonance depth.
        This is NOT an objective truth update, but belief revision.
        """
        likelihood = self._likelihood(resonance_depth)
        posterior = self.belief + self.learning_rate * (likelihood - self.belief)

        self.belief = np.clip(posterior, 0.01, 0.99)
        self.history.append(self.belief)

        return self.belief

    def _likelihood(self, resonance_depth: float) -> float:
        """
        Maps resonance depth to subjective likelihood.
        This replaces 'collapse' with Bayesian coherence.
        """
        return 1 / (1 + np.exp(-resonance_depth))

    def reset(self):
        self.history = []
