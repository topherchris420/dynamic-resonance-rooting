import numpy as np

from drr_framework.qbism_agent import QBistAgent


def test_likelihood_is_centered_and_monotonic():
    agent = QBistAgent()

    assert agent._likelihood(0.0) < 0.5
    assert np.isclose(agent._likelihood(0.5), 0.5)
    assert agent._likelihood(1.0) > 0.5

    depths = np.linspace(0.0, 1.0, 11)
    likelihoods = [agent._likelihood(d) for d in depths]
    assert all(a < b for a, b in zip(likelihoods, likelihoods[1:]))


def test_belief_moves_toward_evidence():
    agent = QBistAgent(prior=0.5, learning_rate=0.2)
    for _ in range(20):
        agent.update_belief(1.0)
    assert agent.belief > 0.65

    agent = QBistAgent(prior=0.5, learning_rate=0.2)
    for _ in range(20):
        agent.update_belief(0.0)
    assert agent.belief < 0.35


def test_reset_restores_prior_and_clears_history():
    agent = QBistAgent(prior=0.4, learning_rate=0.3)
    agent.update_belief(1.0)
    agent.update_belief(1.0)
    assert agent.belief != agent.prior
    assert len(agent.history) == 2

    agent.reset()

    assert agent.belief == agent.prior
    assert agent.history == []
