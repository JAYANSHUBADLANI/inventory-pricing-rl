"""
A contextual bandit that sees the current state as context but does not
plan ahead: it picks the action with the highest estimated immediate reward
under a linear model per action, updated online with ridge regression,
epsilon greedy for exploration. This is the bridge from the multi armed
bandit work already done elsewhere in the portfolio (experimentation-cost-study)
to a full sequential RL policy: same context-to-action idea, no value
propagation across time steps.
"""

from __future__ import annotations

import numpy as np


class LinearContextualBandit:
    def __init__(self, n_actions: int, n_features: int, ridge_lambda: float = 1.0,
                 epsilon: float = 0.1, seed: int = 0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.ridge_lambda = ridge_lambda
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.A = [ridge_lambda * np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def _theta(self, a: int) -> np.ndarray:
        return np.linalg.solve(self.A[a], self.b[a])

    def act(self, context: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        scores = np.array([context @ self._theta(a) for a in range(self.n_actions)])
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

    def train(self, env, n_episodes: int, seed_start: int = 0) -> np.ndarray:
        episode_rewards = np.zeros(n_episodes)
        for ep in range(n_episodes):
            obs = env.reset(seed=seed_start + ep)
            done = False
            total = 0.0
            while not done:
                action = self.act(obs)
                next_obs, reward, done, _ = env.step(action)
                self.update(obs, action, reward)
                obs = next_obs
                total += reward
            episode_rewards[ep] = total
        return episode_rewards

    def act_greedy(self, obs: np.ndarray) -> int:
        return self.act(obs, greedy=True)
