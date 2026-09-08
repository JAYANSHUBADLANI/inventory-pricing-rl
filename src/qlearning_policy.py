"""
Tabular Q-learning over a discretized state space. Inventory position and
days-to-arrival are binned, season and regime-recency are already coarse
enough (season multiplier bucketed into quintiles) to keep the table a
manageable size while still letting the agent plan for lead time and season,
which a contextual bandit cannot do.
"""

from __future__ import annotations

import numpy as np


class TabularQLearning:
    def __init__(self, n_actions: int, inv_bins: int = 12, season_bins: int = 8,
                 lead_bins: int = 4, alpha: float = 0.15, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_episodes: int = 300, max_inventory: float = 60.0,
                 seed: int = 0):
        self.n_actions = n_actions
        self.inv_bins = inv_bins
        self.season_bins = season_bins
        self.lead_bins = lead_bins
        self.max_inventory = max_inventory
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.rng = np.random.default_rng(seed)
        table_shape = (inv_bins + 1, lead_bins + 1, season_bins, n_actions)
        self.Q = np.zeros(table_shape)

    def _discretize(self, obs: np.ndarray) -> tuple[int, int, int]:
        # obs season feature is already log1p(raw multiplier) from the
        # environment, so a linear bucketing here still keeps resolution at
        # the high end instead of one bucket swallowing the entire spike.
        inventory, days_to_arrival_norm, season_log, _day_frac, _since_regime = obs
        inv_idx = int(np.clip(inventory / self.max_inventory * self.inv_bins, 0, self.inv_bins))
        lead_idx = int(np.clip(days_to_arrival_norm * self.lead_bins, 0, self.lead_bins))
        season_idx = int(np.clip(season_log * self.season_bins / 3.5, 0, self.season_bins - 1))
        return inv_idx, lead_idx, season_idx

    def _epsilon(self, episode: int) -> float:
        frac = min(episode / self.epsilon_decay_episodes, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: np.ndarray, episode: int | None = None, greedy: bool = False) -> int:
        state = self._discretize(obs)
        if not greedy:
            eps = self._epsilon(episode if episode is not None else self.epsilon_decay_episodes)
            if self.rng.random() < eps:
                return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def act_greedy(self, obs: np.ndarray) -> int:
        return self.act(obs, greedy=True)

    def update(self, obs, action, reward, next_obs, done):
        s = self._discretize(obs)
        s_next = self._discretize(next_obs)
        target = reward + (0.0 if done else self.gamma * np.max(self.Q[s_next]))
        td_error = target - self.Q[s][action]
        self.Q[s][action] += self.alpha * td_error

    def train(self, env, n_episodes: int, seed_start: int = 0) -> np.ndarray:
        episode_rewards = np.zeros(n_episodes)
        for ep in range(n_episodes):
            obs = env.reset(seed=seed_start + ep)
            done = False
            total = 0.0
            while not done:
                action = self.act(obs, episode=ep)
                next_obs, reward, done, _ = env.step(action)
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total += reward
            episode_rewards[ep] = total
        return episode_rewards
