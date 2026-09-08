"""
A small REINFORCE policy gradient agent, the "if time allows" comparison
against tabular Q-learning. A two layer network maps the continuous
observation directly to action logits, so it does not need the hand chosen
discretization the tabular agent depends on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReinforceAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 2e-4,
                 gamma: float = 0.95, entropy_coef: float = 0.01,
                 baseline_decay: float = 0.95, grad_clip: float = 0.5, seed: int = 0):
        torch.manual_seed(seed)
        self.net = PolicyNet(obs_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions
        self.entropy_coef = entropy_coef
        self.baseline_decay = baseline_decay
        self.grad_clip = grad_clip
        self.position_baseline = None  # set once horizon length is known

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        # the raw observation mixes an unbounded inventory count (0 to 60+)
        # with features already scaled to roughly 0 to 3. Left as is, the
        # network's random init lets inventory dominate every hidden unit's
        # input purely by magnitude, which produced a real failure mode
        # during development: high inventory states got an arbitrarily
        # strong association with the highest order action just from init,
        # and that association fed on itself once training started, since
        # ordering more raises inventory which raises the order probability
        # again. Scaling inventory down to the same rough range as the other
        # features removes that artifact.
        out = obs.copy()
        out[0] = out[0] / 30.0
        return out

    def act(self, obs: np.ndarray, greedy: bool = False):
        x = torch.as_tensor(self._normalize(obs), dtype=torch.float32).unsqueeze(0)
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        if greedy:
            action = int(torch.argmax(probs, dim=-1).item())
            return action, None, None
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()

    def act_greedy(self, obs: np.ndarray) -> int:
        action, _, _ = self.act(obs, greedy=True)
        return action

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.gamma * running
            returns[i] = running
        return torch.as_tensor(returns)

    def train(self, env, n_episodes: int, seed_start: int = 0) -> np.ndarray:
        episode_rewards = np.zeros(n_episodes)
        for ep in range(n_episodes):
            obs = env.reset(seed=seed_start + ep)
            done = False
            log_probs = []
            entropies = []
            rewards = []
            total = 0.0
            while not done:
                action, log_prob, entropy = self.act(obs)
                obs, reward, done, _ = env.step(action)
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(reward)
                total += reward

            returns = self._discounted_returns(rewards)
            # a baseline indexed by position within the episode, not a
            # single scalar. Returns[t] shrinks in magnitude as t approaches
            # the end of the horizon purely because fewer discounted future
            # steps remain, regardless of policy quality. A single episode
            # level or global scalar baseline does not track that shape, so
            # it hands early timesteps systematically large magnitude
            # advantage and late timesteps systematically small or wrong
            # sign advantage, which was the actual cause of the earlier
            # collapse to always ordering the maximum. The environment's
            # calendar is fixed across episodes (same start day of year
            # every time), so timestep t means the same calendar day every
            # episode and an average return at that position is a
            # meaningful, cheap stand in for a learned value function.
            if self.position_baseline is None:
                self.position_baseline = returns.clone()
            advantage = returns - self.position_baseline
            scale = advantage.std()
            if scale > 1e-6:
                advantage = advantage / scale
            self.position_baseline = (
                self.baseline_decay * self.position_baseline
                + (1 - self.baseline_decay) * returns
            )

            policy_loss = -torch.stack([lp * a for lp, a in zip(log_probs, advantage)]).sum()
            entropy_bonus = torch.stack(entropies).sum()
            loss = policy_loss - self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.optimizer.step()
            episode_rewards[ep] = total
        return episode_rewards
