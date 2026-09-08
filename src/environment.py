"""
A periodic review inventory environment with lead time, holding cost, lost
sale penalty, and demand calibrated to a real seasonal SKU. A regime shift
partway through the horizon is a permanent step change in the baseline
demand level, meant to represent something like a supply disruption or a
durable shift in popularity, not another seasonal cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class EnvConfig:
    horizon_days: int = 365
    lead_time: int = 3
    holding_cost_per_unit: float = 0.05
    lost_sale_penalty_per_unit: float = 1.20
    order_fixed_cost: float = 2.0
    max_order: int = 40
    order_step: int = 4  # discrete action granularity
    starting_inventory: int = 15
    regime_shift_day: int | None = None  # set at reset time if enabled
    regime_shift_multiplier: float = 1.6
    demand_multiplier: float = 1.0  # used by robustness tests
    lead_time_override: int | None = None  # used by robustness tests
    seed: int = 0


class InventoryEnv:
    """
    Action: an index into the discrete order quantity grid
        [0, order_step, 2*order_step, ..., max_order].
    Observation: (inventory_position, days_until_next_arrival_norm,
                  season_multiplier, day_frac_of_year, days_since_regime_norm)
    Reward: negative of (holding cost + lost sale penalty + fixed order cost)
    """

    def __init__(self, config: EnvConfig, calibration_path: Path | None = None):
        self.cfg = config
        cal_path = calibration_path or (DATA_DIR / "demand_calibration.json")
        with open(cal_path) as f:
            self.calibration = json.load(f)
        self.doy_multiplier = np.array(self.calibration["day_of_year_multiplier"])
        self.base_rate = self.calibration["baseline_mean_units_per_day"]
        self.dispersion_r = self.calibration["negative_binomial_dispersion_r"]
        self.action_grid = np.arange(0, self.cfg.max_order + 1, self.cfg.order_step)
        self.n_actions = len(self.action_grid)
        self.rng = np.random.default_rng(self.cfg.seed)
        self._start_doy = self.rng.integers(1, 366)
        self.reset()

    def _lead_time(self) -> int:
        return self.cfg.lead_time_override or self.cfg.lead_time

    def reset(self, seed: int | None = None, start_day_of_year: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.start_doy = start_day_of_year if start_day_of_year is not None else self._start_doy
        self.inventory = float(self.cfg.starting_inventory)
        L = self._lead_time()
        self.pipeline = [0.0] * L  # units arriving in 1, 2, ..., L days
        self.done = False
        return self._obs()

    def _doy(self) -> int:
        return ((self.start_doy - 1 + self.t) % 365) + 1

    def _season_multiplier(self) -> float:
        return float(self.doy_multiplier[self._doy() - 1])

    def _demand_rate(self) -> float:
        mult = self._season_multiplier() * self.cfg.demand_multiplier
        if self.cfg.regime_shift_day is not None and self.t >= self.cfg.regime_shift_day:
            mult *= self.cfg.regime_shift_multiplier
        return max(self.base_rate * mult, 1e-6)

    def _draw_demand(self) -> int:
        rate = self._demand_rate()
        r = self.dispersion_r
        p = r / (r + rate)
        return int(self.rng.negative_binomial(r, p))

    def _obs(self) -> np.ndarray:
        L = self._lead_time()
        days_to_arrival = next((i + 1 for i, q in enumerate(self.pipeline) if q > 0), L + 1)
        days_to_arrival_norm = days_to_arrival / (L + 1)
        # log1p compresses the season multiplier's wide dynamic range (0.05
        # to roughly 25x for this SKU) so a tabular agent's bucketing and a
        # linear model's regularization both keep resolution at the extreme
        # high end, where the christmas spike actually lives, instead of it
        # collapsing into one saturated bucket or being swamped by scale.
        season_raw = self._season_multiplier()
        season_log = np.log1p(season_raw)
        day_frac = self._doy() / 365.0
        if self.cfg.regime_shift_day is not None:
            since_regime = max(0, self.t - self.cfg.regime_shift_day)
            since_regime_norm = min(since_regime / 30.0, 1.0)
        else:
            since_regime_norm = 0.0
        return np.array(
            [self.inventory, days_to_arrival_norm, season_log, day_frac, since_regime_norm],
            dtype=np.float32,
        )

    def step(self, action_idx: int):
        if self.done:
            raise RuntimeError("step called after episode ended, call reset first")
        order_qty = float(self.action_grid[action_idx])
        L = self._lead_time()

        arriving = self.pipeline[0] if self.pipeline else 0.0
        self.inventory += arriving
        if self.pipeline:
            self.pipeline = self.pipeline[1:] + [0.0]
        if L > 0:
            self.pipeline[-1] += order_qty
        else:
            self.inventory += order_qty

        demand = self._draw_demand()
        sold = min(demand, self.inventory)
        lost = demand - sold
        self.inventory -= sold

        holding = self.inventory * self.cfg.holding_cost_per_unit
        stockout = lost * self.cfg.lost_sale_penalty_per_unit
        fixed = self.cfg.order_fixed_cost if order_qty > 0 else 0.0
        cost = holding + stockout + fixed
        reward = -cost

        self.t += 1
        self.done = self.t >= self.cfg.horizon_days
        info = {
            "demand": demand,
            "sold": sold,
            "lost": lost,
            "holding_cost": holding,
            "stockout_cost": stockout,
            "fixed_cost": fixed,
            "order_qty": order_qty,
            "inventory_after": self.inventory,
        }
        return self._obs(), reward, self.done, info

    @property
    def observation_dim(self) -> int:
        return 5
