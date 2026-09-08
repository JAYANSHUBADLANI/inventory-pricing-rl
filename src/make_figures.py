import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def smooth(x, window=50):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training_curves():
    with open(RESULTS_DIR / "training_curves.json") as f:
        curves = json.load(f)
    plt.figure(figsize=(8, 5))
    for name, rewards in curves.items():
        rewards = np.array(rewards)
        plt.plot(smooth(rewards), label=name)
    plt.xlabel("training episode")
    plt.ylabel("episode total reward, smoothed")
    plt.title("Training curves, contextual bandit vs tabular Q-learning vs REINFORCE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_curves.png", dpi=150)
    plt.close()


def plot_in_sample_comparison():
    with open(RESULTS_DIR / "in_sample_results.json") as f:
        results = json.load(f)["in_sample"]
    names = list(results.keys())
    means = [results[n]["mean_reward"] for n in names]
    los = [results[n]["mean_reward"] - results[n]["ci_95_low"] for n in names]
    his = [results[n]["ci_95_high"] - results[n]["mean_reward"] for n in names]
    plt.figure(figsize=(7, 5))
    plt.bar(names, means, yerr=[los, his], capsize=5)
    plt.ylabel("mean episode reward, held out evaluation")
    plt.title("In sample comparison, 300 held out episodes, 95 percent CI")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "in_sample_comparison.png", dpi=150)
    plt.close()


def plot_robustness():
    with open(RESULTS_DIR / "robustness_results.json") as f:
        report = json.load(f)
    conditions = list(report.keys())
    policies = list(report[conditions[0]].keys())
    x = np.arange(len(conditions))
    width = 0.2
    plt.figure(figsize=(12, 6))
    for i, pol in enumerate(policies):
        means = [report[c][pol]["mean_reward"] for c in conditions]
        plt.bar(x + i * width, means, width, label=pol)
    plt.xticks(x + width * 1.5, conditions, rotation=30, ha="right")
    plt.ylabel("mean episode reward")
    plt.title("Robustness across out of sample conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "robustness_comparison.png", dpi=150)
    plt.close()


def plot_seasonal_demand():
    with open(ROOT / "data" / "demand_calibration.json") as f:
        cal = json.load(f)
    mult = cal["day_of_year_multiplier"]
    plt.figure(figsize=(8, 4))
    plt.plot(mult)
    plt.xlabel("day of year")
    plt.ylabel("demand multiplier vs baseline")
    plt.title("Calibrated seasonal demand multiplier, SKU 85048")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "seasonal_demand_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_training_curves()
    plot_in_sample_comparison()
    plot_robustness()
    plot_seasonal_demand()
    print("figures written to", FIG_DIR)
