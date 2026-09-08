"""
Fit a daily demand model to a real SKU from the Online Retail II dataset
(reused from the Online Retail Project in this portfolio, see data/README_SOURCE.txt).

The chosen SKU (85048, "15CM CHRISTMAS GLASS BALL 20 LIGHTS") is a genuinely
seasonal item: demand rises sharply in the weeks before Christmas and is near
zero the rest of the year. That gives the environment real non stationary
structure instead of an invented Poisson rate.

Output: data/demand_calibration.json with a day of year seasonal multiplier
curve and a base negative binomial rate, plus data/daily_demand_real.csv with
the raw daily aggregate for reference and plotting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = (
    Path(__file__).resolve().parents[2]
    / "Online Retail Project" / "data" / "cleaned_data_with_customer.csv"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
SKU = "85048"


def load_sku_daily_demand(stock_code: str) -> pd.DataFrame:
    usecols = ["StockCode", "Quantity", "InvoiceDate", "IsCancellation"]
    df = pd.read_csv(RAW_PATH, usecols=usecols, parse_dates=["InvoiceDate"])
    df = df[(df["StockCode"] == stock_code) & (~df["IsCancellation"].astype(bool))]
    df = df[df["Quantity"] > 0]
    daily = (
        df.set_index("InvoiceDate")["Quantity"]
        .resample("D")
        .sum()
        .fillna(0)
        .rename("units_sold")
        .reset_index()
    )
    return daily


def fit_seasonal_curve(daily: pd.DataFrame) -> dict:
    daily = daily.copy()
    daily["doy"] = daily["InvoiceDate"].dt.dayofyear
    # Smooth day of year averages with a 7 day rolling window, wrapped, so a
    # single spike day does not dominate the multiplier for that calendar day.
    by_doy = daily.groupby("doy")["units_sold"].mean().reindex(range(1, 367), fill_value=0.0)
    wrapped = pd.concat([by_doy.iloc[-7:], by_doy, by_doy.iloc[:7]])
    smoothed = wrapped.rolling(7, center=True, min_periods=1).mean().iloc[7:-7]
    baseline = smoothed[smoothed > 0].quantile(0.4) if (smoothed > 0).any() else 1.0
    baseline = max(baseline, 0.5)
    multiplier = (smoothed / baseline).clip(lower=0.05)
    return {
        "day_of_year_multiplier": multiplier.round(4).tolist(),
        "baseline_mean_units_per_day": round(float(baseline), 4),
    }


def fit_dispersion(daily: pd.DataFrame, baseline: float) -> float:
    # Negative binomial dispersion from the variance to mean ratio on days
    # near the baseline level, so the spike days do not distort it.
    near_baseline = daily[
        (daily["units_sold"] > 0) & (daily["units_sold"] < baseline * 4)
    ]["units_sold"]
    if len(near_baseline) < 10:
        return 1.5
    mean = near_baseline.mean()
    var = near_baseline.var()
    if var <= mean:
        return 50.0  # close to Poisson, low overdispersion
    # NB variance = mean + mean^2 / r, solve for r
    r = (mean ** 2) / (var - mean)
    return float(np.clip(r, 0.5, 50.0))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_sku_daily_demand(SKU)
    if daily["units_sold"].sum() == 0:
        raise SystemExit(f"no demand found for SKU {SKU}, check the source file path")

    seasonal = fit_seasonal_curve(daily)
    dispersion = fit_dispersion(daily, seasonal["baseline_mean_units_per_day"])

    calibration = {
        "source": "Online Retail II, StockCode 85048, 15CM CHRISTMAS GLASS BALL 20 LIGHTS",
        "n_days_observed": int(len(daily)),
        "total_units_observed": int(daily["units_sold"].sum()),
        "negative_binomial_dispersion_r": round(dispersion, 3),
        **seasonal,
    }
    with open(OUT_DIR / "demand_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    daily.to_csv(OUT_DIR / "daily_demand_real.csv", index=False)
    print(f"observed days: {calibration['n_days_observed']}")
    print(f"total units: {calibration['total_units_observed']}")
    print(f"baseline mean units/day: {seasonal['baseline_mean_units_per_day']}")
    print(f"peak multiplier: {max(seasonal['day_of_year_multiplier']):.2f}")
    print(f"dispersion r: {calibration['negative_binomial_dispersion_r']}")


if __name__ == "__main__":
    main()
