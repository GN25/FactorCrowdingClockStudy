#!/usr/bin/env python3
"""
Factor Crowding Death Clock
===========================
Tracks monthly factor decay for Momentum, Value, Quality, Low-Vol, and Size
from 1993 to today using:
  1) Rolling Information Coefficient (IC) from ranked portfolio buckets
  2) Rolling Sharpe ratio of each factor long-short spread

Output:
  - CSV summary with current IC/Sharpe and estimated years-to-zero
    - Standalone PNG plots for rolling IC, rolling Sharpe, and death-clock bars
    - Additional diagnostics: cumulative spread performance and decay slopes

Run:
  python factor_crowding_death_clock.py
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from pandas_datareader.famafrench import get_available_datasets
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

START_DATE = "1993-01-01"
ROLLING_WINDOW_MONTHS = 36
TREND_WINDOW_YEARS = 10
YEARS_CAP = 40.0

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass
class FactorSpec:
    name: str
    dataset_candidates: list[str]
    dataset_hints: list[str]
    direction: int


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def month_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp(how="end")

    if np.issubdtype(index.dtype, np.datetime64):
        return pd.to_datetime(index)

    s = index.astype(str)
    parsed = pd.to_datetime(s, format="%Y%m", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(s, errors="coerce")
    return pd.DatetimeIndex(parsed)


def pick_dataset(hints: list[str], available: list[str]) -> str:
    norm_hints = [normalize_name(h) for h in hints]

    # Prefer monthly datasets over daily/weekly variants.
    monthly = [d for d in available if "daily" not in d.lower() and "weekly" not in d.lower()]

    for ds in monthly:
        nds = normalize_name(ds)
        if all(h in nds for h in norm_hints):
            return ds

    for ds in available:
        nds = normalize_name(ds)
        if all(h in nds for h in norm_hints):
            return ds

    raise ValueError(f"Could not find dataset matching hints: {hints}")


def pick_dataset_for_factor(spec: FactorSpec, available: list[str]) -> str:
    for candidate in spec.dataset_candidates:
        if candidate in available:
            return candidate
    return pick_dataset(spec.dataset_hints, available)


def extract_bucket_order(columns: list[str]) -> list[str]:
    parsed: list[tuple[float, str]] = []

    for c in columns:
        c_clean = str(c).strip()
        c_norm = c_clean.lower()

        if "lo" in c_norm and "hi" not in c_norm:
            rank = 1.0
        elif "hi" in c_norm:
            rank = 10.0
        else:
            m = re.search(r"(\d+)", c_norm)
            rank = float(m.group(1)) if m else np.nan

        if np.isfinite(rank):
            parsed.append((rank, c_clean))

    if len(parsed) < 5:
        raise ValueError("Could not infer enough ranked bucket columns for IC computation.")

    parsed.sort(key=lambda x: x[0])

    ordered: list[str] = []
    used = set()
    for _, col in parsed:
        if col not in used:
            ordered.append(col)
            used.add(col)

    return ordered[:10]


def clean_table(table: pd.DataFrame, start_date: str) -> pd.DataFrame:
    out = table.copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out.index = month_index(out.index)
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[out.index >= pd.to_datetime(start_date)]
    return out.dropna(how="all")


def factor_from_ranked_buckets(table: pd.DataFrame, direction: int) -> tuple[pd.Series, pd.Series]:
    ordered_cols = extract_bucket_order([str(c) for c in table.columns])
    ranked = table[ordered_cols].dropna(how="all")

    if ranked.shape[1] < 5:
        raise ValueError("Not enough valid ranked buckets after cleaning.")

    ranks = np.arange(1, ranked.shape[1] + 1)

    ic_vals = []
    spread_vals = []
    idx_vals = []

    for idx, row in ranked.iterrows():
        vals = row.values.astype(float)
        mask = np.isfinite(vals)
        if mask.sum() < 5:
            continue

        rho, _ = spearmanr(ranks[mask], vals[mask])
        if np.isfinite(rho):
            ic_vals.append(float(direction) * float(rho))
            idx_vals.append(idx)
            spread_vals.append(float(direction) * (vals[mask][-1] - vals[mask][0]))

    ic = pd.Series(ic_vals, index=pd.DatetimeIndex(idx_vals), name="ic")
    spread = pd.Series(spread_vals, index=pd.DatetimeIndex(idx_vals), name="factor_return")

    return ic.sort_index(), spread.sort_index()


def rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mean = returns.rolling(window).mean()
    vol = returns.rolling(window).std()
    return (mean / vol) * np.sqrt(12)


def years_to_zero(series: pd.Series, lookback_years: int, cap: float) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 24:
        return np.nan, np.nan

    last_date = s.index.max()
    start = last_date - pd.DateOffset(years=lookback_years)
    w = s[s.index >= start]

    if len(w) < 24:
        return np.nan, np.nan

    x = np.arange(len(w), dtype=float)
    y = w.values.astype(float)

    slope_m, _ = np.polyfit(x, y, 1)
    current = float(w.iloc[-1])
    slope_y = float(slope_m * 12.0)

    if slope_y >= 0:
        return np.inf, slope_y

    if current <= 0:
        return 0.0, slope_y

    years = current / (-slope_y)
    return float(min(years, cap)), slope_y


def get_factor_specs() -> list[FactorSpec]:
    return [
        FactorSpec(
            name="Momentum",
            dataset_candidates=["10_Portfolios_Prior_12_2", "25_Portfolios_ME_Prior_12_2"],
            dataset_hints=["10", "prior", "12", "2"],
            direction=1,
        ),
        FactorSpec(
            name="Value",
            dataset_candidates=["Portfolios_Formed_on_BE-ME", "Portfolios_Formed_on_BE-ME_Wout_Div"],
            dataset_hints=["portfolios", "formed", "be", "me"],
            direction=1,
        ),
        FactorSpec(
            name="Quality",
            dataset_candidates=["Portfolios_Formed_on_OP", "Portfolios_Formed_on_OP_Wout_Div"],
            dataset_hints=["portfolios", "formed", "op"],
            direction=1,
        ),
        FactorSpec(
            name="Low-Vol",
            dataset_candidates=["Portfolios_Formed_on_BETA", "25_Portfolios_ME_BETA_5x5"],
            dataset_hints=["portfolios", "formed", "beta"],
            direction=-1,
        ),
        FactorSpec(
            name="Size",
            dataset_candidates=["Portfolios_Formed_on_ME", "Portfolios_Formed_on_ME_Wout_Div"],
            dataset_hints=["portfolios", "formed", "me"],
            direction=-1,
        ),
    ]


def build_death_clock(start_date: str = START_DATE, rolling_window: int = ROLLING_WINDOW_MONTHS) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    available = get_available_datasets()
    specs = get_factor_specs()

    summaries = []
    series_map: dict[str, pd.DataFrame] = {}

    for spec in specs:
        dataset = pick_dataset_for_factor(spec, available)
        payload = web.DataReader(dataset, "famafrench", start=start_date)
        table = clean_table(payload[0], start_date)

        ic, ret = factor_from_ranked_buckets(table, spec.direction)
        aligned = pd.concat([ic.rename("ic"), ret.rename("ret")], axis=1).dropna()

        aligned["rolling_ic"] = aligned["ic"].rolling(rolling_window).mean()
        aligned["rolling_sharpe"] = rolling_sharpe(aligned["ret"], rolling_window)

        ic_years, ic_slope = years_to_zero(aligned["rolling_ic"], TREND_WINDOW_YEARS, YEARS_CAP)
        sr_years, sr_slope = years_to_zero(aligned["rolling_sharpe"], TREND_WINDOW_YEARS, YEARS_CAP)

        finite_estimates = [v for v in [ic_years, sr_years] if np.isfinite(v)]
        combined = float(np.mean(finite_estimates)) if finite_estimates else np.inf

        summaries.append(
            {
                "factor": spec.name,
                "dataset": dataset,
                "current_rolling_ic": float(aligned["rolling_ic"].dropna().iloc[-1]) if not aligned["rolling_ic"].dropna().empty else np.nan,
                "current_rolling_sharpe": float(aligned["rolling_sharpe"].dropna().iloc[-1]) if not aligned["rolling_sharpe"].dropna().empty else np.nan,
                "ic_slope_per_year": ic_slope,
                "sharpe_slope_per_year": sr_slope,
                "ic_years_to_zero": ic_years,
                "sharpe_years_to_zero": sr_years,
                "combined_years_left": combined,
            }
        )

        series_map[spec.name] = aligned

    summary_df = pd.DataFrame(summaries).sort_values("combined_years_left", ascending=True)
    return summary_df, series_map


def save_outputs(summary_df: pd.DataFrame, series_map: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_dashboard = out_dir / "factor_death_clock_dashboard.png"
    if legacy_dashboard.exists():
        legacy_dashboard.unlink()

    csv_path = out_dir / "factor_death_clock_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # 1) Rolling IC (standalone)
    fig_ic, ax_ic = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for factor, df in series_map.items():
        ax_ic.plot(df.index, df["rolling_ic"], label=factor)
    ax_ic.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax_ic.set_title("Rolling IC (36M)")
    ax_ic.set_ylabel("IC")
    ax_ic.legend(ncol=3)
    rolling_ic_path = out_dir / "rolling_ic.png"
    fig_ic.savefig(rolling_ic_path)
    plt.close(fig_ic)

    # 2) Rolling Sharpe (standalone)
    fig_sr, ax_sr = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for factor, df in series_map.items():
        ax_sr.plot(df.index, df["rolling_sharpe"], label=factor)
    ax_sr.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax_sr.set_title("Rolling Sharpe (36M, annualized)")
    ax_sr.set_ylabel("Sharpe")
    ax_sr.legend(ncol=3)
    rolling_sharpe_path = out_dir / "rolling_sharpe.png"
    fig_sr.savefig(rolling_sharpe_path)
    plt.close(fig_sr)

    # 3) Death clock (standalone)
    fig_dc, ax_dc = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bar = summary_df.copy()
    bar["plot_years"] = bar["combined_years_left"].replace(np.inf, YEARS_CAP)
    ax_dc.barh(bar["factor"], bar["plot_years"], color="#2D7DD2")
    ax_dc.set_xlim(0, YEARS_CAP)
    ax_dc.set_title("Factor Crowding Death Clock (years left to zero)")
    ax_dc.set_xlabel(f"Years (capped at {YEARS_CAP:.0f})")

    for y, v in enumerate(bar["combined_years_left"]):
        label = ">40" if np.isinf(v) else f"{v:.1f}"
        ax_dc.text(min(bar["plot_years"].iloc[y] + 0.5, YEARS_CAP - 1.0), y, label, va="center")

    death_clock_path = out_dir / "death_clock_years_left.png"
    fig_dc.savefig(death_clock_path)
    plt.close(fig_dc)

    # 4) Additional plot: cumulative spread performance (index = 100 at start)
    fig_cum, ax_cum = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for factor, df in series_map.items():
        ret = df["ret"].dropna()
        if ret.empty:
            continue
        # Fama-French portfolio returns are in percent units.
        cum = (1 + ret / 100.0).cumprod() * 100.0
        ax_cum.plot(cum.index, cum.values, label=factor)
    ax_cum.set_title("Cumulative Factor Spread Index (Start = 100)")
    ax_cum.set_ylabel("Index Level")
    ax_cum.legend(ncol=3)
    cumulative_path = out_dir / "cumulative_factor_spreads.png"
    fig_cum.savefig(cumulative_path)
    plt.close(fig_cum)

    # 5) Additional plot: IC and Sharpe trend slopes (per year)
    fig_slp, ax_slp = plt.subplots(figsize=(10, 5), constrained_layout=True)
    slope_plot = summary_df[["factor", "ic_slope_per_year", "sharpe_slope_per_year"]].copy()
    x = np.arange(len(slope_plot))
    w = 0.38
    ax_slp.bar(x - w / 2, slope_plot["ic_slope_per_year"], width=w, label="IC slope")
    ax_slp.bar(x + w / 2, slope_plot["sharpe_slope_per_year"], width=w, label="Sharpe slope")
    ax_slp.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax_slp.set_xticks(x)
    ax_slp.set_xticklabels(slope_plot["factor"], rotation=20)
    ax_slp.set_title("Recent Decay Trend Slopes (10Y regression, per year)")
    ax_slp.set_ylabel("Slope")
    ax_slp.legend()
    slopes_path = out_dir / "factor_decay_slopes.png"
    fig_slp.savefig(slopes_path)
    plt.close(fig_slp)

    print("\nSaved outputs:")
    print(f" - {csv_path}")
    print(f" - {rolling_ic_path}")
    print(f" - {rolling_sharpe_path}")
    print(f" - {death_clock_path}")
    print(f" - {cumulative_path}")
    print(f" - {slopes_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"

    print("Building Factor Crowding Death Clock from 1993 to today...")
    summary_df, series_map = build_death_clock(start_date=START_DATE, rolling_window=ROLLING_WINDOW_MONTHS)

    print("\nCurrent standings (lower years means faster decay):")
    display_cols = [
        "factor",
        "current_rolling_ic",
        "current_rolling_sharpe",
        "ic_years_to_zero",
        "sharpe_years_to_zero",
        "combined_years_left",
    ]
    print(summary_df[display_cols].to_string(index=False, justify="left", col_space=12))

    save_outputs(summary_df, series_map, out_dir)


if __name__ == "__main__":
    main()
