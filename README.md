# Factor Crowding Death Clock Study

A research utility that estimates how quickly classic equity factors are losing predictive power, using Fama-French portfolio data from 1993 to today.

The project converts factor crowding into a single practical question:

**How many years are left before each factor's edge decays to zero, based on recent trend dynamics?**

## Why This Is Useful for Quants

- Turns abstract crowding concerns into a comparable cross-factor horizon (`years left`)
- Uses two independent health signals per factor (cross-sectional signal + realized spread quality)
- Produces visuals and a tabular summary you can drop into research notes, PM updates, or risk committee decks

## Factor Universe

The script tracks five canonical factors:

- Momentum
- Value
- Quality
- Low-Vol
- Size

Data is sourced via `pandas_datareader` from Kenneth French data library datasets.

## Methodology

### 1) Monthly Factor Construction from Ranked Buckets

For each factor dataset:

- Infer ordered ranked portfolio buckets (for example, low to high exposure portfolios)
- Compute monthly **Spearman rank IC** between bucket rank and bucket return
- Compute monthly **long-short spread return** = top bucket minus bottom bucket
- Apply sign conventions so positive values always represent favorable factor performance

### 2) Rolling Signal Quality

On a rolling 36-month window:

- Rolling IC: 36M moving average of monthly IC
- Rolling Sharpe: annualized Sharpe of monthly long-short spread returns

### 3) Death Clock Estimation

For each rolling series (IC and Sharpe):

- Fit a linear trend over the most recent 10 years
- Convert monthly slope to annual slope
- Estimate years-to-zero from current value and negative trend slope
- Cap displayed estimate at 40 years

Combined estimate:

- `combined_years_left` is the mean of finite IC and Sharpe years-to-zero estimates

## Mathematical Summary

Let $s_t$ be a rolling metric series (IC or Sharpe), observed over the lookback window.

Linear trend:

$$
 s_t = a + b t
$$

with annualized slope $b_{yr} = 12b$.

If $b_{yr} < 0$ and current value $s_{now} > 0$:

$$
\text{years-to-zero} = \frac{s_{now}}{-b_{yr}}
$$

If $b_{yr} \ge 0$, estimate is treated as infinite (no current decay to zero under this trend).

## Repository Structure

- `factor_crowding_death_clock.py`: end-to-end pipeline (download, compute, plot, export)
- `outputs/`: generated charts and CSV summary

## Quick Start

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scipy matplotlib pandas_datareader
```

### 2) Run the study

```bash
python factor_crowding_death_clock.py
```

### 3) Inspect outputs

Primary artifacts written to `outputs/`:

- `factor_death_clock_summary.csv`
- `rolling_ic.png`
- `rolling_sharpe.png`
- `death_clock_years_left.png`
- `cumulative_factor_spreads.png`
- `factor_decay_slopes.png`

## Output Interpretation

### Core table fields

- `current_rolling_ic`: current 36M average IC level
- `current_rolling_sharpe`: current 36M annualized Sharpe of factor spread
- `ic_slope_per_year`: 10Y IC trend slope (annualized)
- `sharpe_slope_per_year`: 10Y Sharpe trend slope (annualized)
- `ic_years_to_zero`: projected IC horizon to zero
- `sharpe_years_to_zero`: projected Sharpe horizon to zero
- `combined_years_left`: aggregated horizon estimate

### Practical read

- Lower `combined_years_left` suggests faster potential edge decay
- Negative slopes with still-positive levels imply active decay regime
- Infinite estimate means trend is flat/up, not currently converging to zero under this linear model

## Research Notes and Caveats

- This is a **trend-extrapolation diagnostic**, not a structural forecasting model
- Estimates are sensitive to lookback choices (36M roll, 10Y trend) and recent regime shifts
- Fama-French portfolio construction updates or dataset substitutions can shift levels and trend estimates
- Linear extrapolation may understate convex breakdowns or overstate persistence around turning points

## Suggested Extensions

- Add confidence intervals via block bootstrap on rolling metrics
- Replace linear trend with robust trend estimators (Theil-Sen, Huber)
- Add regional factor sets (ex-US, Europe, EM)
- Segment by market regime (rates, inflation, vol regimes)
- Add turnover/proxy capacity overlays for crowding stress scoring

## License

No license file is currently included. Add one if you plan to distribute or open-source this project.
