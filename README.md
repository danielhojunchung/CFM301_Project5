# CFM 301 Project 5: Comparing Mutual Funds — Are the Fees Justified?

**Research question:** Among US Large Growth mutual funds, does past performance predict
future alpha after accounting for momentum?

## Requirements

| Item | Version |
|---|---|
| Python | 3.12.4 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |
| scipy | 1.16.3 (analysis only) |
| matplotlib | 3.10.6 (analysis only) |

```bash
pip install pandas==2.2.3 numpy==1.26.4 scipy==1.16.3 matplotlib==3.10.6
```

`analysis.py` was also tested with Python 3.13.9, pandas 2.3.3, numpy 2.3.5.

No WRDS login, internet connection, or manual step is needed. All paths are relative to
this folder, and the scripts read the zip files in `data/` directly (no unzipping).

## How to run

Run from this folder, in this order:

| # | File | What it does | Output |
|---|---|---|---|
| 1 | `clean_data.py` | Cleans fund returns, factors, benchmarks, flows and TNA | `output/clean/` |
| 2 | `analysis.py` | Factor regressions, gross/net alpha, index benchmark, fees, persistence, flows, sensitivity, figures | `output/results/` |

```bash
python clean_data.py
python analysis.py
```

## Data (`data/`)

| File | Source | Contents |
|---|---|---|
| `Yahoo_Fund_Data.zip` | Yahoo Finance via `yfinance` | Monthly adjusted-close returns, expense ratios and metadata for 100 active US Large Growth funds and 3 Vanguard index funds (VFIAX, VSMAX, VVIAX). See `Data_Notes.txt` inside |
| `SEC_Partial_Data.zip` | SEC EDGAR N-PORT filings | Monthly sales/redemptions/net flows and quarter-end total net assets (TNA), 2020 onward, for a subset of funds |
| `F-F_Research_Data_Factors_CSV.zip` | Kenneth French Data Library | Monthly Mkt-RF, SMB, HML, RF (percent) |
| `F-F_Momentum_Factor_CSV.zip` | Kenneth French Data Library | Monthly momentum factor (percent) |

## Cleaned output (`output/clean/`)

| File | Contents |
|---|---|
| `fund_panel_clean.csv` | Active funds, one row per fund-month: net and gross returns, excess returns, factors, flags |
| `benchmark_returns_clean.csv` | VFIAX, VSMAX, VVIAX returns on common months (2000-12 to 2025-12) and RF, for the constrained index-fund benchmark |
| `factors_clean.csv` | Monthly MKT_RF, SMB, HML, RF, Mom as decimals |
| `fund_info_clean.csv` | One row per fund: fees, sample dates, number of months, flags |
| `fund_flows_monthly_clean.csv` | SEC monthly net flows, scaled by the most recent prior quarter-end TNA |
| `fund_flows_quarterly_clean.csv` | Quarterly flows, flow(t) = TNA(t) − TNA(t−1) × (1 + r(t)) |
| `return_validation.csv` | Yahoo returns compared with returns reported to the SEC |
| `outlier_months.csv` | Months with the most extreme fund returns, with the market return |
| `cleaning_log.csv` | Observations and funds affected by every cleaning step |

### Key columns in `fund_panel_clean.csv`

- `Return_Net`: reported (after-fee) monthly total return
- `Return_Gross`: `Return_Net` + `Monthly_Expense_Ratio` (annual expense ratio ÷ 12)
- `Exret_Net`, `Exret_Gross`: returns minus the risk-free rate (use these in factor regressions)
- `Zero_Fee_Flag`: fund reports a 0% expense ratio (Fidelity Series funds)
- `Outlier_Flag`: excess return outside the pooled 1st/99th percentile (flag only, not changed)
- `Eligible_Persistence`: fund has at least 120 months of returns

## Analysis output (`output/results/`)

`summary.txt` has the headline numbers for the report. Alphas are annualized (monthly
alpha × 12), tracking error is the regression residual standard deviation × √12, and
the information ratio is alpha ÷ tracking error. A fund's fee counts as **justified** if
its net-of-fee Carhart 4-factor alpha is ≥ 0.

| File | Requirement | Contents |
|---|---|---|
| `factor_regressions.csv` | 5, 6 | Alpha, t-stat, tracking error, IR, R² and loadings for every fund × CAPM/FF3/Carhart × net/gross |
| `alpha_change_summary.csv`, `alpha_change_by_fund.csv` | 5 | How alpha and loadings change as SMB/HML and momentum are added |
| `gross_net_alpha.csv` | 6, 7, 11 | Fee, gross and net Carhart alpha, index-benchmark alpha, Berk–Green class, verdict |
| `aggregate_portfolio.csv` | 6 | Equal-weighted portfolio of all funds (Fama–French 2010 style) |
| `index_benchmark.csv` | 7 | Constrained regression on VFIAX/VSMAX/VVIAX (w ≥ 0, Σw = 1), replicating fee vs fund fee |
| `fee_vs_alpha.csv`, `fee_terciles.csv` | 8 | Cross-sectional alpha-on-fee regressions (Carhart 1997) and fee-tercile averages |
| `fee_distribution.csv`, `fee_universe_summary.csv` | 8 | Each fund's fee percentile among 858 active Large Growth share classes (Yahoo screening universe) |
| `persistence_split_half.csv`, `persistence_fund_alphas.csv` | 9 | Split-half persistence: own-history halves (83 funds) and calendar halves 2001-01–2013-06 / 2013-07–2025-12 |
| `momentum_portfolios.csv`, `momentum_portfolio_returns.csv` | 9 | Funds sorted monthly into terciles on past 12-month return; CAPM/FF3/Carhart alphas |
| `flow_performance.csv` | 9 | Flows vs past 1-year return (quarterly TNA-implied flows and monthly SEC net flows) |
| `berk_green_summary.csv` | 9 | Funds with gross alpha ≤ 0 / gross > 0 > net / net ≥ 0 |
| `sensitivity_verdicts.csv`, `sensitivity_summary.csv`, `rolling_alpha_summary.csv` | 10 | Verdicts under rolling 36-month alphas, calendar sub-periods, 1%/99% winsorization, post-2000 sample, and CAPM/FF3/index-fund benchmarks |
| `fig1`–`fig6` `.png` | — | Gross vs net alpha, fee vs alpha, persistence, momentum portfolios, flows, fee distribution |

## Results

All figures are annualized and from `output/results/summary.txt`. The sample is 96 funds;
"significant" means |t| > 1.96.

**Short answer to the research question:** fund alpha does not persist over long horizons
under any model. Funds that did well over the past 12 months keep doing well the next
month, and momentum explains part of that (about 40% of the CAPM spread), not all of it.
The spread that remains comes mostly from past losers continuing to underperform, not from
winners delivering alpha. Fees are the clearest predictor of investor (net) alpha: the
higher the fee, the lower the net alpha.

### Factor models (Req. 5)

| Model | Mean net alpha | Median | Funds with alpha > 0 | Sig. > 0 | Sig. < 0 | Mean R² |
|---|---|---|---|---|---|---|
| CAPM | −0.17% | −0.16% | 44% | 1 | 7 | 0.84 |
| FF3 | −0.03% | −0.02% | 48% | 4 | 4 | 0.89 |
| Carhart | −0.23% | −0.10% | 45% | 2 | 5 | 0.90 |

Average Carhart loadings: market 1.06, SMB 0.03, HML −0.30, momentum 0.03. Growth funds
load negatively on value, and value stocks had low returns over the sample, so adding HML
*raises* alpha (+0.14% on average). Adding momentum lowers it again (−0.20% on average).

### Gross vs net alpha and the index-fund benchmark (Req. 6–7)

| | Value |
|---|---|
| Mean expense ratio | 0.92% |
| Mean gross Carhart alpha | +0.69% |
| Mean net Carhart alpha | −0.23% |
| Funds with fee justified (net alpha ≥ 0) | 43 of 96 (45%) |
| … with net alpha significantly > 0 / < 0 | 2 / 5 |
| Equal-weighted portfolio of all funds, gross / net | +0.96% (t = 1.74) / −0.00% (t = −0.01) |
| Berk–Green classes | 27% gross alpha ≤ 0; 28% gross > 0 > net; 45% net ≥ 0 |

The equal-weighted portfolio shows the Fama–French (2010) pattern: managers as a group earn
about their fee back, and investors are left with zero alpha.

Constrained index-fund benchmark (weights ≥ 0, sum to 1): average weights are VFIAX 0.90,
VSMAX 0.10 and VVIAX 0.00. Mean net alpha against it is +0.68% (62% of funds ≥ 0; mean
R² 0.83). The replicating mix costs about **0.04% a year vs 0.92%** for the average fund,
a saving of 0.88% a year. The index-fund alpha looks better than the Carhart alpha because
none of the three index funds tilts toward growth (the value weight is stuck at zero), so
it should be read as an upper bound.

### Fees vs performance (Req. 8; zero-fee funds excluded)

| | Slope on fee | t |
|---|---|---|
| Gross alpha | −0.57 | −1.18 (test of slope = 1: t = −3.25) |
| Net alpha | −1.57 | −3.25 |

| Fee tercile | Mean fee | Gross alpha | Net alpha | Justified |
|---|---|---|---|---|
| Low | 0.59% | +0.95% | +0.36% | 53% |
| Mid | 0.89% | +0.66% | −0.22% | 48% |
| High | 1.35% | +0.34% | −1.01% | 29% |

Higher-fee funds do not earn more gross alpha, so investors pay the full fee and get
nothing extra for it. The sample's median fee (0.90%) equals the median of 858 active Large
Growth share classes in the Yahoo screening universe (25th–75th percentile 0.68%–1.21%).
The median for 39 index share classes is 0.29%.

### Persistence and flows (Req. 9)

Split-half test (net alpha in the second half regressed on alpha in the first half):

| Split | Model | N | Slope (t) | Rank corr. | Top − bottom tercile (t) |
|---|---|---|---|---|---|
| Each fund's history halved | CAPM | 83 | −0.03 (−0.29) | −0.02 | −0.05% (−0.08) |
| Each fund's history halved | Carhart | 83 | +0.09 (0.82) | −0.02 | +0.22% (0.41) |
| Calendar 2001-01–2013-06 vs after | CAPM | 63 | +0.09 (1.13) | +0.05 | +0.21% (0.37) |
| Calendar 2001-01–2013-06 vs after | Carhart | 63 | +0.13 (1.26) | +0.15 | +0.73% (1.52) |

Each month, funds were sorted into thirds on their past 12-month return and held for one
month (2000-01 to 2025-12). Annualized net alpha:

| Model | Past losers | Past winners | Winners − losers (t) |
|---|---|---|---|
| CAPM | −2.6% | +1.1% | +3.66% (2.64) |
| Carhart | −1.3% | +0.9% | +2.23% (2.14), momentum loading 0.26 (t = 13.8) |

Flows vs the past 1-year return (7 funds, 2020–2025): quarterly flows implied by TNA have a
slope of 0.018 (t = 1.68), or 0.028 (t = 2.61) within funds. Monthly SEC net flows have a
slope of 0.012 (t = 2.05). Money follows past returns, as Berk and Green (2004) predict, but
the flow sample is very small.

### Sensitivity of the verdicts (Req. 10)

| Alternative | Funds | Justified | Mean net alpha | Verdicts changed |
|---|---|---|---|---|
| **Base: full sample, Carhart** | 96 | 43 (45%) | −0.23% | — |
| Rolling 36-month alphas (mean) | 96 | 40 (42%) | −0.40% | 19 |
| Sub-period 2001-01–2013-06 | 63 | 16 (25%) | −1.27% | 19 |
| Sub-period 2013-07–2025-12 | 63 | 19 (30%) | −0.34% | 18 |
| Winsorized 1%/99% (−13.4% / 12.9%) | 96 | 53 (55%) | +0.09% | 10 |
| Post-2000-12 sample | 96 | 36 (38%) | −0.40% | 11 |
| CAPM benchmark | 96 | 42 (44%) | −0.17% | 17 |
| FF3 benchmark | 96 | 46 (48%) | −0.03% | 7 |
| Index-fund benchmark | 96 | 60 (62%) | +0.68% | 27 |

43 of 96 funds keep the same verdict under every alternative: 21 justified and 22 not
justified. Funds with short histories face fewer alternatives, so they pass more easily.
Also, two of the 21 (FAOFX, FVWSX) charge no fee. `sensitivity_verdicts.csv` has the
fund-by-fund table.

## Cleaning decisions

1. **Returns.** Yahoo adjusted closing prices are used as total returns (distributions
   reinvested). Checked against SEC-reported returns: median absolute difference about
   0.002% per month.
2. **Share classes.** One share class per SEC fund series. No duplicates found.
3. **Data errors.** Monthly returns below −50% or above +100% are treated as data errors.
   Two found (OTPSX and LGPSX, −75% in Oct 2024, an unadjusted share split).
4. **Index funds removed.** IRLAX, SWLGX, OTPSX and LGPSX are index funds labelled as active.
5. **Missing values.** Missing returns are dropped, never filled or interpolated.
6. **No winsorization.** Extreme months are real market events (1987, 1998, 2000–02, 2008,
   2020) that investors experienced. They are flagged so they can be examined, not altered.
7. **Zero-fee funds.** FVWSX and FAOFX are kept but flagged: they are sold only to other
   Fidelity funds, so their gross and net alphas are identical.
8. **Expense ratios.** Converted to monthly by dividing the annual ratio by 12.

The final sample is 96 active funds and 25,022 fund-months (1985-02 to 2025-12); 83 funds
have at least 120 months. Exact counts for every step are in `cleaning_log.csv`.

## Data limitations

- **Survivorship bias:** only funds that still exist are included; closed and merged funds
  are missing.
- **Fees:** expense ratios are current values only; historical fees are assumed unchanged.
- **Turnover:** not available (the `Annual_Turnover` field is empty for every fund).
- **Flows:** SEC data covers 2020 onward for 6–7 funds only. TNA is quarterly. Flows are
  reported per share class while TNA covers the whole fund, so flow as a percentage of TNA is
  approximate.
- **Index-fund benchmark:** Vanguard benchmark returns start in 2000-12.
