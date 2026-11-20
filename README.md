# CFM 301 Project 5: Comparing Mutual Funds — Are the Fees Justified?

**Research question:** Among US Large Growth mutual funds, does past performance predict
future alpha after accounting for momentum?

## Requirements

| Item | Version |
|---|---|
| Python | 3.12.4 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |

```bash
pip install pandas==2.2.3 numpy==1.26.4
```

No WRDS login, internet connection, or manual step is needed. All paths are relative to
this folder, and the scripts read the zip files in `data/` directly (no unzipping).

## How to run

Run from this folder, in this order:

| # | File | What it does | Output |
|---|---|---|---|
| 1 | `clean_data.py` | Cleans fund returns, factors, benchmarks, flows and TNA | `output/clean/` |

```bash
python clean_data.py
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
- **Turnover:** not available.
- **Flows:** SEC data covers 2020 onward for 6–7 funds only. TNA is quarterly. Flows are
  reported per share class while TNA covers the whole fund, so flow as a percentage of TNA is
  approximate.
- **Index-fund benchmark:** Vanguard benchmark returns start in 2000-12.
