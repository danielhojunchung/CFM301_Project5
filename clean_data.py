"""
CFM 301 Project 5 - Data cleaning
=================================
Research question: Among US Large Growth mutual funds, does past performance
predict future alpha after accounting for momentum?

This script reads the raw zip files in data/ (no unzipping, no internet, no
manual steps) and writes cleaned files to output/clean/:

    fund_panel_clean.csv         active funds, one row per fund-month, merged with
                                 FF3 + momentum factors; net and gross (fee
                                 added back) returns and excess returns
    benchmark_returns_clean.csv  Vanguard S&P 500 / small-cap / value index funds
                                 (for the constrained index-fund benchmark),
                                 common months only, merged with RF
    factors_clean.csv            monthly Mkt-RF, SMB, HML, RF, Mom as decimals
    fund_info_clean.csv          one row per fund: fees, history length, flags
    fund_flows_monthly_clean.csv SEC N-PORT monthly net flows (2020+)
    fund_flows_quarterly_clean.csv  flow(t) = TNA(t) - TNA(t-1) x (1 + r(t))
                                 on quarter-end TNA, as in the project brief
    return_validation.csv        Yahoo returns vs SEC-reported returns
    outlier_months.csv           months with the most extreme fund returns
    cleaning_log.csv             observations affected by every cleaning step

Run:  python clean_data.py      (paths are relative to this file)
"""

from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "clean"

YAHOO_ZIP = DATA / "Yahoo_Fund_Data.zip"
SEC_ZIP = DATA / "SEC_Partial_Data.zip"
FF3_ZIP = DATA / "F-F_Research_Data_Factors_CSV.zip"
MOM_ZIP = DATA / "F-F_Momentum_Factor_CSV.zip"

# ---------------------------------------------------------------------------
# Cleaning settings (documented in the report)
# ---------------------------------------------------------------------------
BENCHMARK_TICKERS = ["VFIAX", "VSMAX", "VVIAX"]

# Funds labelled "Active" in the Yahoo pull that are in fact index funds.
MISCLASSIFIED_PASSIVE = {
    "IRLAX": "Index fund: Voya Russell Large Cap Growth Index Portfolio",
    "SWLGX": "Index fund: Schwab US Large-Cap Growth Index",
    "OTPSX": "Index fund: ProFunds NASDAQ-100",
    "LGPSX": "Index fund: ProFunds Large Cap Growth (tracks S&P 500 Growth)",
}

# A diversified equity fund cannot lose more than 50% or gain more than 100%
# in one month; such values are data errors (e.g. an unadjusted share split),
# not market events. Genuine crashes (Oct 1987: about -30%) stay in.
IMPLAUSIBLE_LOW, IMPLAUSIBLE_HIGH = -0.50, 1.00

# No winsorization: extreme months are real crash/boom events (1987, 2000-02,
# 2008, 2020) that investors actually experienced. Extremes are only FLAGGED
# (pooled 1st / 99th percentile of excess returns) so the report can show this.
WINSORIZE = False
OUTLIER_PCTS = (0.01, 0.99)

# Minimum history for the split-half persistence test (flag only).
MIN_MONTHS_PERSISTENCE = 120

FF_MISSING_CODES = [-99.99, -999.0]

# ---------------------------------------------------------------------------
# Cleaning log
# ---------------------------------------------------------------------------
LOG = []


def log(dataset, action, obs_before, obs_after, affected=None,
        funds_before=None, funds_after=None, note=""):
    if affected is None:
        affected = obs_before - obs_after
    LOG.append({
        "Step": len(LOG) + 1,
        "Dataset": dataset,
        "Action": action,
        "Obs_Before": obs_before,
        "Obs_After": obs_after,
        "Obs_Affected": affected,
        "Funds_Before": funds_before,
        "Funds_After": funds_after,
        "Note": note,
    })


def read_zip_csv(zip_path, name, **kwargs):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(name) as f:
            return pd.read_csv(f, **kwargs)


# ---------------------------------------------------------------------------
# 1. Fama-French factors
# ---------------------------------------------------------------------------
def read_ff_monthly(zip_path, csv_name):
    """Parse the monthly block of a Ken French CSV (stops at the annual block)."""
    with zipfile.ZipFile(zip_path) as z:
        lines = z.read(csv_name).decode("utf-8", errors="replace").splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith(","))
    cols = ["Month"] + [c.strip() for c in lines[header_idx].split(",")[1:]]
    rows = []
    for line in lines[header_idx + 1:]:
        if not re.match(r"^\s*\d{6}\s*,", line):
            break  # monthly section ends where the annual section begins
        rows.append([x.strip() for x in line.split(",")])
    df = pd.DataFrame(rows, columns=cols)
    df["Month"] = df["Month"].str[:4] + "-" + df["Month"].str[4:]
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_factors():
    ff3 = read_ff_monthly(FF3_ZIP, "F-F_Research_Data_Factors.csv")
    mom = read_ff_monthly(MOM_ZIP, "F-F_Momentum_Factor.csv")
    mom = mom.rename(columns={c: "Mom" for c in mom.columns if c.strip().lower() == "mom"})
    fac = ff3.merge(mom, on="Month", how="inner")
    n0 = len(fac)
    log("Factors", "Load FF3 + momentum monthly (inner join on month)",
        len(ff3), n0, affected=len(ff3) - n0,
        note=f"{fac.Month.min()} to {fac.Month.max()}")

    vals = ["Mkt-RF", "SMB", "HML", "RF", "Mom"]
    n_codes = int(fac[vals].isin(FF_MISSING_CODES).sum().sum())
    fac[vals] = fac[vals].replace(FF_MISSING_CODES, np.nan)
    fac = fac.dropna(subset=vals)
    log("Factors", "Missing-value codes (-99.99 / -999) and NaN rows dropped",
        n0, len(fac), note=f"{n_codes} coded values found")

    fac[vals] = fac[vals] / 100.0  # percent -> decimal
    log("Factors", "Converted from percent to decimal", len(fac), len(fac), affected=0)

    dups = int(fac.duplicated("Month").sum())
    fac = fac.drop_duplicates("Month")
    log("Factors", "Duplicate months dropped", len(fac) + dups, len(fac))
    return fac.rename(columns={"Mkt-RF": "MKT_RF"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Fund returns (Yahoo)
# ---------------------------------------------------------------------------
def clean_fund_returns(fac):
    meta = read_zip_csv(YAHOO_ZIP, "Fund_Metadata.csv")
    panel = read_zip_csv(YAHOO_ZIP, "Fund_Panel.csv")
    D = "Fund returns"

    def nf(df):
        return df["Ticker"].nunique()

    log(D, "Load Yahoo monthly fund panel (adjusted-close total returns)",
        len(panel), len(panel), affected=0, funds_before=nf(panel), funds_after=nf(panel),
        note=f"{(meta.Fund_Type == 'Active').sum()} active + "
             f"{(meta.Fund_Type == 'Benchmark').sum()} benchmark funds")

    # (a) implausible returns -> data errors -> set to missing
    bad = (panel["Return"] < IMPLAUSIBLE_LOW) | (panel["Return"] > IMPLAUSIBLE_HIGH)
    bad_rows = panel.loc[bad, ["Ticker", "Month", "Return"]]
    panel.loc[bad, "Return"] = np.nan
    log(D, f"Implausible monthly returns (< {IMPLAUSIBLE_LOW:.0%} or > {IMPLAUSIBLE_HIGH:.0%}) "
           "set to missing as data errors",
        len(panel), len(panel), affected=int(bad.sum()),
        funds_before=nf(panel), funds_after=nf(panel),
        note="; ".join(f"{r.Ticker} {r.Month} {r.Return:.1%}" for r in bad_rows.itertuples()))

    # (b) missing returns
    n0, f0 = len(panel), nf(panel)
    panel = panel.dropna(subset=["Return"])
    log(D, "Fund-months with missing return dropped (no filling/interpolation)",
        n0, len(panel), funds_before=f0, funds_after=nf(panel))

    # (c) duplicate fund-months
    n0 = len(panel)
    panel = panel.drop_duplicates(["Ticker", "Month"])
    log(D, "Duplicate (Ticker, Month) rows dropped", n0, len(panel),
        funds_before=nf(panel), funds_after=nf(panel))

    # (d) duplicate share classes of the same fund (one class per SEC series)
    series_per_ticker = meta.groupby("SEC_Series_ID")["Ticker"].transform("count")
    extra = meta.loc[series_per_ticker > 1].sort_values("Return_Observations", ascending=False)
    drop_classes = extra[extra.duplicated("SEC_Series_ID")]["Ticker"].tolist()
    n0, f0 = len(panel), nf(panel)
    panel = panel[~panel["Ticker"].isin(drop_classes)]
    log(D, "Extra share classes of the same SEC fund series dropped "
           "(keep the class with the longest history)",
        n0, len(panel), funds_before=f0, funds_after=nf(panel),
        note=", ".join(drop_classes) if drop_classes else "none: one class per series")

    # (e) index funds misclassified as active
    n0, f0 = len(panel), nf(panel)
    panel = panel[~panel["Ticker"].isin(MISCLASSIFIED_PASSIVE)]
    log(D, "Index funds labelled 'Active' removed", n0, len(panel),
        funds_before=f0, funds_after=nf(panel),
        note="; ".join(f"{k} ({v})" for k, v in MISCLASSIFIED_PASSIVE.items()))

    # (f) missing expense ratio
    no_fee = meta.loc[meta["Annual_Expense_Ratio"].isna(), "Ticker"]
    n0, f0 = len(panel), nf(panel)
    panel = panel[~panel["Ticker"].isin(no_fee)]
    log(D, "Funds with no expense ratio dropped", n0, len(panel),
        funds_before=f0, funds_after=nf(panel))

    # (g) merge factors (drop months without factor data)
    n0, f0 = len(panel), nf(panel)
    panel = panel.merge(fac, on="Month", how="inner")
    log(D, "Merged with FF3 + momentum factors (months without factors dropped)",
        n0, len(panel), funds_before=f0, funds_after=nf(panel))

    # returns: net (as reported), gross (monthly expense ratio added back)
    panel["Return_Net"] = panel["Return"]
    panel["Return_Gross"] = panel["Return"] + panel["Monthly_Expense_Ratio"]
    panel["Exret_Net"] = panel["Return_Net"] - panel["RF"]
    panel["Exret_Gross"] = panel["Return_Gross"] - panel["RF"]

    bench = panel[panel["Ticker"].isin(BENCHMARK_TICKERS)].copy()
    active = panel[panel["Fund_Type"] == "Active"].copy()
    log(D, "Benchmark index funds moved to their own file "
           "(remaining panel = active funds only)",
        len(panel), len(active), funds_before=nf(panel), funds_after=nf(active),
        note=", ".join(BENCHMARK_TICKERS))

    # (h) zero-fee funds: flag (gross alpha == net alpha; not open to retail)
    zero_fee = sorted(active.loc[active["Annual_Expense_Ratio"] <= 0, "Ticker"].unique())
    active["Zero_Fee_Flag"] = active["Ticker"].isin(zero_fee)
    log(D, "Zero expense-ratio funds FLAGGED, not dropped "
           "(Fidelity Series funds: only sold to other Fidelity funds)",
        len(active), len(active), affected=int(active["Zero_Fee_Flag"].sum()),
        funds_before=nf(active), funds_after=nf(active), note=", ".join(zero_fee))

    # (i) outliers: flag only, no winsorization
    lo, hi = active["Exret_Net"].quantile(list(OUTLIER_PCTS))
    active["Outlier_Flag"] = (active["Exret_Net"] < lo) | (active["Exret_Net"] > hi)
    log(D, f"Outliers FLAGGED (pooled excess return outside {OUTLIER_PCTS[0]:.0%}/"
           f"{OUTLIER_PCTS[1]:.0%} percentiles: {lo:.1%} / {hi:.1%}); "
           "NOT winsorized - real market events",
        len(active), len(active), affected=int(active["Outlier_Flag"].sum()),
        funds_before=nf(active), funds_after=nf(active))

    # (j) minimum history for persistence test: flag only
    months = active.groupby("Ticker")["Month"].transform("count")
    active["Eligible_Persistence"] = months >= MIN_MONTHS_PERSISTENCE
    n_elig = active.loc[active["Eligible_Persistence"], "Ticker"].nunique()
    log(D, f"Funds with >= {MIN_MONTHS_PERSISTENCE} months FLAGGED as eligible for "
           "split-half persistence",
        len(active), len(active), affected=int((~active["Eligible_Persistence"]).sum()),
        funds_before=nf(active), funds_after=nf(active),
        note=f"{n_elig} eligible funds; Obs_Affected = fund-months of ineligible funds")

    # outlier summary by month (evidence that extremes are market events)
    out_months = (active[active["Outlier_Flag"]]
                  .groupby("Month")
                  .agg(Flagged_Fund_Months=("Ticker", "count"),
                       Mean_Fund_Return=("Return_Net", "mean"))
                  .join(fac.set_index("Month")[["MKT_RF"]])
                  .sort_values("Flagged_Fund_Months", ascending=False)
                  .reset_index())

    # benchmark index funds: common months only
    bw = bench.pivot(index="Month", columns="Ticker", values="Return_Net")
    n_all = len(bw)
    bw = bw.dropna()
    bw = bw.join(fac.set_index("Month")["RF"]).reset_index()
    log("Benchmark index funds", "Kept months where all three Vanguard index funds "
        "have returns", n_all, len(bw), funds_before=len(BENCHMARK_TICKERS),
        funds_after=len(BENCHMARK_TICKERS), note=f"{bw.Month.min()} to {bw.Month.max()}")

    # fund-level info
    keep_cols = ["Ticker", "Fund_Name", "Category", "SEC_Series_ID", "SEC_Class_ID",
                 "Annual_Expense_Ratio", "Monthly_Expense_Ratio", "Current_TNA",
                 "Inception_Date"]
    info = (active.groupby("Ticker")
            .agg(First_Month=("Month", "min"), Last_Month=("Month", "max"),
                 N_Months=("Month", "count"),
                 Zero_Fee_Flag=("Zero_Fee_Flag", "first"),
                 Eligible_Persistence=("Eligible_Persistence", "first"),
                 N_Outlier_Months=("Outlier_Flag", "sum"))
            .reset_index()
            .merge(meta[keep_cols], on="Ticker", how="left"))
    info = info[keep_cols[:2] + [c for c in info.columns if c not in keep_cols[:2]]]

    panel_cols = ["Month", "Ticker", "Fund_Name", "SEC_Series_ID",
                  "Annual_Expense_Ratio", "Monthly_Expense_Ratio",
                  "Return_Net", "Return_Gross", "RF", "Exret_Net", "Exret_Gross",
                  "MKT_RF", "SMB", "HML", "Mom",
                  "Zero_Fee_Flag", "Outlier_Flag", "Eligible_Persistence"]
    active = active[panel_cols].sort_values(["Ticker", "Month"]).reset_index(drop=True)
    return active, bw, info, out_months


# ---------------------------------------------------------------------------
# 3. Fund flows and TNA (SEC N-PORT)
# ---------------------------------------------------------------------------
def clean_flows(active):
    D = "SEC flows"
    flows = read_zip_csv(SEC_ZIP, "Fund_Flows.csv")
    tna = read_zip_csv(SEC_ZIP, "Fund_TNA.csv")
    keep = set(active["Ticker"])

    log(D, "Load SEC N-PORT monthly flows", len(flows), len(flows), affected=0,
        funds_before=flows.Ticker.nunique(), funds_after=flows.Ticker.nunique(),
        note="Funds in SEC selection without any flow rows: "
             + ", ".join(sorted(set(read_zip_csv(SEC_ZIP, "Selected_Funds.csv").Ticker)
                                - set(flows.Ticker))))

    n0, f0 = len(flows), flows.Ticker.nunique()
    flows = flows.drop_duplicates(["Ticker", "Month"], keep="last")
    log(D, "Duplicate (Ticker, Month) rows dropped (keep latest filing)", n0, len(flows),
        funds_before=f0, funds_after=flows.Ticker.nunique())

    n0 = len(flows)
    flows = flows[flows["Month_Order"] == "Chronological"]
    log(D, "Rows whose month ordering could not be validated dropped", n0, len(flows),
        funds_before=f0, funds_after=flows.Ticker.nunique())

    n0 = len(flows)
    flows = flows.dropna(subset=["Net_Flow"])
    log(D, "Rows with missing net flow dropped", n0, len(flows),
        funds_before=flows.Ticker.nunique(), funds_after=flows.Ticker.nunique())

    n0, f0 = len(flows), flows.Ticker.nunique()
    dropped = sorted(set(flows.Ticker) - keep)
    flows = flows[flows["Ticker"].isin(keep)]
    log(D, "Funds not in the cleaned active sample dropped", n0, len(flows),
        funds_before=f0, funds_after=flows.Ticker.nunique(), note=", ".join(dropped))

    # validate Yahoo returns against SEC-reported class returns
    yr = active[["Ticker", "Month", "Return_Net"]]
    v = flows[["Ticker", "Month", "SEC_Class_Return"]].merge(yr, on=["Ticker", "Month"])
    v["Abs_Diff"] = (v["Return_Net"] - v["SEC_Class_Return"]).abs()
    validation = (v.groupby("Ticker")["Abs_Diff"]
                  .agg(N_Months="count", Median_Abs_Diff="median",
                       Mean_Abs_Diff="mean", Max_Abs_Diff="max")
                  .reset_index())

    # TNA (series level, quarter-ends)
    T = "SEC TNA"
    n0, f0 = len(tna), tna.Ticker.nunique()
    tna = tna.drop_duplicates(["Ticker", "Month"], keep="last")
    tna = tna.dropna(subset=["Total_Net_Assets"])
    tna = tna[tna["Total_Net_Assets"] > 0]
    log(T, "Duplicate, missing or non-positive TNA rows dropped", n0, len(tna),
        funds_before=f0, funds_after=tna.Ticker.nunique())
    n0, f0 = len(tna), tna.Ticker.nunique()
    tna = tna[tna["Ticker"].isin(keep)].sort_values(["Ticker", "Month"])
    log(T, "Funds not in the cleaned active sample dropped", n0, len(tna),
        funds_before=f0, funds_after=tna.Ticker.nunique())

    # monthly flows scaled by most recent prior TNA
    flows["Month_P"] = pd.PeriodIndex(flows["Month"], freq="M")
    tna["Month_P"] = pd.PeriodIndex(tna["Month"], freq="M")
    flows = flows.sort_values("Month_P")
    t2 = tna[["Ticker", "Month_P", "Month", "Total_Net_Assets"]].rename(
        columns={"Month": "TNA_Month", "Total_Net_Assets": "TNA_Prior"})
    # TNA must be from a month strictly before the flow month
    t2["Key"] = (t2["Month_P"] + 1).dt.to_timestamp()
    flows["Key"] = flows["Month_P"].dt.to_timestamp()
    flows = pd.merge_asof(flows.sort_values("Key"), t2.drop(columns="Month_P").sort_values("Key"),
                          on="Key", by="Ticker", direction="backward")
    flows["Net_Flow_Pct_TNA"] = flows["Net_Flow"] / flows["TNA_Prior"]
    flows = flows.merge(yr, on=["Ticker", "Month"], how="left")
    monthly = flows[["Ticker", "Month", "Class_ID", "Series_ID", "Sales", "Redemptions",
                     "Reinvestment", "Net_Flow", "TNA_Month", "TNA_Prior",
                     "Net_Flow_Pct_TNA", "SEC_Class_Return", "Return_Net"]] \
        .sort_values(["Ticker", "Month"]).reset_index(drop=True)
    log(D, "Monthly net flow scaled by most recent prior quarter-end TNA "
           "(class-level flow / series-level TNA: approximate)",
        len(monthly), len(monthly), affected=int(monthly["TNA_Prior"].isna().sum()),
        funds_before=monthly.Ticker.nunique(), funds_after=monthly.Ticker.nunique(),
        note="Obs_Affected = months with no prior TNA (flow % left missing)")

    # quarterly implied flows: flow(t) = TNA(t) - TNA(t-1) x (1 + r(t))
    ret = active.set_index(["Ticker", "Month"])["Return_Net"]
    rows = []
    for tic, g in tna.groupby("Ticker"):
        g = g.sort_values("Month_P").reset_index(drop=True)
        for i in range(1, len(g)):
            p, c = g.loc[i - 1], g.loc[i]
            gap = (c["Month_P"] - p["Month_P"]).n
            months = [str(p["Month_P"] + k) for k in range(1, gap + 1)]
            r = [ret.get((tic, m), np.nan) for m in months]
            r_q = np.prod([1 + x for x in r]) - 1 if not np.isnan(r).any() else np.nan
            implied = c["Total_Net_Assets"] - p["Total_Net_Assets"] * (1 + r_q)
            reported = monthly.loc[(monthly.Ticker == tic) & monthly.Month.isin(months), "Net_Flow"]
            rows.append({
                "Ticker": tic, "Month": c["Month"], "Prev_Month": p["Month"],
                "Gap_Months": gap, "TNA": c["Total_Net_Assets"],
                "TNA_Prev": p["Total_Net_Assets"], "Return_Period": r_q,
                "Flow_Implied": implied,
                "Flow_Implied_Pct": implied / p["Total_Net_Assets"],
                "Net_Flow_Reported_Sum": reported.sum() if len(reported) == gap else np.nan,
            })
    quarterly = pd.DataFrame(rows)
    n0 = len(quarterly)
    quarterly = quarterly[(quarterly["Gap_Months"] == 3) & quarterly["Return_Period"].notna()]
    log(T, "Quarterly implied flows: kept consecutive quarters (3-month gap) "
           "with fund returns for every month", n0, len(quarterly),
        funds_before=tna.Ticker.nunique(), funds_after=quarterly.Ticker.nunique())
    return monthly, quarterly.reset_index(drop=True), validation


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fac = clean_factors()
    active, bench, info, out_months = clean_fund_returns(fac)
    flows_m, flows_q, validation = clean_flows(active)

    fac.to_csv(OUT / "factors_clean.csv", index=False)
    active.to_csv(OUT / "fund_panel_clean.csv", index=False)
    bench.to_csv(OUT / "benchmark_returns_clean.csv", index=False)
    info.to_csv(OUT / "fund_info_clean.csv", index=False)
    flows_m.to_csv(OUT / "fund_flows_monthly_clean.csv", index=False)
    flows_q.to_csv(OUT / "fund_flows_quarterly_clean.csv", index=False)
    validation.to_csv(OUT / "return_validation.csv", index=False)
    out_months.to_csv(OUT / "outlier_months.csv", index=False)
    log_df = pd.DataFrame(LOG)
    log_df.to_csv(OUT / "cleaning_log.csv", index=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print("\nCLEANING LOG")
    print(log_df[["Step", "Dataset", "Action", "Obs_Before", "Obs_After",
                  "Obs_Affected", "Funds_After"]].to_string(index=False))
    print(f"\nActive funds: {active.Ticker.nunique()}  |  fund-months: {len(active)}  |  "
          f"{active.Month.min()} to {active.Month.max()}")
    print(f"Eligible for persistence (>= {MIN_MONTHS_PERSISTENCE} months): "
          f"{info.Eligible_Persistence.sum()}")
    print("\nTop outlier months (most flagged fund-months):")
    print(out_months.head(8).to_string(index=False))
    print("\nYahoo vs SEC return check:")
    print(validation.to_string(index=False))
    print(f"\nFiles written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
