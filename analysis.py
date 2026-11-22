"""
CFM 301 Project 5 - Analysis
============================
Research question: Among US Large Growth mutual funds, does past performance
predict future alpha after accounting for momentum?  And, following the brief:
are each fund's fees justified (gross alpha large enough to leave investors a
non-negative net alpha)?

Reads the cleaned files written by clean_data.py (output/clean/) plus fund
metadata straight from data/Yahoo_Fund_Data.zip, and writes every table and
figure used in the report to output/results/.

    Req. 5   factor_regressions.csv        CAPM / FF3 / Carhart, net and gross
             alpha_change_summary.csv      how alpha and loadings move as factors are added
    Req. 6   gross_net_alpha.csv           fee, gross alpha, net alpha, index-benchmark alpha, verdict
             aggregate_portfolio.csv       equal-weighted portfolio of all funds (Fama-French 2010 style)
    Req. 7   index_benchmark.csv           constrained regression on VFIAX / VSMAX / VVIAX
    Req. 8   fee_vs_alpha.csv              cross-sectional regression of alpha on fee (Carhart 1997)
             fee_terciles.csv              alphas by fee tercile
             fee_distribution.csv          each fund's fee percentile in the Large Growth universe
    Req. 9   persistence_split_half.csv    split-half persistence tests (CAPM, FF3, Carhart)
             persistence_fund_alphas.csv   first- and second-half alpha of every fund
             momentum_portfolios.csv       Carhart-style tercile portfolios sorted on past 12-month return
             flow_performance.csv          flows vs past 1-year return
             berk_green_summary.csv        skill vs investor-alpha summary
    Req. 10  sensitivity_verdicts.csv      verdict per fund under alternative assumptions
             sensitivity_summary.csv       how many verdicts change under each alternative
             summary.txt                   headline numbers for the report
    Figures  fig1 ... fig6 (.png)

Run:  python clean_data.py   then   python analysis.py
"""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import optimize, stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "output" / "clean"
OUT = ROOT / "output" / "results"
YAHOO_ZIP = ROOT / "data" / "Yahoo_Fund_Data.zip"

# ---------------------------------------------------------------------------
# Analysis settings (documented in the report)
# ---------------------------------------------------------------------------
MODELS = {
    "CAPM": ["MKT_RF"],
    "FF3": ["MKT_RF", "SMB", "HML"],
    "Carhart": ["MKT_RF", "SMB", "HML", "Mom"],
}
MAIN_MODEL = "Carhart"          # verdicts use the 4-factor model (controls for momentum)
BENCH = ["VFIAX", "VSMAX", "VVIAX"]
MIN_OBS = 36                    # minimum months for any regression
ROLL_WINDOW = 36                # rolling-alpha window (sensitivity)
MIN_FUNDS_SORT = 30             # min funds in cross-section to form momentum terciles
CALENDAR_SPLIT = "2013-06"      # last month of first calendar half (2001-01..2025-12 midpoint)
CALENDAR_START = "2001-01"
WINSOR_PCTS = (0.01, 0.99)      # winsorization cut-offs for the sensitivity test
T_CRIT = 1.96

# Plot styling: categorical slots 1-3 of the validated reference palette
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

SUMMARY = []


def note(line=""):
    SUMMARY.append(line)
    print(line)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ols(y, X):
    """OLS with intercept. Returns coefficients, t-stats, residuals, R^2."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / sst if sst > 0 else np.nan
    return {"beta": beta, "se": se, "t": beta / se, "resid": resid,
            "r2": r2, "n": n, "k": k, "sigma": np.sqrt(s2)}


def factor_alpha(df, ycol, model):
    """Alpha and stats for one fund / one model, annualized as on Ch. 10 slide 29."""
    facs = MODELS[model]
    d = df.dropna(subset=[ycol] + facs)
    if len(d) < MIN_OBS:
        return None
    r = ols(d[ycol], d[facs])
    alpha_m, te_m = r["beta"][0], r["sigma"]   # regression tracking error = sd(residual)
    out = {
        "N_Months": r["n"], "Start": d["Month"].min(), "End": d["Month"].max(),
        "Alpha_Ann": alpha_m * 12, "Alpha_t": r["t"][0],
        "TE_Ann": te_m * np.sqrt(12),
        "IR_Ann": (alpha_m * 12) / (te_m * np.sqrt(12)),
        "R2": r["r2"],
    }
    for f, b, t in zip(facs, r["beta"][1:], r["t"][1:]):
        out[f"b_{f}"] = b
        out[f"t_{f}"] = t
    return out


def constrained_index_regression(y, R):
    """y = a + R w + e with w >= 0 and sum(w) = 1 (Ch. 10 slides 31-34)."""
    y, R = np.asarray(y, float), np.asarray(R, float)
    k = R.shape[1]

    def sse(w):
        e = y - R @ w
        e = e - e.mean()            # optimal intercept given w is the mean residual
        return e @ e

    res = optimize.minimize(
        sse, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    e = y - R @ w
    a = e.mean()
    resid = e - a
    n = len(y)
    dof = n - 1 - (k - 1)           # intercept + (k-1) free weights
    sigma = np.sqrt(resid @ resid / dof)
    sst = ((y - y.mean()) ** 2).sum()
    return {"alpha": a, "t": a / (sigma / np.sqrt(n)), "w": w, "sigma": sigma,
            "r2": 1 - resid @ resid / sst, "n": n, "converged": bool(res.success)}


def verdict(alpha):
    if pd.isna(alpha):
        return "n/a"
    return "Justified" if alpha >= 0 else "Not justified"


def past_12m_return(panel):
    """Compounded net return over months t-12..t-1 (needs 12 consecutive months)."""
    out = []
    for tic, g in panel.groupby("Ticker"):
        s = g.set_index(pd.PeriodIndex(g["Month"], freq="M"))["Return_Net"]
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
        cum = np.log1p(s).rolling(12, min_periods=12).sum()
        past = np.expm1(cum.shift(1))
        out.append(pd.DataFrame({"Ticker": tic, "Month": past.index.astype(str),
                                 "Past12_Return": past.values}))
    return pd.concat(out, ignore_index=True)


def cross_section(y, x, label_y, label_x):
    r = ols(y, x)
    return {"Dependent": label_y, "Regressor": label_x, "N_Funds": r["n"],
            "Intercept": r["beta"][0], "Intercept_t": r["t"][0],
            "Slope": r["beta"][1], "Slope_t": r["t"][1], "Slope_se": r["se"][1],
            "R2": r["r2"]}


def save(df, name):
    df.to_csv(OUT / name, index=False, float_format="%.6g")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load():
    panel = pd.read_csv(CLEAN / "fund_panel_clean.csv")
    info = pd.read_csv(CLEAN / "fund_info_clean.csv")
    bench = pd.read_csv(CLEAN / "benchmark_returns_clean.csv")
    factors = pd.read_csv(CLEAN / "factors_clean.csv")
    flows_q = pd.read_csv(CLEAN / "fund_flows_quarterly_clean.csv")
    flows_m = pd.read_csv(CLEAN / "fund_flows_monthly_clean.csv")
    with zipfile.ZipFile(YAHOO_ZIP) as z:
        with z.open("Fund_Metadata.csv") as f:
            meta = pd.read_csv(f)
        with z.open("Yahoo_Screening_Universe.csv") as f:
            universe = pd.read_csv(f, usecols=["symbol", "longName", "netExpenseRatio"])
    bench_fees = meta.set_index("Ticker").loc[BENCH, "Annual_Expense_Ratio"]
    return panel, info, bench, factors, flows_q, flows_m, bench_fees, universe


# ---------------------------------------------------------------------------
# Req. 5 / 6: factor regressions, net and gross
# ---------------------------------------------------------------------------
def factor_regressions(panel):
    rows = []
    for tic, g in panel.groupby("Ticker"):
        for basis, ycol in [("Net", "Exret_Net"), ("Gross", "Exret_Gross")]:
            for m in MODELS:
                r = factor_alpha(g, ycol, m)
                if r:
                    rows.append({"Ticker": tic, "Basis": basis, "Model": m, **r})
    return pd.DataFrame(rows)


def alpha_change_summary(fr):
    net = fr[fr["Basis"] == "Net"]
    rows = []
    for m in MODELS:
        d = net[net["Model"] == m]
        row = {"Model": m, "N_Funds": len(d),
               "Mean_Alpha_Ann": d["Alpha_Ann"].mean(),
               "Median_Alpha_Ann": d["Alpha_Ann"].median(),
               "Share_Alpha_Pos": (d["Alpha_Ann"] > 0).mean(),
               "N_Sig_Pos": (d["Alpha_t"] > T_CRIT).sum(),
               "N_Sig_Neg": (d["Alpha_t"] < -T_CRIT).sum(),
               "Mean_TE_Ann": d["TE_Ann"].mean(), "Mean_R2": d["R2"].mean()}
        for f in MODELS["Carhart"]:
            if f"b_{f}" in d:
                row[f"Mean_b_{f}"] = d[f"b_{f}"].mean()
        rows.append(row)
    summ = pd.DataFrame(rows)

    # per-fund change in alpha when SMB/HML and then momentum are added
    w = net.pivot(index="Ticker", columns="Model", values="Alpha_Ann")
    mom = net[net["Model"] == "Carhart"].set_index("Ticker")["b_Mom"]
    per_fund = pd.DataFrame({
        "Alpha_CAPM": w["CAPM"], "Alpha_FF3": w["FF3"], "Alpha_Carhart": w["Carhart"],
        "dAlpha_FF3_minus_CAPM": w["FF3"] - w["CAPM"],
        "dAlpha_Carhart_minus_FF3": w["Carhart"] - w["FF3"],
        "b_Mom": mom,
    }).reset_index()
    return summ, per_fund


def aggregate_portfolio(panel):
    """Equal-weighted portfolio of all active funds, as in Fama-French (2010)."""
    ew = panel.groupby("Month").agg(
        Exret_Net=("Exret_Net", "mean"), Exret_Gross=("Exret_Gross", "mean"),
        N_Funds=("Ticker", "count"),
        MKT_RF=("MKT_RF", "first"), SMB=("SMB", "first"),
        HML=("HML", "first"), Mom=("Mom", "first")).reset_index()
    ew = ew[ew["N_Funds"] >= MIN_FUNDS_SORT]
    rows = []
    for basis, ycol in [("Net", "Exret_Net"), ("Gross", "Exret_Gross")]:
        for m in MODELS:
            r = factor_alpha(ew, ycol, m)
            rows.append({"Portfolio": f"EW all funds (months with >= {MIN_FUNDS_SORT} funds)",
                         "Basis": basis, "Model": m, **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Req. 7: implementable index-fund benchmark
# ---------------------------------------------------------------------------
def index_benchmark(panel, bench, bench_fees, info):
    b = bench.set_index("Month")
    rows = []
    for tic, g in panel.groupby("Ticker"):
        d = g.set_index("Month").join(b[BENCH], how="inner").dropna(subset=BENCH + ["Return_Net"])
        if len(d) < MIN_OBS:
            continue
        fee = info.set_index("Ticker").loc[tic, "Annual_Expense_Ratio"]
        for basis, ycol in [("Net", "Return_Net"), ("Gross", "Return_Gross")]:
            r = constrained_index_regression(d[ycol], d[BENCH])
            rows.append({
                "Ticker": tic, "Basis": basis, "N_Months": r["n"],
                "Start": d.index.min(), "End": d.index.max(),
                "Alpha_Ann": r["alpha"] * 12, "Alpha_t": r["t"],
                "TE_Ann": r["sigma"] * np.sqrt(12),
                "IR_Ann": r["alpha"] * 12 / (r["sigma"] * np.sqrt(12)), "R2": r["r2"],
                **{f"w_{k}": v for k, v in zip(BENCH, r["w"])},
                "Fund_Fee": fee,
                "Replicating_Fee": float(np.dot(r["w"], bench_fees.values)),
                "Converged": r["converged"],
            })
    out = pd.DataFrame(rows)
    out["Fee_Saving"] = out["Fund_Fee"] - out["Replicating_Fee"]
    return out


# ---------------------------------------------------------------------------
# Req. 6 / 7: gross vs net table with verdicts
# ---------------------------------------------------------------------------
def gross_net_table(fr, ib, info):
    c = fr[fr["Model"] == MAIN_MODEL]
    net = c[c["Basis"] == "Net"].set_index("Ticker")
    gro = c[c["Basis"] == "Gross"].set_index("Ticker")
    ibn = ib[ib["Basis"] == "Net"].set_index("Ticker")
    inf = info.set_index("Ticker")
    t = pd.DataFrame({
        "Fund_Name": inf["Fund_Name"],
        "Annual_Fee": inf["Annual_Expense_Ratio"],
        "Zero_Fee_Flag": inf["Zero_Fee_Flag"],
        "N_Months": net["N_Months"], "Start": net["Start"], "End": net["End"],
        "Gross_Alpha": gro["Alpha_Ann"], "Gross_t": gro["Alpha_t"],
        "Net_Alpha": net["Alpha_Ann"], "Net_t": net["Alpha_t"],
        "Index_Bench_Alpha_Net": ibn["Alpha_Ann"], "Index_Bench_t": ibn["Alpha_t"],
        "Replicating_Fee": ibn["Replicating_Fee"],
    })
    t["Fee_Share_of_Gross_Alpha"] = np.where(t["Gross_Alpha"] > 0,
                                             t["Annual_Fee"] / t["Gross_Alpha"], np.nan)

    def classify(r):
        if r["Gross_Alpha"] <= 0:
            return "No skill (gross alpha <= 0)"
        if r["Net_Alpha"] < 0:
            return "Skill, fee absorbs it (gross > 0 > net)"
        return "Skill, investors keep alpha (net >= 0)"

    t["Berk_Green_Class"] = t.apply(classify, axis=1)
    t["Verdict"] = t["Net_Alpha"].map(verdict)
    t["Verdict_Significant"] = np.select(
        [t["Net_t"] > T_CRIT, t["Net_t"] < -T_CRIT],
        ["Justified (net alpha sig. > 0)", "Not justified (net alpha sig. < 0)"],
        "Not distinguishable from zero")
    return t.reset_index().sort_values("Net_Alpha", ascending=False)


# ---------------------------------------------------------------------------
# Req. 8: fees vs performance, fee distribution
# ---------------------------------------------------------------------------
def fee_analysis(gn, universe):
    d = gn[~gn["Zero_Fee_Flag"]].copy()
    reg = pd.DataFrame([
        cross_section(d["Gross_Alpha"], d["Annual_Fee"], "Gross alpha (Carhart, ann.)", "Annual fee"),
        cross_section(d["Net_Alpha"], d["Annual_Fee"], "Net alpha (Carhart, ann.)", "Annual fee"),
    ])
    # H0: slope on gross alpha = 1 -> each extra 1% of fee buys 1% of extra gross alpha
    g = reg.iloc[0]
    reg["t_Slope_eq_1"] = [(g["Slope"] - 1) / g["Slope_se"], np.nan]
    reg["Note"] = ["Slope >= 1 would mean higher fees are fully earned back",
                   "Slope of -1 means fees pass straight through to investors"]

    d["Fee_Tercile"] = pd.qcut(d["Annual_Fee"], 3, labels=["Low", "Mid", "High"])
    terc = d.groupby("Fee_Tercile", observed=True).agg(
        N_Funds=("Ticker", "count"), Mean_Fee=("Annual_Fee", "mean"),
        Mean_Gross_Alpha=("Gross_Alpha", "mean"), Mean_Net_Alpha=("Net_Alpha", "mean"),
        Share_Justified=("Net_Alpha", lambda x: (x >= 0).mean())).reset_index()

    # fee distribution: Yahoo Large Growth screening universe (all share classes)
    u = universe.dropna(subset=["netExpenseRatio"]).copy()
    u["Fee"] = u["netExpenseRatio"] / 100
    u["Is_Index"] = u["longName"].fillna("").str.contains(r"\bindex\b|\bidx\b|\bindx\b", case=False)
    active_fees = np.sort(u.loc[~u["Is_Index"], "Fee"].values)
    dist = gn[["Ticker", "Fund_Name", "Annual_Fee"]].copy()
    dist["Pctile_in_Active_Universe"] = dist["Annual_Fee"].map(
        lambda f: stats.percentileofscore(active_fees, f, kind="weak"))
    dist["Pctile_in_Our_Sample"] = dist["Annual_Fee"].rank(pct=True) * 100
    uni_summary = pd.DataFrame([{
        "Group": name, "N_Share_Classes": len(x), "Mean": x.mean(),
        "P10": x.quantile(.1), "P25": x.quantile(.25), "Median": x.median(),
        "P75": x.quantile(.75), "P90": x.quantile(.9)}
        for name, x in [("Yahoo Large Growth universe: active", u.loc[~u["Is_Index"], "Fee"]),
                        ("Yahoo Large Growth universe: index", u.loc[u["Is_Index"], "Fee"]),
                        ("Our sample (96 funds)", gn["Annual_Fee"])]])
    return reg, terc, dist.sort_values("Annual_Fee"), uni_summary, u


# ---------------------------------------------------------------------------
# Req. 9: persistence
# ---------------------------------------------------------------------------
def half_alphas(panel, split="own"):
    rows = []
    for tic, g in panel.groupby("Ticker"):
        g = g.sort_values("Month")
        if split == "own":
            if not g["Eligible_Persistence"].iloc[0]:
                continue
            h = len(g) // 2
            halves = (g.iloc[:h], g.iloc[h:])
        else:
            g = g[g["Month"] >= CALENDAR_START]
            halves = (g[g["Month"] <= CALENDAR_SPLIT], g[g["Month"] > CALENDAR_SPLIT])
        row = {"Ticker": tic, "Split": split}
        ok = True
        for i, hd in enumerate(halves, 1):
            for m in MODELS:
                r = factor_alpha(hd, "Exret_Net", m)
                if r is None:
                    ok = False
                    break
                row[f"H{i}_{m}_Alpha"] = r["Alpha_Ann"]
                row[f"H{i}_{m}_t"] = r["Alpha_t"]
                if m == "Carhart":
                    row[f"H{i}_b_Mom"] = r["b_Mom"]
            row[f"H{i}_Start"], row[f"H{i}_End"] = hd["Month"].min(), hd["Month"].max()
        if ok:
            row["H1_Raw_Return_Ann"] = halves[0]["Exret_Net"].mean() * 12
            rows.append(row)
    return pd.DataFrame(rows)


def persistence_tests(ha):
    rows = []
    for split, d in ha.groupby("Split"):
        rank_vars = [(m, f"H1_{m}_Alpha") for m in MODELS] + [("Raw excess return", "H1_Raw_Return_Ann")]
        for rank_name, rcol in rank_vars:
            for m in MODELS:
                y = d[f"H2_{m}_Alpha"]
                r = ols(y, d[rcol])
                rho, p = stats.spearmanr(d[rcol], y)
                terc = pd.qcut(d[rcol], 3, labels=False)
                top, bot = y[terc == 2], y[terc == 0]
                tt = stats.ttest_ind(top, bot, equal_var=False)
                rows.append({
                    "Split": split, "N_Funds": len(d),
                    "Ranked_On_First_Half": rank_name, "Second_Half_Alpha": m,
                    "Slope": r["beta"][1], "Slope_t": r["t"][1], "R2": r["r2"],
                    "Spearman_rho": rho, "Spearman_p": p,
                    "Top_Tercile_H2_Alpha": top.mean(), "Bottom_Tercile_H2_Alpha": bot.mean(),
                    "Top_minus_Bottom": top.mean() - bot.mean(), "TmB_t": tt.statistic,
                })
    return pd.DataFrame(rows)


def momentum_portfolios(panel, factors, past):
    """Carhart (1997): each month sort funds on past 12-month return into
    terciles, hold one month (equal-weighted), then estimate alphas."""
    d = panel.merge(past, on=["Ticker", "Month"], how="left").dropna(subset=["Past12_Return"])
    counts = d.groupby("Month")["Ticker"].transform("count")
    d = d[counts >= MIN_FUNDS_SORT].copy()
    d["Tercile"] = d.groupby("Month")["Past12_Return"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    port = d.groupby(["Month", "Tercile"])["Return_Net"].mean().unstack()
    port.columns = ["Losers", "Middle", "Winners"]
    port = port.reset_index().merge(factors, on="Month")
    for c in ["Losers", "Middle", "Winners"]:
        port[c] = port[c] - port["RF"]                     # excess returns
    port["Winners_minus_Losers"] = port["Winners"] - port["Losers"]   # zero-cost spread
    rows = []
    for c in ["Losers", "Middle", "Winners", "Winners_minus_Losers"]:
        for m in MODELS:
            r = factor_alpha(port, c, m)
            rows.append({"Portfolio": c, "Model": m, **r})
    return pd.DataFrame(rows), port


def flow_performance(flows_q, flows_m, past):
    # quarterly: flow over (Prev_Month, Month]; past 1-year return = 12 months
    # ending at Prev_Month, i.e. known at the start of the flow period
    q = flows_q.copy()
    pq = past.copy()
    pq["Month"] = (pd.PeriodIndex(pq["Month"], freq="M") - 1).astype(str)   # 12m ending at Month
    q = q.merge(pq.rename(columns={"Month": "Prev_Month"}), on=["Ticker", "Prev_Month"], how="left")
    q = q.rename(columns={"Past12_Return": "Past1Y_Return", "Flow_Implied_Pct": "Flow"})
    m = flows_m.merge(past, on=["Ticker", "Month"], how="left")       # 12m ending t-1
    m = m.rename(columns={"Past12_Return": "Past1Y_Return", "Net_Flow_Pct_TNA": "Flow"})

    rows, data = [], {}
    for name, d in [("Quarterly implied flow: TNA(t)-TNA(t-1)(1+r)", q),
                    ("Monthly SEC net flow / prior TNA", m)]:
        d = d.dropna(subset=["Flow", "Past1Y_Return"])
        data[name] = d
        r = ols(d["Flow"], d["Past1Y_Return"])
        dm = d[["Flow", "Past1Y_Return"]] - d.groupby("Ticker")[["Flow", "Past1Y_Return"]].transform("mean")
        rf = ols(dm["Flow"], dm["Past1Y_Return"])
        rows.append({"Flow_Measure": name, "N_Obs": len(d), "N_Funds": d["Ticker"].nunique(),
                     "Period": f"{d['Month'].min()} to {d['Month'].max()}",
                     "Mean_Flow": d["Flow"].mean(),
                     "Pooled_Slope": r["beta"][1], "Pooled_t": r["t"][1], "Pooled_R2": r["r2"],
                     "FundFE_Slope": rf["beta"][1], "FundFE_t": rf["t"][1]})
    return pd.DataFrame(rows), data


# ---------------------------------------------------------------------------
# Req. 10: sensitivity
# ---------------------------------------------------------------------------
def rolling_alphas(panel):
    facs = MODELS[MAIN_MODEL]
    rows = []
    for tic, g in panel.groupby("Ticker"):
        g = g.sort_values("Month").reset_index(drop=True)
        per = pd.PeriodIndex(g["Month"], freq="M")
        y, X = g["Exret_Net"].values, g[facs].values
        alphas = []
        for end in range(ROLL_WINDOW, len(g) + 1):
            s = end - ROLL_WINDOW
            if (per[end - 1] - per[s]).n != ROLL_WINDOW - 1:
                continue                                    # skip windows with gaps
            alphas.append(ols(y[s:end], X[s:end])["beta"][0] * 12)
        if alphas:
            a = np.array(alphas)
            rows.append({"Ticker": tic, "N_Windows": len(a), "Mean_Rolling_Alpha": a.mean(),
                         "Median_Rolling_Alpha": np.median(a), "Share_Windows_Pos": (a >= 0).mean()})
    return pd.DataFrame(rows)


def sensitivity(panel, gn, fr, ib, ha_cal):
    base = gn.set_index("Ticker")["Net_Alpha"]
    alts = {}

    # A. rolling 36-month windows
    ra = rolling_alphas(panel).set_index("Ticker")
    alts["Rolling 36m (mean alpha)"] = ra["Mean_Rolling_Alpha"]

    # B. calendar sub-periods
    cal = ha_cal.set_index("Ticker")
    alts[f"Sub-period {CALENDAR_START} to {CALENDAR_SPLIT}"] = cal[f"H1_{MAIN_MODEL}_Alpha"]
    alts[f"Sub-period after {CALENDAR_SPLIT}"] = cal[f"H2_{MAIN_MODEL}_Alpha"]

    # C. winsorized excess returns (pooled cut-offs)
    lo, hi = panel["Exret_Net"].quantile(list(WINSOR_PCTS))
    pw = panel.assign(Exret_Net=panel["Exret_Net"].clip(lo, hi))
    alts[f"Winsorized {WINSOR_PCTS[0]:.0%}/{WINSOR_PCTS[1]:.0%} ({lo:.1%}/{hi:.1%})"] = pd.Series(
        {t: (factor_alpha(g, "Exret_Net", MAIN_MODEL) or {}).get("Alpha_Ann", np.nan)
         for t, g in pw.groupby("Ticker")})

    # D. common post-2000 sample (same months as the index benchmark)
    p2 = panel[panel["Month"] >= "2000-12"]
    alts["Post-2000-12 sample"] = pd.Series(
        {t: (factor_alpha(g, "Exret_Net", MAIN_MODEL) or {}).get("Alpha_Ann", np.nan)
         for t, g in p2.groupby("Ticker")})

    # E/F. benchmark choice
    net = fr[fr["Basis"] == "Net"]
    alts["Benchmark: CAPM"] = net[net["Model"] == "CAPM"].set_index("Ticker")["Alpha_Ann"]
    alts["Benchmark: FF3"] = net[net["Model"] == "FF3"].set_index("Ticker")["Alpha_Ann"]
    alts["Benchmark: index funds"] = ib[ib["Basis"] == "Net"].set_index("Ticker")["Alpha_Ann"]

    tab = pd.DataFrame({"Base_Net_Alpha": base, "Base_Verdict": base.map(verdict)})
    summ = []
    for name, s in alts.items():
        s = s.reindex(tab.index)
        tab[f"{name} | alpha"] = s
        v = s.map(verdict)
        tab[f"{name} | verdict"] = v
        ok = v != "n/a"
        changed = (v[ok] != tab.loc[ok, "Base_Verdict"])
        summ.append({"Alternative": name, "N_Funds": int(ok.sum()),
                     "N_Justified": int((v == "Justified").sum()),
                     "Share_Justified": (v[ok] == "Justified").mean(),
                     "Mean_Net_Alpha": s.mean(),
                     "N_Verdicts_Changed": int(changed.sum()),
                     "Changed_to_Justified": int((changed & (v[ok] == "Justified")).sum()),
                     "Changed_to_Not": int((changed & (v[ok] == "Not justified")).sum())})
    vcols = [c for c in tab.columns if c.endswith("| verdict")]
    tab["N_Alternatives"] = (tab[vcols] != "n/a").sum(axis=1)
    tab["N_Agree_With_Base"] = sum((tab[c] == tab["Base_Verdict"]) for c in vcols)
    tab["Robust"] = tab["N_Agree_With_Base"] == tab["N_Alternatives"]
    base_row = {"Alternative": f"BASE: full sample, {MAIN_MODEL}", "N_Funds": len(base),
                "N_Justified": int((base >= 0).sum()), "Share_Justified": (base >= 0).mean(),
                "Mean_Net_Alpha": base.mean(), "N_Verdicts_Changed": 0,
                "Changed_to_Justified": 0, "Changed_to_Not": 0}
    return tab.reset_index(names="Ticker"), pd.DataFrame([base_row] + summ), ra


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def pct_axis(ax, which="both", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def fig_gross_net(gn):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)
    a = pd.concat([gn["Gross_Alpha"], gn["Net_Alpha"]])
    bins = np.linspace(a.min(), a.max(), 30)
    for ax, col, color, lab in [(axes[0], "Gross_Alpha", C_BLUE, "Gross-of-fee alpha"),
                                (axes[1], "Net_Alpha", C_ORANGE, "Net-of-fee alpha")]:
        x = gn[col]
        ax.hist(x, bins=bins, color=color, edgecolor="white", linewidth=1.5)
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(x.median(), color=INK, lw=1, ls="--")
        ax.set_title(f"{lab}: median {x.median():.2%}, {(x >= 0).mean():.0%} of funds at or above zero",
                     loc="left", fontsize=9)
        ax.set_ylabel("Funds")
        ax.grid(axis="x", visible=False)
    pct_axis(axes[1], "x")
    axes[1].set_xlabel(f"Annualized {MAIN_MODEL} alpha")
    fig.suptitle("Fees move the alpha distribution below zero", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_gross_vs_net_alpha.png")
    plt.close(fig)


def fig_fee_alpha(gn, reg):
    d = gn[~gn["Zero_Fee_Flag"]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = np.linspace(d["Annual_Fee"].min(), d["Annual_Fee"].max(), 50)
    for col, color, lab, i in [("Gross_Alpha", C_BLUE, "Gross alpha", 0),
                               ("Net_Alpha", C_ORANGE, "Net alpha", 1)]:
        ax.scatter(d["Annual_Fee"], d[col], s=22, color=color, edgecolor="white",
                   linewidth=0.8, label=lab, zorder=3)
        r = reg.iloc[i]
        ax.plot(xs, r["Intercept"] + r["Slope"] * xs, color=color, lw=2,
                label=f"{lab} fit: slope {r['Slope']:.2f} (t = {r['Slope_t']:.2f})")
    ax.plot(xs, xs, color=INK2, lw=1, ls=":", label="Gross alpha = fee (net alpha = 0)")
    ax.axhline(0, color=INK2, lw=0.8)
    pct_axis(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Annual expense ratio")
    ax.set_ylabel(f"Annualized {MAIN_MODEL} alpha")
    ax.set_title("Higher-fee funds do not earn more gross alpha", loc="left")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_fee_vs_alpha.png")
    plt.close(fig)


def fig_persistence(ha, pt):
    d = ha[ha["Split"] == "own"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
    for ax, m in zip(axes, ["CAPM", "Carhart"]):
        x, y = d[f"H1_{m}_Alpha"], d[f"H2_{m}_Alpha"]
        ax.scatter(x, y, s=20, color=C_BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        r = pt[(pt["Split"] == "own") & (pt["Ranked_On_First_Half"] == m) & (pt["Second_Half_Alpha"] == m)].iloc[0]
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, ols(y, x)["beta"][0] + r["Slope"] * xs, color=INK, lw=1.5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{m}: slope {r['Slope']:.2f} (t = {r['Slope_t']:.2f})\n"
                     f"rank correlation {r['Spearman_rho']:.2f}", loc="left", fontsize=8.5)
        ax.set_xlabel("First-half alpha")
        pct_axis(ax)
    axes[0].set_ylabel("Second-half alpha")
    fig.suptitle(f"Split-half alpha persistence, net returns ({len(d)} funds)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_persistence.png")
    plt.close(fig)


def fig_momentum(mp):
    d = mp[mp["Portfolio"] != "Winners_minus_Losers"]
    ports = ["Losers", "Middle", "Winners"]
    x = np.arange(len(ports))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, (m, color) in enumerate([("CAPM", C_BLUE), ("Carhart", C_ORANGE)]):
        v = d[d["Model"] == m].set_index("Portfolio").loc[ports, "Alpha_Ann"]
        bars = ax.bar(x + (i - 0.5) * 0.36, v, width=0.34, color=color, label=f"{m} alpha",
                      edgecolor="white", linewidth=1)
        for b, val in zip(bars, v):
            ax.annotate(f"{val:.1%}", (b.get_x() + b.get_width() / 2, val),
                        xytext=(0, 3 if val >= 0 else -10), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, ["Past losers", "Middle", "Past winners"])
    ax.grid(axis="x", visible=False)
    pct_axis(ax, "y", decimals=1)
    ax.set_ylabel("Annualized net alpha")
    ax.set_title("Fund portfolios sorted on past 12-month return", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_momentum_portfolios.png")
    plt.close(fig)


def fig_flows(fdata, fp):
    name = fp.iloc[0]["Flow_Measure"]
    d = fdata[name]
    r = fp.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(d["Past1Y_Return"], d["Flow"], s=20, color=C_BLUE, edgecolor="white",
               linewidth=0.8, zorder=3)
    xs = np.linspace(d["Past1Y_Return"].min(), d["Past1Y_Return"].max(), 20)
    fit = ols(d["Flow"], d["Past1Y_Return"])
    ax.plot(xs, fit["beta"][0] + fit["beta"][1] * xs, color=INK, lw=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    pct_axis(ax)
    ax.set_xlabel("Past 1-year net return (12 months before the quarter)")
    ax.set_ylabel("Quarterly flow, % of prior TNA")
    ax.set_title(f"Flows vs past 1-year return: slope {r['Pooled_Slope']:.3f} (t = {r['Pooled_t']:.2f})\n"
                 f"{int(r['N_Funds'])} funds, quarters {r['Period']}", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_flows_vs_past_return.png")
    plt.close(fig)


def fig_fee_distribution(u, gn):
    act = u.loc[~u["Is_Index"], "Fee"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(act, bins=np.arange(0, act.max() + 0.001, 0.001), color=C_BLUE,
            edgecolor="white", linewidth=1, label=f"Active Large Growth share classes (n = {len(act)})")
    ax.plot(gn["Annual_Fee"], np.full(len(gn), -2), "|", color=C_ORANGE, ms=10, mew=1.5,
            label=f"Funds in our sample (n = {len(gn)})")
    idx_med = u.loc[u["Is_Index"], "Fee"].median()
    ax.axvline(idx_med, color=INK, lw=1, ls="--")
    ax.annotate(f"Index fund median {idx_med:.2%}", (idx_med, ax.get_ylim()[1] * 0.9),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Annual expense ratio")
    ax.set_ylabel("Share classes")
    ax.grid(axis="x", visible=False)
    ax.set_title("Where the sample's fees sit in the Large Growth fee distribution", loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_fee_distribution.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel, info, bench, factors, flows_q, flows_m, bench_fees, universe = load()
    note("CFM 301 Project 5 - analysis summary (generated by analysis.py)")
    note(f"Sample: {panel['Ticker'].nunique()} active US Large Growth funds, {len(panel):,} fund-months, "
         f"{panel['Month'].min()} to {panel['Month'].max()}")
    note(f"Verdict rule: fee justified if annualized net {MAIN_MODEL} alpha >= 0")
    note()

    # Req. 5
    fr = factor_regressions(panel)
    save(fr, "factor_regressions.csv")
    acs, acf = alpha_change_summary(fr)
    save(acs, "alpha_change_summary.csv")
    save(acf, "alpha_change_by_fund.csv")
    note("[Req 5] Net alpha across models (annualized)")
    for _, r in acs.iterrows():
        note(f"  {r['Model']:8s} mean {r['Mean_Alpha_Ann']:+.2%}  median {r['Median_Alpha_Ann']:+.2%}  "
             f"share > 0 {r['Share_Alpha_Pos']:.0%}  sig+ {r['N_Sig_Pos']}  sig- {r['N_Sig_Neg']}  "
             f"mean R2 {r['Mean_R2']:.2f}")
    c = acs.set_index("Model").loc["Carhart"]
    note(f"  Carhart mean loadings: MKT {c['Mean_b_MKT_RF']:.2f}, SMB {c['Mean_b_SMB']:.2f}, "
         f"HML {c['Mean_b_HML']:.2f}, MOM {c['Mean_b_Mom']:.2f}")
    note(f"  Mean change in alpha CAPM->FF3 {acf['dAlpha_FF3_minus_CAPM'].mean():+.2%}, "
         f"FF3->Carhart {acf['dAlpha_Carhart_minus_FF3'].mean():+.2%}")
    note()

    # Req. 7
    ib = index_benchmark(panel, bench, bench_fees, info)
    save(ib, "index_benchmark.csv")

    # Req. 6
    gn = gross_net_table(fr, ib, info)
    save(gn, "gross_net_alpha.csv")
    agg = aggregate_portfolio(panel)
    save(agg, "aggregate_portfolio.csv")
    note(f"[Req 6] Gross vs net {MAIN_MODEL} alpha")
    note(f"  Mean fee {gn['Annual_Fee'].mean():.2%}; mean gross alpha {gn['Gross_Alpha'].mean():+.2%}; "
         f"mean net alpha {gn['Net_Alpha'].mean():+.2%}")
    note(f"  Fees justified (net alpha >= 0): {(gn['Verdict'] == 'Justified').sum()} of {len(gn)} funds; "
         f"net alpha significantly > 0: {(gn['Net_t'] > T_CRIT).sum()}, significantly < 0: {(gn['Net_t'] < -T_CRIT).sum()}")
    for _, r in agg[agg["Model"] == MAIN_MODEL].iterrows():
        note(f"  EW portfolio of all funds, {r['Basis']}: alpha {r['Alpha_Ann']:+.2%} (t = {r['Alpha_t']:.2f}), "
             f"{r['Start']} to {r['End']}")
    note()

    ibn = ib[ib["Basis"] == "Net"]
    note("[Req 7] Constrained index-fund benchmark (VFIAX/VSMAX/VVIAX, w >= 0, sum = 1)")
    note(f"  Index fees: " + ", ".join(f"{k} {v:.2%}" for k, v in bench_fees.items()))
    note(f"  {len(ibn)} funds; mean net alpha {ibn['Alpha_Ann'].mean():+.2%}; "
         f"share >= 0 {(ibn['Alpha_Ann'] >= 0).mean():.0%}; mean R2 {ibn['R2'].mean():.2f}")
    note(f"  Mean weights: " + ", ".join(f"{k} {ibn[f'w_{k}'].mean():.2f}" for k in BENCH))
    note(f"  Mean replicating fee {ibn['Replicating_Fee'].mean():.3%} vs mean fund fee "
         f"{ibn['Fund_Fee'].mean():.2%} (saving {ibn['Fee_Saving'].mean():.2%} a year)")
    note()

    # Req. 8
    reg, terc, dist, uni_summ, u = fee_analysis(gn, universe)
    save(reg, "fee_vs_alpha.csv")
    save(terc, "fee_terciles.csv")
    save(dist, "fee_distribution.csv")
    save(uni_summ, "fee_universe_summary.csv")
    note("[Req 8] Fees vs performance (zero-fee funds excluded)")
    for _, r in reg.iterrows():
        note(f"  {r['Dependent']} on fee: slope {r['Slope']:.2f} (t = {r['Slope_t']:.2f}), R2 {r['R2']:.2f}")
    note(f"  Test slope on gross alpha = 1: t = {reg.iloc[0]['t_Slope_eq_1']:.2f}")
    for _, r in terc.iterrows():
        note(f"  {r['Fee_Tercile']:4s} fee tercile: fee {r['Mean_Fee']:.2%}, gross {r['Mean_Gross_Alpha']:+.2%}, "
             f"net {r['Mean_Net_Alpha']:+.2%}, justified {r['Share_Justified']:.0%}")
    for _, r in uni_summ.iterrows():
        note(f"  {r['Group']}: n {r['N_Share_Classes']}, median {r['Median']:.2%}, P25 {r['P25']:.2%}, P75 {r['P75']:.2%}")
    note()

    # Req. 9: persistence
    ha = pd.concat([half_alphas(panel, "own"), half_alphas(panel, "calendar")], ignore_index=True)
    save(ha, "persistence_fund_alphas.csv")
    pt = persistence_tests(ha)
    save(pt, "persistence_split_half.csv")
    note("[Req 9] Split-half persistence (net alpha; first-half alpha predicts second-half alpha of the same model)")
    for split, label in [("own", "each fund's history split in half"),
                         ("calendar", f"calendar halves {CALENDAR_START}..{CALENDAR_SPLIT} | after")]:
        for m in MODELS:
            r = pt[(pt["Split"] == split) & (pt["Ranked_On_First_Half"] == m) & (pt["Second_Half_Alpha"] == m)]
            if len(r):
                r = r.iloc[0]
                note(f"  {label} | {m:7s}: N {r['N_Funds']}, slope {r['Slope']:+.2f} (t = {r['Slope_t']:.2f}), "
                     f"rho {r['Spearman_rho']:+.2f} (p = {r['Spearman_p']:.2f}), "
                     f"top-bottom tercile {r['Top_minus_Bottom']:+.2%} (t = {r['TmB_t']:.2f})")
    r = pt[(pt["Split"] == "own") & (pt["Ranked_On_First_Half"] == "CAPM") & (pt["Second_Half_Alpha"] == "Carhart")].iloc[0]
    note(f"  Ranked on first-half CAPM alpha, second-half Carhart alpha: slope {r['Slope']:+.2f} (t = {r['Slope_t']:.2f})")

    past = past_12m_return(panel)
    mp, port = momentum_portfolios(panel, factors, past)
    save(mp, "momentum_portfolios.csv")
    save(port, "momentum_portfolio_returns.csv")
    wml = mp[mp["Portfolio"] == "Winners_minus_Losers"].set_index("Model")
    note(f"  Carhart-style sort on past 12-month return ({mp['Start'].iloc[0]} to {mp['End'].iloc[0]}, "
         f">= {MIN_FUNDS_SORT} funds/month): winners-minus-losers alpha")
    for m in MODELS:
        note(f"    {m:8s} {wml.loc[m, 'Alpha_Ann']:+.2%} (t = {wml.loc[m, 'Alpha_t']:.2f})"
             + (f", MOM loading {wml.loc[m, 'b_Mom']:.2f} (t = {wml.loc[m, 't_Mom']:.2f})" if m == "Carhart" else ""))

    fp, fdata = flow_performance(flows_q, flows_m, past)
    save(fp, "flow_performance.csv")
    for _, r in fp.iterrows():
        note(f"  Flows ({r['Flow_Measure']}): {r['N_Obs']} obs, {r['N_Funds']} funds, {r['Period']}; "
             f"slope on past 1y return {r['Pooled_Slope']:.3f} (t = {r['Pooled_t']:.2f}); "
             f"within-fund {r['FundFE_Slope']:.3f} (t = {r['FundFE_t']:.2f})")

    bg = gn["Berk_Green_Class"].value_counts().rename_axis("Class").reset_index(name="N_Funds")
    bg["Share"] = bg["N_Funds"] / len(gn)
    save(bg, "berk_green_summary.csv")
    for _, r in bg.iterrows():
        note(f"  {r['Class']}: {r['N_Funds']} funds ({r['Share']:.0%})")
    note()

    # Req. 10
    ha_cal = ha[ha["Split"] == "calendar"]
    sv, ss, ra = sensitivity(panel, gn, fr, ib, ha_cal)
    save(sv, "sensitivity_verdicts.csv")
    save(ss, "sensitivity_summary.csv")
    save(ra, "rolling_alpha_summary.csv")
    note("[Req 10] Sensitivity of verdicts")
    for _, r in ss.iterrows():
        note(f"  {r['Alternative']:42s} N {r['N_Funds']:3d}  justified {r['N_Justified']:3d} "
             f"({r['Share_Justified']:.0%})  mean alpha {r['Mean_Net_Alpha']:+.2%}  changed {r['N_Verdicts_Changed']}")
    rob = sv[sv["Robust"]]
    note(f"  Verdict identical under every alternative for {len(rob)} of {len(sv)} funds "
         f"({(rob['Base_Verdict'] == 'Justified').sum()} robustly justified, "
         f"{(rob['Base_Verdict'] == 'Not justified').sum()} robustly not justified)")
    rj = rob.loc[rob["Base_Verdict"] == "Justified", "Ticker"].tolist()
    note(f"  Robustly justified: {', '.join(rj) if rj else 'none'}")

    # Figures
    fig_gross_net(gn)
    fig_fee_alpha(gn, reg)
    fig_persistence(ha, pt)
    fig_momentum(mp)
    fig_flows(fdata, fp)
    fig_fee_distribution(u, gn)

    (OUT / "summary.txt").write_text("\n".join(SUMMARY) + "\n")
    print(f"\nWrote tables, figures and summary.txt to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
