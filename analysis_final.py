"""
Korea Income and Welfare — analysis pipeline
=============================================
Generates 14 publication-ready figures and 13 CSV tables from the raw
Korea Welfare Panel Survey data.

Usage
-----
    python analysis_final.py [--input-path PATH]

Output directories are created automatically under outputs/figures/ and
outputs/tables/.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

BASE_DIR      = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "Korea_Income_and_Welfare.csv"
TABLES_DIR    = BASE_DIR / "outputs" / "tables"
FIGURES_DIR   = BASE_DIR / "outputs" / "figures"

# ── Colour palette (navy / amber / teal / purple / red) ──────────────────────
NAVY   = "#1a3a5c"
AMBER  = "#c9772b"
TEAL   = "#0f7b6c"
PURPLE = "#6a3d9a"
RED    = "#b94a48"
SLATE  = "#4a5568"
LIGHT  = "#e8edf2"

# ── Lookup tables ─────────────────────────────────────────────────────────────
REGION_LABELS = {
    1: "Seoul", 2: "Gyeonggi",
    3: "S. Gyeongsang", 4: "N. Gyeongsang",
    5: "S. Chungcheong", 6: "Gangwon & N. Chungcheong",
    7: "Jeolla & Jeju",
}
EDU_SHORT = {
    1: "No formal\n(u/7)", 2: "No formal\n(7+)", 3: "Elementary",
    4: "Middle", 5: "High school", 6: "College",
    7: "University", 8: "Master's", 9: "Doctorate",
}
EDU_FULL = {
    1: "No formal education (under 7)", 2: "No formal education (age 7+)",
    3: "Elementary school",            4: "Middle school",
    5: "High school",                  6: "College",
    7: "University degree",            8: "Master's degree",
    9: "Doctorate",
}
MARRIAGE_LABELS = {
    0: "Unknown", 1: "N/A", 2: "Married",
    3: "Widowed",  4: "Separated", 5: "Single", 6: "Other", 9: "Unknown",
}
GENDER_LABELS   = {1: "Male", 2: "Female"}
RELIGION_LABELS = {1: "Has religion", 2: "No religion", 9: "Unknown"}

NUMERIC_FEATURES     = ["education_level", "age", "family_member", "year"]
CATEGORICAL_FEATURES = ["region_label", "gender_label", "marriage_label", "religion_label"]

BOOTSTRAP_N = 500   # iterations for confidence intervals


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    return p.parse_args()


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading and cleaning ─────────────────────────────────────────────────
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["region_label"]    = df["region"].map(REGION_LABELS)
    df["education_label"] = df["education_level"].map(EDU_FULL)
    df["edu_short"]       = df["education_level"].map(EDU_SHORT)
    df["marriage_label"]  = df["marriage"].map(MARRIAGE_LABELS)
    df["gender_label"]    = df["gender"].map(GENDER_LABELS)
    df["religion_label"]  = df["religion"].map(RELIGION_LABELS)
    df["income_usd"]      = (df["income"] * 1000) / 1100   # KRW 000s → USD
    df["age"]             = 2018 - df["year_born"]
    return df


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    af = df.loc[
        df["income_usd"].notna() & (df["income_usd"] > 0)
        & df["age"].between(18, 90)
        & df["education_label"].notna()
    ].copy()
    for col in ["family_member", "year", "education_level"]:
        af[col] = pd.to_numeric(af[col], errors="coerce")
    return af


def trim_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    return df.loc[df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)].copy()


def build_model_frame(af: pd.DataFrame) -> pd.DataFrame:
    mf = af[[
        "income_usd", "education_level", "age", "family_member", "year",
        "region_label", "gender_label", "marriage_label", "religion_label",
    ]].dropna().copy()
    mf = trim_iqr(mf, "income_usd")
    mf["log_income_usd"]  = np.log(mf["income_usd"])
    mf["age_centered"]    = mf["age"] - mf["age"].mean()
    mf["age_centered_sq"] = mf["age_centered"] ** 2
    return mf


# ── Summary tables ────────────────────────────────────────────────────────────
def build_education_summary(af: pd.DataFrame) -> pd.DataFrame:
    s = (
        af.groupby("education_label", dropna=False)
        .agg(n=("id", "count"),
             median_usd=("income_usd", "median"),
             mean_usd=("income_usd", "mean"),
             p25=("income_usd", lambda x: x.quantile(0.25)),
             p75=("income_usd", lambda x: x.quantile(0.75)))
        .reset_index()
    )
    s["order"] = s["education_label"].map({v: k for k, v in EDU_FULL.items()})
    return s.sort_values("order").drop(columns="order")


def build_region_summary(af: pd.DataFrame) -> pd.DataFrame:
    return (
        af.groupby("region_label", dropna=False)
        .agg(n=("id", "count"),
             median_usd=("income_usd", "median"),
             mean_usd=("income_usd", "mean"))
        .reset_index().sort_values("median_usd", ascending=False)
    )


def build_year_summary(af: pd.DataFrame) -> pd.DataFrame:
    return (
        af.groupby("year", dropna=False)
        .agg(n=("id", "count"),
             median_usd=("income_usd", "median"),
             mean_usd=("income_usd", "mean"))
        .reset_index().sort_values("year")
    )


def build_gender_edu_summary(af: pd.DataFrame) -> pd.DataFrame:
    s = (
        af.groupby(["education_level", "gender_label"], dropna=False)
        .agg(median_usd=("income_usd", "median"), n=("id", "count"))
        .reset_index().dropna(subset=["gender_label"])
    )
    s["edu_short"] = s["education_level"].map(EDU_SHORT)
    return s.sort_values("education_level")


def build_statistical_tests(af: pd.DataFrame) -> pd.DataFrame:
    edu_groups    = [g["income_usd"].dropna().values
                     for _, g in af.groupby("education_label")
                     if g["income_usd"].notna().sum() >= 25]
    region_groups = [g["income_usd"].dropna().values
                     for _, g in af.groupby("region_label")
                     if g["income_usd"].notna().sum() >= 25]
    kw_edu  = scipy_stats.kruskal(*edu_groups)
    kw_reg  = scipy_stats.kruskal(*region_groups)
    sp      = scipy_stats.spearmanr(af["education_level"], af["income_usd"], nan_policy="omit")
    male    = af.loc[af["gender_label"] == "Male",   "income_usd"]
    female  = af.loc[af["gender_label"] == "Female", "income_usd"]
    mw      = scipy_stats.mannwhitneyu(male, female, alternative="two-sided")
    return pd.DataFrame([
        {"id": "K1", "question": "Income distributions differ across education levels?",
         "test": "Kruskal-Wallis", "stat": float(kw_edu.statistic), "p": float(kw_edu.pvalue)},
        {"id": "K2", "question": "Education monotonically associated with income?",
         "test": "Spearman ρ",     "stat": float(sp.statistic),      "p": float(sp.pvalue)},
        {"id": "K3", "question": "Men and women show different income distributions?",
         "test": "Mann-Whitney U", "stat": float(mw.statistic),       "p": float(mw.pvalue)},
        {"id": "K4", "question": "Income distributions differ across regions?",
         "test": "Kruskal-Wallis (regions)", "stat": float(kw_reg.statistic), "p": float(kw_reg.pvalue)},
    ])


# ── ML pipeline ───────────────────────────────────────────────────────────────
def _linear_pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])


def _tree_pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])


def _collapse(name: str) -> str:
    clean = name.replace("numeric__", "").replace("categorical__", "")
    for base in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if clean == base or clean.startswith(f"{base}_"):
            return base
    return clean


def _eval(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "mae_usd": mean_absolute_error(y_true, y_pred),
        "rmse_usd": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def _build_models() -> dict:
    models: dict = {
        "Ridge": Pipeline([("pre", _linear_pre()),
                           ("m", Ridge(alpha=1.0))]),
        "Elastic Net": Pipeline([("pre", _linear_pre()),
                                 ("m", ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=10000))]),
        "Multivariable OLS": Pipeline([("pre", _linear_pre()),
                                       ("m", LinearRegression())]),
        "Gradient Boosting": Pipeline([("pre", _tree_pre()),
                                       ("m", GradientBoostingRegressor(
                                           n_estimators=300, learning_rate=0.05,
                                           max_depth=4, subsample=0.8, random_state=42))]),
        "Random Forest": Pipeline([("pre", _tree_pre()),
                                   ("m", RandomForestRegressor(
                                       n_estimators=400, max_depth=16,
                                       min_samples_leaf=3, random_state=42, n_jobs=-1))]),
    }
    if _HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("pre", _tree_pre()),
            ("m", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, n_jobs=-1)),
        ])
    return models


def run_models(mf: pd.DataFrame):
    target   = mf["income_usd"]
    edu_only = mf[["education_level"]]
    multi    = mf[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    Xte, Xve, yte, yve = train_test_split(edu_only, target, test_size=0.3, random_state=42)
    edu_m = LinearRegression().fit(Xte, yte)
    edu_p = edu_m.predict(Xve)

    Xtr, Xts, ytr, yts = train_test_split(multi, target, test_size=0.3, random_state=42)
    models  = _build_models()
    rows    = [_eval("Education-only OLS", yve, edu_p)]
    best_r2, best_name, best_model = float(rows[0]["r2"]), "Education-only OLS", None

    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        preds = mdl.predict(Xts)
        m = _eval(name, yts, preds)
        rows.append(m)
        if m["r2"] > best_r2:
            best_r2, best_name, best_model = float(m["r2"]), name, mdl

    comparison = pd.DataFrame(rows).sort_values("r2", ascending=False)

    if best_model is None:
        best_model = models["Random Forest"]
        best_model.fit(Xtr, ytr)
        best_name = "Random Forest"

    best_est = best_model.named_steps["m"]

    # best hyperparameters
    keep = {"alpha", "l1_ratio", "learning_rate", "max_depth",
            "min_samples_leaf", "n_estimators", "subsample", "colsample_bytree"}
    best_params = pd.DataFrame([
        {"model": best_name, "parameter": k, "value": v}
        for k, v in best_est.get_params().items() if k in keep
    ])

    # feature importance
    if hasattr(best_est, "feature_importances_"):
        fnames = best_model.named_steps["pre"].get_feature_names_out()
        fi = (
            pd.DataFrame({"feature": fnames, "importance": best_est.feature_importances_})
            .assign(group=lambda d: d["feature"].map(_collapse))
            .groupby("group", as_index=False)["importance"].sum()
            .sort_values("importance", ascending=False)
            .rename(columns={"group": "feature_group"})
        )
    else:
        fi = pd.DataFrame(columns=["feature_group", "importance"])

    pred_frame = pd.DataFrame({
        "actual": yts.values,
        "predicted": best_model.predict(Xts),
        "model": best_name,
    })

    # 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error",
               "neg_mse": "neg_mean_squared_error"}
    cv_rows = []
    all_cv_models = {"Education-only OLS": LinearRegression(), **models}
    for name, mdl in all_cv_models.items():
        X = edu_only if name == "Education-only OLS" else multi
        s = cross_validate(mdl, X, target, cv=cv, scoring=scoring, n_jobs=1)
        rmse_v = np.sqrt(-s["test_neg_mse"])
        cv_rows.append({
            "model": name,
            "cv_r2_mean": float(s["test_r2"].mean()),
            "cv_r2_std":  float(s["test_r2"].std(ddof=0)),
            "cv_rmse_mean": float(rmse_v.mean()),
            "cv_rmse_std":  float(rmse_v.std(ddof=0)),
        })
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_r2_mean", ascending=False)

    return comparison, fi, best_params, pred_frame, cv_df, best_model, Xts, yts


# ── Econometrics ──────────────────────────────────────────────────────────────
def run_econometric_models(mf: pd.DataFrame):
    formula = (
        "log_income_usd ~ education_level + age_centered + I(age_centered**2) "
        "+ family_member + year "
        "+ C(region_label) + C(gender_label) + C(marriage_label) + C(religion_label)"
    )
    ols_base = smf.ols(formula, data=mf).fit()
    ols_hc3  = smf.ols(formula, data=mf).fit(cov_type="HC3")
    qr = {q: smf.quantreg(formula, data=mf).fit(q=q, max_iter=2000) for q in (0.25, 0.5, 0.75)}
    return ols_base, ols_hc3, qr


def build_ols_summary(ols_hc3) -> pd.DataFrame:
    terms = {
        "education_level":       "Education step premium",
        "age_centered":          "Age (centered, linear)",
        "I(age_centered ** 2)":   "Age squared (lifecycle)",
        "family_member":         "Household size",
        "year":                  "Survey year trend",
    }
    rows = []
    for term, label in terms.items():
        coef = float(ols_hc3.params[term])
        ci   = ols_hc3.conf_int().loc[term]
        rows.append({
            "term": term, "label": label,
            "coefficient": coef,
            "std_error": float(ols_hc3.bse[term]),
            "p_value": float(ols_hc3.pvalues[term]),
            "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]),
            "pct_effect": (np.exp(coef) - 1) * 100,
        })
    return pd.DataFrame(rows)


def build_econometric_diagnostics(ols_base) -> pd.DataFrame:
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_base.resid, ols_base.model.exog)
    reset  = linear_reset(ols_base, power=2, use_f=True)
    jb     = scipy_stats.jarque_bera(ols_base.resid)
    return pd.DataFrame([
        {"diagnostic": "Adjusted R²",    "stat": float(ols_base.rsquared_adj), "p": np.nan},
        {"diagnostic": "Breusch-Pagan LM", "stat": float(bp_lm), "p": float(bp_lm_p)},
        {"diagnostic": "Breusch-Pagan F",  "stat": float(bp_f),  "p": float(bp_f_p)},
        {"diagnostic": "RESET F",          "stat": float(reset.fvalue), "p": float(reset.pvalue)},
        {"diagnostic": "Jarque-Bera",      "stat": float(jb.statistic), "p": float(jb.pvalue)},
        {"diagnostic": "Observations",     "stat": float(ols_base.nobs), "p": np.nan},
    ])


def build_econometric_vif(mf: pd.DataFrame) -> pd.DataFrame:
    design = sm.add_constant(
        mf[["education_level", "age_centered", "age_centered_sq", "family_member", "year"]].dropna(),
        has_constant="add",
    )
    return pd.DataFrame([
        {"term": col, "vif": float(variance_inflation_factor(design.values, i))}
        for i, col in enumerate(design.columns) if col != "const"
    ])


def build_quantile_summary(qr: dict) -> pd.DataFrame:
    rows = []
    for q, mdl in qr.items():
        ci   = mdl.conf_int().loc["education_level"]
        coef = float(mdl.params["education_level"])
        rows.append({
            "model": f"Q{int(q*100)} regression",
            "quantile": q,
            "education_coef": coef,
            "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]),
            "pct_premium": (np.exp(coef) - 1) * 100,
            "family_member_coef": float(mdl.params["family_member"]),
            "year_coef": float(mdl.params["year"]),
            "pseudo_r2": float(mdl.prsquared),
        })
    return pd.DataFrame(rows).sort_values("quantile")


def build_bootstrap_education_premium(mf: pd.DataFrame, n: int = BOOTSTRAP_N) -> pd.DataFrame:
    """Bootstrap distribution of the OLS education premium (log-scale coef)."""
    formula = (
        "log_income_usd ~ education_level + age_centered + I(age_centered**2) "
        "+ family_member + year + C(region_label) + C(gender_label)"
    )
    rng  = np.random.default_rng(42)
    coefs = []
    for _ in range(n):
        s = mf.sample(len(mf), replace=True, random_state=int(rng.integers(0, 2**31)))
        coefs.append(float(smf.ols(formula, data=s).fit().params["education_level"]))
    return pd.DataFrame({"bootstrap_coef": coefs,
                         "pct_premium": [(np.exp(c) - 1) * 100 for c in coefs]})


# ── Style helpers ─────────────────────────────────────────────────────────────
def _style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.08)
    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#cccccc",
        "grid.color":        "#e5e5e5",
        "grid.linewidth":    0.7,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "font.family":       "sans-serif",
    })


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── Individual figure functions ───────────────────────────────────────────────

def fig01_income_by_education(edu_summary: pd.DataFrame) -> None:
    _style()
    short_map  = {v: k for k, v in EDU_FULL.items()}
    s = edu_summary.copy()
    s["short"] = s["education_label"].map({v: EDU_SHORT[k] for k, v in EDU_FULL.items()})
    s = s.dropna(subset=["short"])

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(s)), s["mean_usd"], color=NAVY, width=0.6, zorder=3, alpha=0.9)
    # IQR whiskers
    for i, (_, row) in enumerate(s.iterrows()):
        ax.plot([i, i], [row["p25"], row["p75"]],
                color=AMBER, linewidth=2.5, solid_capstyle="round", zorder=4)
        ax.plot(i, row["median_usd"], "o", color=AMBER, markersize=6, zorder=5)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s["short"], rotation=0, ha="center", fontsize=9)
    ax.set_title("Average income by education level\n(amber line = IQR, dot = median)",
                 fontsize=13, pad=14)
    ax.set_ylabel("Income (USD / month)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", alpha=0.5, zorder=0)
    _save("fig01_income_by_education.png")


def fig02_income_distribution_by_education(af: pd.DataFrame) -> None:
    _style()
    order = [EDU_SHORT[k] for k in sorted(EDU_SHORT)]
    af2   = af.copy()
    af2["edu_s"] = af2["education_level"].map(EDU_SHORT)
    # cap at 97th for visual clarity
    cap   = af2["income_usd"].quantile(0.97)
    af2   = af2[af2["income_usd"] <= cap]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(
        data=af2, x="edu_s", y="income_usd",
        order=[o for o in order if o in af2["edu_s"].unique()],
        color=NAVY, inner="quartile", linewidth=0.9,
        alpha=0.75, cut=0, ax=ax,
    )
    ax.set_title("Income distribution by education level (violin + quartiles)",
                 fontsize=13, pad=14)
    ax.set_xlabel("Education level")
    ax.set_ylabel("Income (USD / month)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    _save("fig02_income_distribution_education.png")


def fig03_income_by_region(region_summary: pd.DataFrame) -> None:
    _style()
    s = region_summary.sort_values("median_usd", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(s["region_label"], s["median_usd"], color=AMBER, zorder=3, alpha=0.9)
    ax.bar_label(bars, fmt="$%.0f", padding=5, fontsize=9)
    ax.set_title("Median monthly income by region", fontsize=13, pad=14)
    ax.set_xlabel("Median income (USD / month)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="x", alpha=0.5, zorder=0)
    _save("fig03_median_income_by_region.png")


def fig04_income_trend_over_time(year_summary: pd.DataFrame) -> None:
    _style()
    s = year_summary.copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(s["year"], s["median_usd"], alpha=0.12, color=TEAL)
    ax.plot(s["year"], s["median_usd"], marker="o", linewidth=2.2, color=TEAL, label="Median")
    ax.plot(s["year"], s["mean_usd"],   marker="s", linewidth=2.2, linestyle="--",
            color=AMBER, label="Mean")
    ax.set_title("Income trend across survey waves", fontsize=13, pad=14)
    ax.set_xlabel("Survey year")
    ax.set_ylabel("Income (USD / month)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(frameon=False)
    ax.grid(alpha=0.4)
    _save("fig04_income_trend_over_time.png")


def fig05_gender_income_gap(af: pd.DataFrame) -> None:
    _style()
    s = (
        af.groupby("gender_label", dropna=False)["income_usd"]
        .agg(["median", "mean", "count"])
        .reset_index().dropna(subset=["gender_label"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label, title in [
        (axes[0], "median", "Median income (USD)", "Median"),
        (axes[1], "mean",   "Mean income (USD)",   "Mean"),
    ]:
        colors = [NAVY, AMBER]
        bars = ax.bar(s["gender_label"], s[metric], color=colors, width=0.5, zorder=3, alpha=0.9)
        ax.bar_label(bars, fmt="$%.0f", padding=4, fontsize=11)
        ax.set_title(f"{title} income by gender", fontsize=12, pad=10)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(axis="y", alpha=0.4, zorder=0)
        gap = (s[metric].iloc[0] - s[metric].iloc[1]) / s[metric].iloc[0] * 100
        ax.text(0.5, 0.06, f"Gap: {gap:.1f}%", ha="center", va="bottom",
                transform=ax.transAxes, fontsize=10, color=RED)
    fig.suptitle("Raw gender income gap (unconditional)", fontsize=13, y=1.01)
    _save("fig05_gender_income_gap.png")


def fig06_gender_gap_by_education(gender_edu: pd.DataFrame) -> None:
    _style()
    pivot = gender_edu.pivot(index="edu_short", columns="gender_label", values="median_usd")
    if "Male" not in pivot.columns or "Female" not in pivot.columns:
        return
    pivot = pivot.dropna()
    pivot["gap_pct"] = (pivot["Male"] - pivot["Female"]) / pivot["Male"] * 100
    pivot["order"]   = pivot.index.map({v: k for k, v in EDU_SHORT.items()})
    pivot = pivot.sort_values("order")

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [RED if g > pivot["gap_pct"].mean() else PURPLE for g in pivot["gap_pct"]]
    ax.bar(pivot.index, pivot["gap_pct"], color=colors, zorder=3, alpha=0.9)
    ax.axhline(pivot["gap_pct"].mean(), color=SLATE, linewidth=1.4,
               linestyle="--", label=f"Sample avg ({pivot['gap_pct'].mean():.1f}%)")
    ax.set_title("Gender income gap by education level\n"
                 "(% by which male median exceeds female median)", fontsize=13, pad=14)
    ax.set_xlabel("Education level")
    ax.set_ylabel("Gender gap (%)")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    _save("fig06_gender_gap_by_education.png")


def fig07_hypothesis_tests(stat_tests: pd.DataFrame) -> None:
    _style()
    t = stat_tests.copy()
    t["neg_log_p"] = -np.log10(t["p"].clip(lower=1e-300))
    colors = [NAVY if v > 2 else AMBER for v in t["neg_log_p"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(t["id"], t["neg_log_p"], color=colors, zorder=3, alpha=0.9)
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9, label_type="edge")
    ax.axvline(2, color=RED, linestyle="--", linewidth=1.4, label="p = 0.01 threshold")
    ax.set_title("Statistical significance of hypothesis tests (−log₁₀ p)", fontsize=13, pad=14)
    ax.set_xlabel("−log₁₀(p-value)  [higher = more significant]")
    ax.set_ylabel("Test")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.4, zorder=0)
    # annotations
    labels = {"K1": "Education vs income", "K2": "Spearman rank",
              "K3": "Gender gap",          "K4": "Regional gap"}
    for _, row in t.iterrows():
        ax.text(0.3, list(t["id"]).index(row["id"]),
                labels.get(row["id"], ""), va="center", fontsize=8.5, color="white")
    _save("fig07_hypothesis_tests.png")


def fig08_ols_coefficients(ols_summary: pd.DataFrame) -> None:
    _style()
    s = ols_summary.copy()
    s["err_lo"] = s["coefficient"] - s["ci_low"]
    s["err_hi"] = s["ci_high"] - s["coefficient"]
    colors = [TEAL if p < 0.001 else AMBER for p in s["p_value"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(s))
    ax.bar(x, s["coefficient"], color=colors, alpha=0.85, zorder=3, width=0.55)
    ax.errorbar(x, s["coefficient"],
                yerr=[s["err_lo"], s["err_hi"]],
                fmt="none", ecolor=RED, capsize=6, linewidth=1.6, zorder=4)
    ax.axhline(0, color=SLATE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(s["label"], rotation=18, ha="right", fontsize=10)
    ax.set_title("OLS earnings equation — key coefficients with 95 % HC3 CI\n"
                 "(teal = p<0.001)", fontsize=13, pad=14)
    ax.set_ylabel("Coefficient (log-income scale)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    _save("fig08_ols_coefficients.png")


def fig09_ols_diagnostics(mf: pd.DataFrame, ols_base) -> None:
    _style()
    resid  = ols_base.resid
    fitted = ols_base.fittedvalues

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Q-Q
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid, dist="norm")
    axes[0].scatter(osm, osr, s=5, alpha=0.25, color=NAVY, zorder=3)
    axes[0].plot([min(osm), max(osm)],
                 [slope * min(osm) + intercept, slope * max(osm) + intercept],
                 color=RED, linewidth=1.8)
    axes[0].set_title("Normal Q-Q (residuals)", fontsize=11)
    axes[0].set_xlabel("Theoretical quantiles")
    axes[0].set_ylabel("Sample quantiles")

    # Residuals vs fitted (density-coloured)
    axes[1].scatter(fitted, resid, s=4, alpha=0.20, color=TEAL, zorder=3)
    axes[1].axhline(0, color=RED, linewidth=1.8, linestyle="--")
    # LOWESS smoother
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw = lowess(resid, fitted, frac=0.15)
    axes[1].plot(lw[:, 0], lw[:, 1], color=AMBER, linewidth=2, label="LOWESS")
    axes[1].set_title("Residuals vs fitted (LOWESS overlay)", fontsize=11)
    axes[1].set_xlabel("Fitted log-income")
    axes[1].set_ylabel("Residuals")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("OLS diagnostic plots — semilog earnings equation", fontsize=13, y=1.02)
    _save("fig09_ols_diagnostics.png")


def fig10_education_premium_by_quantile(
    ols_summary: pd.DataFrame, q_summary: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> None:
    _style()
    ols_row = ols_summary.loc[ols_summary["term"] == "education_level"].iloc[0]
    combined = pd.concat([
        pd.DataFrame([{
            "model": "OLS (mean)",
            "prem": (np.exp(ols_row["coefficient"]) - 1) * 100,
            "lo":   (np.exp(ols_row["ci_low"])  - 1) * 100,
            "hi":   (np.exp(ols_row["ci_high"]) - 1) * 100,
        }]),
        q_summary.assign(
            prem = lambda d: d["pct_premium"],
            lo   = lambda d: (np.exp(d["ci_low"])  - 1) * 100,
            hi   = lambda d: (np.exp(d["ci_high"]) - 1) * 100,
        )[["model", "prem", "lo", "hi"]],
    ], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: point estimates + CI
    x = np.arange(len(combined))
    axes[0].errorbar(x, combined["prem"],
                     yerr=[combined["prem"] - combined["lo"],
                           combined["hi"]   - combined["prem"]],
                     fmt="o", capsize=7, color=NAVY,
                     linewidth=2, markersize=9, zorder=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(combined["model"], rotation=18, ha="right", fontsize=10)
    axes[0].set_title("Education premium:\nOLS mean vs quantile regressions", fontsize=11)
    axes[0].set_ylabel("Income premium per step (%)")
    axes[0].grid(axis="y", alpha=0.4)

    # Right: bootstrap distribution
    axes[1].hist(bootstrap_df["pct_premium"], bins=40,
                 color=TEAL, edgecolor="white", alpha=0.85)
    pct_lo = np.percentile(bootstrap_df["pct_premium"], 2.5)
    pct_hi = np.percentile(bootstrap_df["pct_premium"], 97.5)
    axes[1].axvline(pct_lo, color=RED, linestyle="--", linewidth=1.5,
                    label=f"95 % CI [{pct_lo:.1f}%, {pct_hi:.1f}%]")
    axes[1].axvline(pct_hi, color=RED, linestyle="--", linewidth=1.5)
    axes[1].axvline(bootstrap_df["pct_premium"].mean(), color=AMBER,
                    linewidth=2, label=f"Mean {bootstrap_df['pct_premium'].mean():.2f}%")
    axes[1].set_title(f"Bootstrap distribution of education premium\n({BOOTSTRAP_N} iterations)",
                      fontsize=11)
    axes[1].set_xlabel("Premium (%)")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Education income premium — stability and distributional heterogeneity",
                 fontsize=13, y=1.02)
    _save("fig10_education_premium_by_quantile.png")


def fig11_model_comparison(comparison: pd.DataFrame) -> None:
    _style()
    s = comparison.sort_values("r2")
    max_r2 = s["r2"].max()
    colors = [RED if abs(r - max_r2) < 1e-9 else NAVY for r in s["r2"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(s["model"], s["r2"], color=colors, zorder=3, alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=9.5)
    ax.set_title("Out-of-sample R² — model comparison (30 % holdout)", fontsize=13, pad=14)
    ax.set_xlabel("R-squared")
    ax.set_xlim(0, 0.9)
    ax.axvline(0.5, color=SLATE, linewidth=0.8, linestyle=":")
    ax.grid(axis="x", alpha=0.4, zorder=0)
    _save("fig11_model_comparison.png")


def fig12_cross_validation(cv_df: pd.DataFrame) -> None:
    _style()
    s = cv_df.sort_values("cv_r2_mean")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(s["model"], s["cv_r2_mean"], color=TEAL, alpha=0.85, zorder=3)
    ax.errorbar(
        s["cv_r2_mean"], np.arange(len(s)),
        xerr=s["cv_r2_std"],
        fmt="none", ecolor=RED, capsize=5, linewidth=1.6, zorder=4,
    )
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=9.5)
    ax.set_title("Five-fold cross-validated R² (mean ± std)", fontsize=13, pad=14)
    ax.set_xlabel("CV R-squared")
    ax.set_xlim(0, 0.9)
    ax.grid(axis="x", alpha=0.4, zorder=0)
    _save("fig12_cross_validation_r2.png")


def fig13_feature_importance(fi: pd.DataFrame) -> None:
    _style()
    s = fi.head(8).sort_values("importance")
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [RED if s.iloc[i]["importance"] == s["importance"].max() else PURPLE
              for i in range(len(s))]
    bars = ax.barh(s["feature_group"], s["importance"], color=colors, zorder=3, alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_title("Feature group importance — best ML model\n(red = dominant predictor)",
                 fontsize=13, pad=14)
    ax.set_xlabel("Importance (mean impurity decrease)")
    ax.grid(axis="x", alpha=0.4, zorder=0)
    _save("fig13_feature_importance.png")


def fig14_actual_vs_predicted(pred_frame: pd.DataFrame) -> None:
    _style()
    pf = pred_frame.copy()
    mn = min(pf["actual"].min(), pf["predicted"].min())
    mx = max(pf["actual"].max(), pf["predicted"].max())
    r2 = r2_score(pf["actual"], pf["predicted"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pf["actual"], pf["predicted"],
               s=7, alpha=0.25, color=NAVY, zorder=3, rasterized=True)
    ax.plot([mn, mx], [mn, mx], color=RED, linewidth=1.8, label="Perfect fit")
    # ±20 % band
    ax.fill_between([mn, mx], [mn * 0.8, mx * 0.8], [mn * 1.2, mx * 1.2],
                    alpha=0.08, color=AMBER, label="±20% band")
    ax.set_title(f"Actual vs predicted income\n{pf['model'].iloc[0]}  (R² = {r2:.3f})",
                 fontsize=13, pad=14)
    ax.set_xlabel("Actual income (USD)")
    ax.set_ylabel("Predicted income (USD)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(frameon=False)
    _save("fig14_actual_vs_predicted.png")


# ── Export wrappers ───────────────────────────────────────────────────────────
def export_tables(
    edu_s, reg_s, yr_s, stat_tests,
    comparison, fi, best_params,
    ols_s, diag, vif, q_s, cv_df, bootstrap_df,
) -> None:
    edu_s.to_csv(TABLES_DIR / "education_income_summary.csv", index=False)
    reg_s.to_csv(TABLES_DIR / "region_income_summary.csv", index=False)
    yr_s.to_csv(TABLES_DIR  / "year_income_summary.csv", index=False)
    stat_tests.to_csv(TABLES_DIR  / "statistical_tests.csv", index=False)
    comparison.to_csv(TABLES_DIR  / "model_comparison.csv", index=False)
    fi.to_csv(TABLES_DIR          / "feature_importance.csv", index=False)
    best_params.to_csv(TABLES_DIR / "ml_best_params.csv", index=False)
    ols_s.to_csv(TABLES_DIR       / "econometric_ols_summary.csv", index=False)
    diag.to_csv(TABLES_DIR        / "econometric_diagnostics.csv", index=False)
    vif.to_csv(TABLES_DIR         / "econometric_vif.csv", index=False)
    q_s.to_csv(TABLES_DIR         / "quantile_regression_summary.csv", index=False)
    cv_df.to_csv(TABLES_DIR       / "ml_cross_validation.csv", index=False)
    bootstrap_df.to_csv(TABLES_DIR / "bootstrap_education_premium.csv", index=False)


def export_figures(
    edu_s, af, reg_s, yr_s, gender_edu,
    stat_tests, ols_s, mf, ols_base,
    q_s, bootstrap_df, comparison, cv_df, fi, pred_frame,
) -> None:
    fig01_income_by_education(edu_s)
    fig02_income_distribution_by_education(af)
    fig03_income_by_region(reg_s)
    fig04_income_trend_over_time(yr_s)
    fig05_gender_income_gap(af)
    fig06_gender_gap_by_education(gender_edu)
    fig07_hypothesis_tests(stat_tests)
    fig08_ols_coefficients(ols_s)
    fig09_ols_diagnostics(mf, ols_base)
    fig10_education_premium_by_quantile(ols_s, q_s, bootstrap_df)
    fig11_model_comparison(comparison)
    fig12_cross_validation(cv_df)
    fig13_feature_importance(fi)
    fig14_actual_vs_predicted(pred_frame)


def print_summary(af, reg_s, stat_tests, comparison, ols_s, q_s, bootstrap_df) -> None:
    top  = reg_s.iloc[0]
    best = comparison.iloc[0]
    edu_p = float(ols_s.loc[ols_s["term"] == "education_level", "pct_effect"].iloc[0])
    q25   = float(q_s.loc[q_s["quantile"] == 0.25, "pct_premium"].iloc[0])
    q75   = float(q_s.loc[q_s["quantile"] == 0.75, "pct_premium"].iloc[0])
    bs_lo = np.percentile(bootstrap_df["pct_premium"], 2.5)
    bs_hi = np.percentile(bootstrap_df["pct_premium"], 97.5)

    print("=== Korea Income and Welfare — analysis summary ===")
    print(f"Records analyzed:            {len(af):,}")
    print(f"Top region (median income):  {top['region_label']} (${top['median_usd']:,.0f})")
    print(f"Best model:                  {best['model']} (holdout R² = {best['r2']:.3f})")
    print(f"OLS education premium:       {edu_p:.2f}% per step")
    print(f"Bootstrap 95% CI:            [{bs_lo:.2f}%, {bs_hi:.2f}%]")
    print(f"Quantile premium Q25 / Q75:  {q25:.2f}% / {q75:.2f}%")
    print("Figures saved:               outputs/figures/ (14 files)")
    print("Tables saved:                outputs/tables/ (13 files)")


def main() -> None:
    args = parse_args()
    ensure_directories()

    print("Loading data...")
    df = load_dataset(args.input_path)
    af = build_analysis_frame(df)
    mf = build_model_frame(af)

    print("Building summaries...")
    edu_s      = build_education_summary(af)
    reg_s      = build_region_summary(af)
    yr_s       = build_year_summary(af)
    gender_edu = build_gender_edu_summary(af)
    stat_tests = build_statistical_tests(af)

    print("Running ML models...")
    comparison, fi, best_params, pred_frame, cv_df, best_model, Xts, yts = run_models(mf)

    print("Running econometric models...")
    ols_base, ols_hc3, qr = run_econometric_models(mf)
    ols_s  = build_ols_summary(ols_hc3)
    diag   = build_econometric_diagnostics(ols_base)
    vif    = build_econometric_vif(mf)
    q_s    = build_quantile_summary(qr)

    print(f"Running bootstrap ({BOOTSTRAP_N} iterations)...")
    bootstrap_df = build_bootstrap_education_premium(mf, BOOTSTRAP_N)

    print("Exporting tables...")
    export_tables(
        edu_s, reg_s, yr_s, stat_tests,
        comparison, fi, best_params,
        ols_s, diag, vif, q_s, cv_df, bootstrap_df,
    )

    print("Generating figures...")
    export_figures(
        edu_s, af, reg_s, yr_s, gender_edu,
        stat_tests, ols_s, mf, ols_base,
        q_s, bootstrap_df, comparison, cv_df, fi, pred_frame,
    )

    print_summary(af, reg_s, stat_tests, comparison, ols_s, q_s, bootstrap_df)


if __name__ == "__main__":
    main()
