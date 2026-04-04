from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "Korea_Income_and_Welfare.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

REGION_LABELS = {
    1: "Seoul",
    2: "Gyeonggi",
    3: "South Gyeongsang",
    4: "North Gyeongsang",
    5: "South Chungcheong",
    6: "Gangwon and North Chungcheong",
    7: "Jeolla and Jeju",
}

EDUCATION_LABELS = {
    1: "No formal education (under age 7)",
    2: "No formal education (age 7+)",
    3: "Elementary school",
    4: "Middle school",
    5: "High school",
    6: "College",
    7: "University degree",
    8: "Master's degree",
    9: "Doctorate",
}

MARRIAGE_LABELS = {
    0: "Unknown",
    1: "Not applicable",
    2: "Married",
    3: "Separated due to widowhood",
    4: "Separated",
    5: "Single",
    6: "Other",
    9: "Unknown",
}

GENDER_LABELS = {
    1: "Male",
    2: "Female",
}

RELIGION_LABELS = {
    1: "Has religion",
    2: "No religion",
    9: "Unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reusable outputs and benchmarks for the Korea income and welfare project."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Korea income and welfare CSV file.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    dataframe["region_label"] = dataframe["region"].map(REGION_LABELS)
    dataframe["education_label"] = dataframe["education_level"].map(EDUCATION_LABELS)
    dataframe["marriage_label"] = dataframe["marriage"].map(MARRIAGE_LABELS)
    dataframe["gender_label"] = dataframe["gender"].map(GENDER_LABELS)
    dataframe["religion_label"] = dataframe["religion"].map(RELIGION_LABELS)
    dataframe["income_usd"] = (dataframe["income"] * 1000) / 1100
    dataframe["age"] = 2018 - dataframe["year_born"]
    return dataframe


def build_analysis_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    analysis_frame = dataframe.loc[
        dataframe["income_usd"].notna()
        & (dataframe["income_usd"] > 0)
        & dataframe["age"].between(18, 90)
        & dataframe["education_label"].notna()
    ].copy()

    analysis_frame["family_member"] = pd.to_numeric(analysis_frame["family_member"], errors="coerce")
    analysis_frame["year"] = pd.to_numeric(analysis_frame["year"], errors="coerce")
    analysis_frame["education_level"] = pd.to_numeric(analysis_frame["education_level"], errors="coerce")
    return analysis_frame


def trim_target_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    q1 = dataframe["income_usd"].quantile(0.25)
    q3 = dataframe["income_usd"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return dataframe.loc[dataframe["income_usd"].between(lower_bound, upper_bound)].copy()


def build_education_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataframe.groupby("education_label", dropna=False)
        .agg(
            respondents=("id", "count"),
            median_income_usd=("income_usd", "median"),
            average_income_usd=("income_usd", "mean"),
        )
        .reset_index()
    )

    summary["education_order"] = summary["education_label"].map(
        {label: code for code, label in EDUCATION_LABELS.items()}
    )
    summary = summary.sort_values("education_order").drop(columns="education_order")
    return summary


def build_region_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.groupby("region_label", dropna=False)
        .agg(
            respondents=("id", "count"),
            median_income_usd=("income_usd", "median"),
            average_income_usd=("income_usd", "mean"),
        )
        .reset_index()
        .sort_values("median_income_usd", ascending=False)
    )


def build_year_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.groupby("year", dropna=False)
        .agg(
            respondents=("id", "count"),
            median_income_usd=("income_usd", "median"),
            average_income_usd=("income_usd", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | str]:
    return {
        "model": model_name,
        "mae_usd": mean_absolute_error(y_true, y_pred),
        "rmse_usd": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def run_models(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_frame = trim_target_outliers(
        dataframe[
            [
                "income_usd",
                "education_level",
                "age",
                "family_member",
                "year",
                "region_label",
                "gender_label",
                "marriage_label",
                "religion_label",
            ]
        ].dropna()
    )

    target = model_frame["income_usd"]
    education_only = model_frame[["education_level"]]
    multivariable_features = model_frame[
        ["education_level", "age", "family_member", "year", "region_label", "gender_label", "marriage_label", "religion_label"]
    ]

    X_train_edu, X_test_edu, y_train_edu, y_test_edu = train_test_split(
        education_only, target, test_size=0.3, random_state=42
    )
    education_only_model = LinearRegression()
    education_only_model.fit(X_train_edu, y_train_edu)
    education_only_predictions = education_only_model.predict(X_test_edu)

    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        multivariable_features, target, test_size=0.3, random_state=42
    )

    numeric_features = ["education_level", "age", "family_member", "year"]
    categorical_features = ["region_label", "gender_label", "marriage_label", "religion_label"]

    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    forest_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    multivariable_linear_model = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            ("model", LinearRegression()),
        ]
    )
    multivariable_linear_model.fit(X_train_multi, y_train_multi)
    multivariable_linear_predictions = multivariable_linear_model.predict(X_test_multi)

    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", forest_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    random_forest_model.fit(X_train_multi, y_train_multi)
    random_forest_predictions = random_forest_model.predict(X_test_multi)

    model_comparison = pd.DataFrame(
        [
            evaluate_model("Education-only linear regression", y_test_edu, education_only_predictions),
            evaluate_model("Multivariable linear regression", y_test_multi, multivariable_linear_predictions),
            evaluate_model("Multivariable random forest", y_test_multi, random_forest_predictions),
        ]
    ).sort_values("r2", ascending=False)

    feature_names = random_forest_model.named_steps["preprocessor"].get_feature_names_out()
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": random_forest_model.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance["feature"] = feature_importance["feature"].str.replace("numeric__", "", regex=False)
    feature_importance["feature"] = feature_importance["feature"].str.replace("categorical__", "", regex=False)

    return model_comparison, feature_importance


def export_tables(
    education_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    education_summary.to_csv(TABLES_DIR / "education_income_summary.csv", index=False)
    region_summary.to_csv(TABLES_DIR / "region_income_summary.csv", index=False)
    year_summary.to_csv(TABLES_DIR / "year_income_summary.csv", index=False)
    model_comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    feature_importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)


def export_figures(
    education_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=education_summary,
        x="education_label",
        y="average_income_usd",
        color="#145da0",
    )
    plt.title("Average income by education level")
    plt.xlabel("Education level")
    plt.ylabel("Average income (USD)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "income_by_education.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=region_summary,
        x="region_label",
        y="median_income_usd",
        color="#c9772b",
    )
    plt.title("Median income by region")
    plt.xlabel("Region")
    plt.ylabel("Median income (USD)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "median_income_by_region.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=year_summary,
        x="year",
        y="median_income_usd",
        marker="o",
        linewidth=2.2,
        color="#0f7b6c",
    )
    plt.title("Median income trend over time")
    plt.xlabel("Year")
    plt.ylabel("Median income (USD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "median_income_over_time.png", dpi=150)
    plt.close()

    top_features = feature_importance.head(10).copy()
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=top_features,
        x="importance",
        y="feature",
        color="#7a4cc2",
    )
    plt.title("Top random-forest feature importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_feature_importance.png", dpi=150)
    plt.close()


def print_summary(
    analysis_frame: pd.DataFrame,
    education_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> None:
    top_education = education_summary.sort_values("average_income_usd", ascending=False).iloc[0]
    top_region = region_summary.iloc[0]
    strongest_model = model_comparison.iloc[0]

    print("=== Korea income and welfare summary ===")
    print(f"Records analyzed: {len(analysis_frame):,}")
    print(
        "Top education tier by average income: "
        f"{top_education['education_label']} ({top_education['average_income_usd']:.2f} USD)"
    )
    print(
        "Top region by median income: "
        f"{top_region['region_label']} ({top_region['median_income_usd']:.2f} USD)"
    )
    print(
        "Best benchmark model: "
        f"{strongest_model['model']} (R^2 = {strongest_model['r2']:.3f})"
    )


def main() -> None:
    args = parse_args()
    ensure_directories()
    dataset = load_dataset(args.input_path)
    analysis_frame = build_analysis_frame(dataset)

    education_summary = build_education_summary(analysis_frame)
    region_summary = build_region_summary(analysis_frame)
    year_summary = build_year_summary(analysis_frame)
    model_comparison, feature_importance = run_models(analysis_frame)

    export_tables(education_summary, region_summary, year_summary, model_comparison, feature_importance)
    export_figures(education_summary, region_summary, year_summary, feature_importance)
    print_summary(analysis_frame, education_summary, region_summary, model_comparison)


if __name__ == "__main__":
    main()
