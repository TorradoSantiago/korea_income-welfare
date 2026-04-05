from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / 'Korea_Income_and_Welfare.csv'
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'

REGION_LABELS = {
    1: 'Seoul',
    2: 'Gyeonggi',
    3: 'South Gyeongsang',
    4: 'North Gyeongsang',
    5: 'South Chungcheong',
    6: 'Gangwon and North Chungcheong',
    7: 'Jeolla and Jeju',
}

EDUCATION_LABELS = {
    1: 'No formal education (under age 7)',
    2: 'No formal education (age 7+)',
    3: 'Elementary school',
    4: 'Middle school',
    5: 'High school',
    6: 'College',
    7: 'University degree',
    8: "Master's degree",
    9: 'Doctorate',
}

MARRIAGE_LABELS = {
    0: 'Unknown',
    1: 'Not applicable',
    2: 'Married',
    3: 'Separated due to widowhood',
    4: 'Separated',
    5: 'Single',
    6: 'Other',
    9: 'Unknown',
}

GENDER_LABELS = {
    1: 'Male',
    2: 'Female',
}

RELIGION_LABELS = {
    1: 'Has religion',
    2: 'No religion',
    9: 'Unknown',
}

NUMERIC_FEATURES = ['education_level', 'age', 'family_member', 'year']
CATEGORICAL_FEATURES = ['region_label', 'gender_label', 'marriage_label', 'religion_label']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate reusable outputs and econometric plus ML benchmarks for the Korea income and welfare project.'
    )
    parser.add_argument(
        '--input-path',
        type=Path,
        default=DEFAULT_INPUT,
        help='Path to the Korea income and welfare CSV file.',
    )
    return parser.parse_args()


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f'Input file was not found: {input_path}')

    dataframe = pd.read_csv(input_path)
    dataframe['region_label'] = dataframe['region'].map(REGION_LABELS)
    dataframe['education_label'] = dataframe['education_level'].map(EDUCATION_LABELS)
    dataframe['marriage_label'] = dataframe['marriage'].map(MARRIAGE_LABELS)
    dataframe['gender_label'] = dataframe['gender'].map(GENDER_LABELS)
    dataframe['religion_label'] = dataframe['religion'].map(RELIGION_LABELS)
    dataframe['income_usd'] = (dataframe['income'] * 1000) / 1100
    dataframe['age'] = 2018 - dataframe['year_born']
    return dataframe


def build_analysis_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    analysis_frame = dataframe.loc[
        dataframe['income_usd'].notna()
        & (dataframe['income_usd'] > 0)
        & dataframe['age'].between(18, 90)
        & dataframe['education_label'].notna()
    ].copy()

    analysis_frame['family_member'] = pd.to_numeric(analysis_frame['family_member'], errors='coerce')
    analysis_frame['year'] = pd.to_numeric(analysis_frame['year'], errors='coerce')
    analysis_frame['education_level'] = pd.to_numeric(analysis_frame['education_level'], errors='coerce')
    return analysis_frame


def trim_target_outliers(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    q1 = dataframe[target_column].quantile(0.25)
    q3 = dataframe[target_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return dataframe.loc[dataframe[target_column].between(lower_bound, upper_bound)].copy()


def build_model_frame(analysis_frame: pd.DataFrame) -> pd.DataFrame:
    model_frame = analysis_frame[
        [
            'income_usd',
            'education_level',
            'age',
            'family_member',
            'year',
            'region_label',
            'gender_label',
            'marriage_label',
            'religion_label',
        ]
    ].dropna().copy()

    model_frame = trim_target_outliers(model_frame, 'income_usd')
    model_frame['log_income_usd'] = np.log(model_frame['income_usd'])
    return model_frame


def build_education_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataframe.groupby('education_label', dropna=False)
        .agg(
            respondents=('id', 'count'),
            median_income_usd=('income_usd', 'median'),
            average_income_usd=('income_usd', 'mean'),
        )
        .reset_index()
    )

    summary['education_order'] = summary['education_label'].map(
        {label: code for code, label in EDUCATION_LABELS.items()}
    )
    return summary.sort_values('education_order').drop(columns='education_order')


def build_region_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.groupby('region_label', dropna=False)
        .agg(
            respondents=('id', 'count'),
            median_income_usd=('income_usd', 'median'),
            average_income_usd=('income_usd', 'mean'),
        )
        .reset_index()
        .sort_values('median_income_usd', ascending=False)
    )


def build_year_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.groupby('year', dropna=False)
        .agg(
            respondents=('id', 'count'),
            median_income_usd=('income_usd', 'median'),
            average_income_usd=('income_usd', 'mean'),
        )
        .reset_index()
        .sort_values('year')
    )


def get_linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                'numeric',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                'categorical',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('numeric', SimpleImputer(strategy='median'), NUMERIC_FEATURES),
            (
                'categorical',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def collapse_feature_name(feature_name: str) -> str:
    clean_name = feature_name.replace('numeric__', '').replace('categorical__', '')
    for base_feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if clean_name == base_feature or clean_name.startswith(f'{base_feature}_'):
            return base_feature
    return clean_name


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        'model': model_name,
        'mae_usd': mean_absolute_error(y_true, y_pred),
        'rmse_usd': mean_squared_error(y_true, y_pred) ** 0.5,
        'r2': r2_score(y_true, y_pred),
    }


def run_models(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = model_frame['income_usd']
    education_only = model_frame[['education_level']]
    multivariable_features = model_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X_train_edu, X_test_edu, y_train_edu, y_test_edu = train_test_split(
        education_only, target, test_size=0.3, random_state=42
    )
    education_only_model = LinearRegression()
    education_only_model.fit(X_train_edu, y_train_edu)
    education_only_predictions = education_only_model.predict(X_test_edu)

    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        multivariable_features, target, test_size=0.3, random_state=42
    )

    models: dict[str, Pipeline] = {
        'Elastic Net': Pipeline(
            steps=[
                ('preprocessor', get_linear_preprocessor()),
                ('model', ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=10000)),
            ]
        ),
        'Multivariable linear regression': Pipeline(
            steps=[
                ('preprocessor', get_linear_preprocessor()),
                ('model', LinearRegression()),
            ]
        ),
        'Gradient boosting': Pipeline(
            steps=[
                ('preprocessor', get_tree_preprocessor()),
                (
                    'model',
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        ),
        'Tuned random forest': Pipeline(
            steps=[
                ('preprocessor', get_tree_preprocessor()),
                (
                    'model',
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=16,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    metrics_rows = [evaluate_model('Education-only linear regression', y_test_edu, education_only_predictions)]
    best_model_name = 'Education-only linear regression'
    best_r2 = float(metrics_rows[0]['r2'])
    best_model: Pipeline | None = None

    for model_name, model in models.items():
        model.fit(X_train_multi, y_train_multi)
        predictions = model.predict(X_test_multi)
        metrics = evaluate_model(model_name, y_test_multi, predictions)
        metrics_rows.append(metrics)
        if metrics['r2'] > best_r2:
            best_r2 = float(metrics['r2'])
            best_model_name = model_name
            best_model = model

    model_comparison = pd.DataFrame(metrics_rows).sort_values('r2', ascending=False)

    if best_model is None:
        # The tuned random forest should dominate the education-only baseline, but keep a safe fallback.
        best_model = models['Tuned random forest']
        best_model.fit(X_train_multi, y_train_multi)
        best_model_name = 'Tuned random forest'

    best_estimator = best_model.named_steps['model']
    best_params = pd.DataFrame(
        [
            {'model': best_model_name, 'parameter': key, 'value': value}
            for key, value in best_estimator.get_params().items()
            if key in {'alpha', 'l1_ratio', 'learning_rate', 'max_depth', 'min_samples_leaf', 'n_estimators', 'subsample'}
        ]
    )

    if hasattr(best_estimator, 'feature_importances_'):
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        feature_importance = pd.DataFrame(
            {
                'feature': feature_names,
                'importance': best_estimator.feature_importances_,
            }
        )
        feature_importance['feature_group'] = feature_importance['feature'].map(collapse_feature_name)
        feature_importance = (
            feature_importance.groupby('feature_group', as_index=False)['importance']
            .sum()
            .sort_values('importance', ascending=False)
        )
    else:
        feature_importance = pd.DataFrame(columns=['feature_group', 'importance'])

    best_predictions = best_model.predict(X_test_multi)
    prediction_frame = pd.DataFrame(
        {
            'actual_income': y_test_multi,
            'predicted_income': best_predictions,
            'model': best_model_name,
        }
    ).reset_index(drop=True)

    return model_comparison, feature_importance, best_params, prediction_frame


def run_econometric_models(model_frame: pd.DataFrame):
    formula = (
        'log_income_usd ~ education_level + age + I(age ** 2) + family_member + year + '
        'C(region_label) + C(gender_label) + C(marriage_label) + C(religion_label)'
    )
    ols_model = smf.ols(formula, data=model_frame).fit(cov_type='HC3')

    quantile_models: dict[float, object] = {}
    for quantile in (0.25, 0.5, 0.75):
        quantile_models[quantile] = smf.quantreg(formula, data=model_frame).fit(q=quantile, max_iter=2000)

    return ols_model, quantile_models


def build_ols_summary(ols_model) -> pd.DataFrame:
    key_terms = {
        'education_level': 'Education step premium',
        'age': 'Age effect',
        'I(age ** 2)': 'Age squared',
        'family_member': 'Household size effect',
        'year': 'Survey year trend',
    }

    rows: list[dict[str, float | str]] = []
    for term, label in key_terms.items():
        coefficient = float(ols_model.params[term])
        conf_int = ols_model.conf_int().loc[term]
        rows.append(
            {
                'term': term,
                'label': label,
                'coefficient': coefficient,
                'std_error': float(ols_model.bse[term]),
                'p_value': float(ols_model.pvalues[term]),
                'ci_low': float(conf_int.iloc[0]),
                'ci_high': float(conf_int.iloc[1]),
                'approx_pct_effect': (np.exp(coefficient) - 1) * 100,
            }
        )

    return pd.DataFrame(rows)


def build_quantile_summary(quantile_models: dict[float, object]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for quantile, model in quantile_models.items():
        conf_int = model.conf_int().loc['education_level']
        coefficient = float(model.params['education_level'])
        rows.append(
            {
                'model': f'Quantile regression ({int(quantile * 100)}th percentile)',
                'quantile': quantile,
                'education_coefficient': coefficient,
                'education_ci_low': float(conf_int.iloc[0]),
                'education_ci_high': float(conf_int.iloc[1]),
                'education_pct_premium': (np.exp(coefficient) - 1) * 100,
                'family_member_coefficient': float(model.params['family_member']),
                'year_coefficient': float(model.params['year']),
                'pseudo_r2': float(model.prsquared),
            }
        )

    return pd.DataFrame(rows).sort_values('quantile')


def export_tables(
    education_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_params: pd.DataFrame,
    ols_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
) -> None:
    education_summary.to_csv(TABLES_DIR / 'education_income_summary.csv', index=False)
    region_summary.to_csv(TABLES_DIR / 'region_income_summary.csv', index=False)
    year_summary.to_csv(TABLES_DIR / 'year_income_summary.csv', index=False)
    model_comparison.to_csv(TABLES_DIR / 'model_comparison.csv', index=False)
    feature_importance.to_csv(TABLES_DIR / 'feature_importance.csv', index=False)
    best_params.to_csv(TABLES_DIR / 'ml_best_params.csv', index=False)
    ols_summary.to_csv(TABLES_DIR / 'econometric_ols_summary.csv', index=False)
    quantile_summary.to_csv(TABLES_DIR / 'quantile_regression_summary.csv', index=False)


def export_figures(
    education_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    ols_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> None:
    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=education_summary,
        x='education_label',
        y='average_income_usd',
        color='#145da0',
    )
    plt.title('Average income by education level')
    plt.xlabel('Education level')
    plt.ylabel('Average income (USD)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'income_by_education.png', dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=region_summary,
        x='region_label',
        y='median_income_usd',
        color='#c9772b',
    )
    plt.title('Median income by region')
    plt.xlabel('Region')
    plt.ylabel('Median income (USD)')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'median_income_by_region.png', dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=year_summary,
        x='year',
        y='median_income_usd',
        marker='o',
        linewidth=2.2,
        color='#0f7b6c',
    )
    plt.title('Median income trend over time')
    plt.xlabel('Year')
    plt.ylabel('Median income (USD)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'median_income_over_time.png', dpi=150)
    plt.close()

    top_features = feature_importance.head(10).copy()
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=top_features,
        x='importance',
        y='feature_group',
        color='#7a4cc2',
    )
    plt.title('Top feature groups in the tuned random forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature group')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_feature_importance.png', dpi=150)
    plt.close()

    education_effect = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        'model': 'OLS mean effect',
                        'premium_pct': float(
                            ols_summary.loc[ols_summary['term'] == 'education_level', 'approx_pct_effect'].iloc[0]
                        ),
                        'ci_low_pct': (
                            np.exp(float(ols_summary.loc[ols_summary['term'] == 'education_level', 'ci_low'].iloc[0])) - 1
                        ) * 100,
                        'ci_high_pct': (
                            np.exp(float(ols_summary.loc[ols_summary['term'] == 'education_level', 'ci_high'].iloc[0])) - 1
                        ) * 100,
                    }
                ]
            ),
            quantile_summary.rename(columns={'education_pct_premium': 'premium_pct'})[
                ['model', 'premium_pct', 'education_ci_low', 'education_ci_high']
            ].assign(
                ci_low_pct=lambda frame: (np.exp(frame['education_ci_low']) - 1) * 100,
                ci_high_pct=lambda frame: (np.exp(frame['education_ci_high']) - 1) * 100,
            )[['model', 'premium_pct', 'ci_low_pct', 'ci_high_pct']],
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(11, 6))
    x_positions = np.arange(len(education_effect))
    errors = np.vstack(
        [
            education_effect['premium_pct'] - education_effect['ci_low_pct'],
            education_effect['ci_high_pct'] - education_effect['premium_pct'],
        ]
    )
    plt.errorbar(
        x_positions,
        education_effect['premium_pct'],
        yerr=errors,
        fmt='o',
        capsize=5,
        color='#145da0',
        linewidth=1.8,
    )
    plt.xticks(x_positions, education_effect['model'], rotation=20, ha='right')
    plt.ylabel('Income premium per education step (%)')
    plt.title('Education premium across mean and quantile regressions')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'education_premium_by_quantile.png', dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(data=model_comparison, x='model', y='r2', color='#2e8b57')
    plt.title('Out-of-sample model comparison')
    plt.xlabel('Model')
    plt.ylabel('R-squared')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_performance_r2.png', dpi=150)
    plt.close()

    plt.figure(figsize=(7, 7))
    plot_frame = prediction_frame.copy().sort_values('actual_income')
    sns.scatterplot(data=plot_frame, x='actual_income', y='predicted_income', s=18, alpha=0.45, color='#1f4b6e')
    min_value = min(plot_frame['actual_income'].min(), plot_frame['predicted_income'].min())
    max_value = max(plot_frame['actual_income'].max(), plot_frame['predicted_income'].max())
    plt.plot([min_value, max_value], [min_value, max_value], color='#b94a48', linewidth=1.5)
    plt.title('Actual vs predicted income in the tuned random forest')
    plt.xlabel('Actual income (USD)')
    plt.ylabel('Predicted income (USD)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'actual_vs_predicted_income.png', dpi=150)
    plt.close()


def print_summary(
    analysis_frame: pd.DataFrame,
    region_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    ols_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
) -> None:
    top_region = region_summary.iloc[0]
    strongest_model = model_comparison.iloc[0]
    ols_education_premium = float(
        ols_summary.loc[ols_summary['term'] == 'education_level', 'approx_pct_effect'].iloc[0]
    )
    low_quantile = float(quantile_summary.loc[quantile_summary['quantile'] == 0.25, 'education_pct_premium'].iloc[0])
    high_quantile = float(quantile_summary.loc[quantile_summary['quantile'] == 0.75, 'education_pct_premium'].iloc[0])

    print('=== Korea income and welfare summary ===')
    print(f'Records analyzed: {len(analysis_frame):,}')
    print(
        'Top region by median income: '
        f"{top_region['region_label']} ({top_region['median_income_usd']:.2f} USD)"
    )
    print(
        'Best benchmark model: '
        f"{strongest_model['model']} (R^2 = {strongest_model['r2']:.3f})"
    )
    print(f'Average education premium per step (OLS): {ols_education_premium:.2f}%')
    print(f'Education premium at the 25th percentile: {low_quantile:.2f}%')
    print(f'Education premium at the 75th percentile: {high_quantile:.2f}%')


def main() -> None:
    args = parse_args()
    ensure_directories()
    dataset = load_dataset(args.input_path)
    analysis_frame = build_analysis_frame(dataset)
    model_frame = build_model_frame(analysis_frame)

    education_summary = build_education_summary(analysis_frame)
    region_summary = build_region_summary(analysis_frame)
    year_summary = build_year_summary(analysis_frame)
    model_comparison, feature_importance, best_params, prediction_frame = run_models(model_frame)
    ols_model, quantile_models = run_econometric_models(model_frame)
    ols_summary = build_ols_summary(ols_model)
    quantile_summary = build_quantile_summary(quantile_models)

    export_tables(
        education_summary,
        region_summary,
        year_summary,
        model_comparison,
        feature_importance,
        best_params,
        ols_summary,
        quantile_summary,
    )
    export_figures(
        education_summary,
        region_summary,
        year_summary,
        model_comparison,
        feature_importance,
        ols_summary,
        quantile_summary,
        prediction_frame,
    )
    print_summary(
        analysis_frame,
        region_summary,
        model_comparison,
        ols_summary,
        quantile_summary,
    )


if __name__ == '__main__':
    main()
