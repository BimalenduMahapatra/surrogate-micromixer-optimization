#!/usr/bin/env python3
"""
02_model_validation_and_shap.py

Numerical validation, cross-validation, GP uncertainty, and TreeSHAP export
for the fitted micromixer surrogate models.

Prerequisite:
    Run 01_data_and_model_pipeline.py first.

This script generates CSV tables only. It does not generate graphs or images.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import shap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42
FEATURE_COLUMNS = ["A_h", "sin_phi", "cos_phi", "lambda_um"]
TARGET_COLUMNS = ["MI", "dP"]
MODEL_NAMES = ["Linear", "Polynomial", "Gaussian Process", "XGBoost"]

# Five-fold refitting uses three GP restarts (mirrors the master pipeline).
CV_GP_RESTARTS = 3


def slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def compute_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]

    if true.size < 2:
        raise ValueError("At least two finite observations are required.")

    absolute_error = np.abs(pred - true)
    relative_error = (
        100.0
        * absolute_error
        / np.maximum(np.abs(true), 1e-15)
    )

    return {
        "R2": float(r2_score(true, pred)),
        "RMSE": float(mean_squared_error(true, pred) ** 0.5),
        "MAE": float(mean_absolute_error(true, pred)),
        "MaxAE": float(np.max(absolute_error)),
        "Bias": float(np.mean(pred - true)),
        "MedianAPE_pct": float(np.median(relative_error)),
    }


def gp_kernel(initial_noise: float) -> Any:
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(4),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=initial_noise,
            noise_level_bounds=(1e-10, 1e0),
        )
    )


def build_model(
    model_name: str,
    target: str,
    selected_parameters: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gp_restarts: int,
) -> Any:
    if model_name == "Linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if model_name == "Polynomial":
        parameters = selected_parameters["Polynomial"][target]
        return Pipeline(
            [
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=int(parameters["degree"]),
                        include_bias=False,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "model",
                    Ridge(alpha=float(parameters["alpha"])),
                ),
            ]
        )

    if model_name == "Gaussian Process":
        parameters = selected_parameters["Gaussian Process"][target]
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=gp_kernel(
                            float(parameters["noise_init"])
                        ),
                        normalize_y=True,
                        optimizer="fmin_l_bfgs_b",
                        n_restarts_optimizer=int(gp_restarts),
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    if model_name == "XGBoost":
        parameters = dict(selected_parameters["XGBoost"][target])
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            **parameters,
        )

    raise KeyError(f"Unknown model: {model_name}")


def fit_target_model(
    model: Any,
    target: str,
    data: pd.DataFrame,
) -> None:
    response = data["MI"] if target == "MI" else data["log_dP"]
    model.fit(data[FEATURE_COLUMNS], response)


def predict_target_model(
    model: Any,
    target: str,
    data: pd.DataFrame,
) -> np.ndarray:
    prediction = np.asarray(
        model.predict(data[FEATURE_COLUMNS]),
        dtype=float,
    )
    return prediction if target == "MI" else np.exp(prediction)


def load_model_collection(
    model_directory: Path,
) -> Dict[str, Dict[str, Any]]:
    collection: Dict[str, Dict[str, Any]] = {}

    for model_name in MODEL_NAMES:
        collection[model_name] = {}
        model_slug = slugify_model_name(model_name)

        for target in TARGET_COLUMNS:
            model_path = model_directory / f"{model_slug}_{target}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(model_path)

            with model_path.open("rb") as handle:
                collection[model_name][target] = pickle.load(handle)

    return collection


def model_predictions(
    models: Mapping[str, Mapping[str, Any]],
    data: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, target_models in models.items():
        predictions[model_name] = {}
        for target in TARGET_COLUMNS:
            predictions[model_name][target] = predict_target_model(
                target_models[target],
                target,
                data,
            )

    return predictions


def evaluate_same_fit_models(
    same_fit_models: Mapping[str, Mapping[str, Any]],
    internal_test: pd.DataFrame,
    external: pd.DataFrame,
    metrics_directory: Path,
) -> None:
    internal_predictions = model_predictions(
        same_fit_models,
        internal_test,
    )
    external_predictions = model_predictions(
        same_fit_models,
        external,
    )

    metric_rows: List[Dict[str, Any]] = []
    internal_export = internal_test.copy()
    external_export = external.copy()

    for model_name in MODEL_NAMES:
        model_slug = slugify_model_name(model_name)

        for target in TARGET_COLUMNS:
            internal_prediction = internal_predictions[model_name][target]
            external_prediction = external_predictions[model_name][target]

            internal_export[
                f"{model_slug}_{target}_pred"
            ] = internal_prediction
            external_export[
                f"{model_slug}_{target}_pred"
            ] = external_prediction

            metric_rows.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "Training_Set": "Internal_Train_119",
                    "Evaluation_Set": "Internal_Test_25",
                    **compute_metrics(
                        internal_test[target],
                        internal_prediction,
                    ),
                }
            )
            metric_rows.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "Training_Set": "Internal_Train_119",
                    "Evaluation_Set": "External_Holdout_49",
                    **compute_metrics(
                        external[target],
                        external_prediction,
                    ),
                }
            )

    pd.DataFrame(metric_rows).to_csv(
        metrics_directory / "same_fit_model_metrics.csv",
        index=False,
    )
    internal_export.to_csv(
        metrics_directory / "same_fit_internal_test_predictions.csv",
        index=False,
    )
    external_export.to_csv(
        metrics_directory / "same_fit_external_predictions.csv",
        index=False,
    )


def evaluate_final_refit_models(
    final_models: Mapping[str, Mapping[str, Any]],
    external: pd.DataFrame,
    metrics_directory: Path,
) -> pd.DataFrame:
    predictions = model_predictions(final_models, external)
    export = external.copy()
    metric_rows: List[Dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        model_slug = slugify_model_name(model_name)

        for target in TARGET_COLUMNS:
            prediction = predictions[model_name][target]
            truth = external[target].to_numpy(dtype=float)

            export[f"{model_slug}_{target}_pred"] = prediction
            export[
                f"{model_slug}_{target}_signed_error"
            ] = prediction - truth
            export[
                f"{model_slug}_{target}_abs_error"
            ] = np.abs(prediction - truth)

            metric_rows.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "Training_Set": "All_169_Primary_Cases",
                    "Evaluation_Set": "External_Holdout_49",
                    **compute_metrics(truth, prediction),
                }
            )

    pd.DataFrame(metric_rows).to_csv(
        metrics_directory / "final_refit_external_metrics.csv",
        index=False,
    )
    export.to_csv(
        metrics_directory / "final_refit_external_predictions.csv",
        index=False,
    )

    return export


def run_five_fold_cross_validation(
    primary: pd.DataFrame,
    selected_parameters: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metrics_directory: Path,
) -> None:
    sinusoidal = (
        primary.loc[primary["geometry_class"].eq("sinusoidal")]
        .copy()
        .reset_index(drop=True)
    )
    flat_reference = primary.loc[
        primary["geometry_class"].eq("flat_reference")
    ].copy()

    cross_validator = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    records: List[Dict[str, Any]] = []

    for fold_number, (
        training_indices,
        validation_indices,
    ) in enumerate(cross_validator.split(sinusoidal), start=1):
        fold_training = pd.concat(
            [
                flat_reference,
                sinusoidal.iloc[training_indices],
            ],
            ignore_index=True,
        )
        fold_validation = sinusoidal.iloc[
            validation_indices
        ].copy()

        for model_name in MODEL_NAMES:
            for target in TARGET_COLUMNS:
                model = build_model(
                    model_name,
                    target,
                    selected_parameters,
                    gp_restarts=CV_GP_RESTARTS,
                )
                fit_target_model(
                    model,
                    target,
                    fold_training,
                )
                prediction = predict_target_model(
                    model,
                    target,
                    fold_validation,
                )

                records.append(
                    {
                        "Fold": fold_number,
                        "Model": model_name,
                        "Target": target,
                        "Training_N": len(fold_training),
                        "Validation_N": len(fold_validation),
                        **compute_metrics(
                            fold_validation[target],
                            prediction,
                        ),
                    }
                )

    all_folds = pd.DataFrame(records)
    all_folds.to_csv(
        metrics_directory / "five_fold_cv_all_folds.csv",
        index=False,
    )

    summary = (
        all_folds.groupby(["Model", "Target"])[
            ["R2", "RMSE", "MAE", "MaxAE", "Bias", "MedianAPE_pct"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part)
        )
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(
        metrics_directory / "five_fold_cv_summary.csv",
        index=False,
    )


def export_gp_uncertainty(
    final_models: Mapping[str, Mapping[str, Any]],
    external: pd.DataFrame,
    metrics_directory: Path,
) -> None:
    mi_pipeline = final_models["Gaussian Process"]["MI"]
    pressure_pipeline = final_models["Gaussian Process"]["dP"]

    mi_scaler = mi_pipeline.named_steps["scaler"]
    mi_model = mi_pipeline.named_steps["model"]
    pressure_scaler = pressure_pipeline.named_steps["scaler"]
    pressure_model = pressure_pipeline.named_steps["model"]

    mi_features = mi_scaler.transform(external[FEATURE_COLUMNS])
    pressure_features = pressure_scaler.transform(
        external[FEATURE_COLUMNS]
    )

    mi_mean, mi_standard_deviation = mi_model.predict(
        mi_features,
        return_std=True,
    )
    (
        log_pressure_mean,
        log_pressure_standard_deviation,
    ) = pressure_model.predict(
        pressure_features,
        return_std=True,
    )

    uncertainty = external[
        [
            "case",
            "A_h",
            "phi_deg",
            "n_waves",
            "lambda_um",
            "MI",
            "dP",
        ]
    ].copy()

    uncertainty["MI_pred_gp"] = mi_mean
    uncertainty["MI_std_gp"] = mi_standard_deviation
    uncertainty["MI_lower95"] = (
        mi_mean - 1.96 * mi_standard_deviation
    )
    uncertainty["MI_upper95"] = (
        mi_mean + 1.96 * mi_standard_deviation
    )

    uncertainty["log_dP_pred_gp"] = log_pressure_mean
    uncertainty[
        "log_dP_std_gp"
    ] = log_pressure_standard_deviation
    uncertainty["dP_pred_gp_Pa"] = np.exp(log_pressure_mean)
    uncertainty["dP_lower95_Pa"] = np.exp(
        log_pressure_mean
        - 1.96 * log_pressure_standard_deviation
    )
    uncertainty["dP_upper95_Pa"] = np.exp(
        log_pressure_mean
        + 1.96 * log_pressure_standard_deviation
    )

    uncertainty["MI_inside95"] = (
        (uncertainty["MI"] >= uncertainty["MI_lower95"])
        & (uncertainty["MI"] <= uncertainty["MI_upper95"])
    )
    uncertainty["dP_inside95"] = (
        (uncertainty["dP"] >= uncertainty["dP_lower95_Pa"])
        & (uncertainty["dP"] <= uncertainty["dP_upper95_Pa"])
    )

    uncertainty.to_csv(
        metrics_directory / "gp_external_uncertainty.csv",
        index=False,
    )

    coverage = pd.DataFrame(
        [
            {
                "Target": "MI",
                "Nominal_Interval": "95%",
                "Empirical_Coverage": float(
                    uncertainty["MI_inside95"].mean()
                ),
            },
            {
                "Target": "dP",
                "Nominal_Interval": "95%",
                "Empirical_Coverage": float(
                    uncertainty["dP_inside95"].mean()
                ),
            },
        ]
    )
    coverage.to_csv(
        metrics_directory / "gp_external_interval_coverage.csv",
        index=False,
    )


def export_shap_values(
    final_models: Mapping[str, Mapping[str, Any]],
    primary: pd.DataFrame,
    shap_directory: Path,
) -> None:
    sinusoidal = (
        primary.loc[primary["geometry_class"].eq("sinusoidal")]
        .copy()
        .reset_index(drop=True)
    )
    features = sinusoidal[FEATURE_COLUMNS]
    case_ids = sinusoidal["case"].reset_index(drop=True)

    raw_importance_records: List[Dict[str, Any]] = []
    grouped_importance_records: List[Dict[str, Any]] = []

    for target, output_name in [("MI", "MI"), ("dP", "log_dP")]:
        model = final_models["XGBoost"][target]
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(
            explainer.shap_values(features),
            dtype=float,
        )

        export = pd.concat(
            [
                sinusoidal[
                    [
                        "case",
                        "A_h",
                        "phi_deg",
                        "n_waves",
                        "lambda_um",
                    ]
                ],
                features.reset_index(drop=True),
                pd.DataFrame(
                    shap_values,
                    columns=[
                        f"SHAP_{feature}"
                        for feature in FEATURE_COLUMNS
                    ],
                ),
            ],
            axis=1,
        )
        export["Model_Output"] = output_name
        export.to_csv(
            shap_directory / f"shap_values_{output_name}.csv",
            index=False,
        )

        mean_absolute_shap = np.abs(shap_values).mean(axis=0)
        total = float(mean_absolute_shap.sum())

        feature_importance = dict(
            zip(FEATURE_COLUMNS, mean_absolute_shap)
        )

        for feature, importance in feature_importance.items():
            raw_importance_records.append(
                {
                    "Model_Output": output_name,
                    "Feature": feature,
                    "Mean_Absolute_SHAP": float(importance),
                    "Relative_Importance_Pct": (
                        float(100.0 * importance / total)
                        if total > 0
                        else np.nan
                    ),
                }
            )

        grouped = {
            "A_h": feature_importance["A_h"],
            "phase_offset": (
                feature_importance["sin_phi"]
                + feature_importance["cos_phi"]
            ),
            "lambda_um": feature_importance["lambda_um"],
        }
        grouped_total = float(sum(grouped.values()))

        for feature_group, importance in grouped.items():
            grouped_importance_records.append(
                {
                    "Model_Output": output_name,
                    "Feature_Group": feature_group,
                    "Mean_Absolute_SHAP": float(importance),
                    "Relative_Importance_Pct": (
                        float(
                            100.0
                            * importance
                            / grouped_total
                        )
                        if grouped_total > 0
                        else np.nan
                    ),
                }
            )

    pd.DataFrame(raw_importance_records).to_csv(
        shap_directory / "shap_global_importance_raw_features.csv",
        index=False,
    )
    pd.DataFrame(grouped_importance_records).to_csv(
        shap_directory / "shap_global_importance_grouped.csv",
        index=False,
    )


def run_analysis(output_directory: Path) -> None:
    data_directory = output_directory / "data"
    configuration_directory = output_directory / "config"
    model_directory = output_directory / "models"
    metrics_directory = output_directory / "metrics"
    shap_directory = output_directory / "shap"

    metrics_directory.mkdir(parents=True, exist_ok=True)
    shap_directory.mkdir(parents=True, exist_ok=True)

    required_data_files = {
        "primary": data_directory / "primary_processed_169.csv",
        "external": data_directory / "external_processed_49.csv",
        "training": data_directory / "internal_train_119.csv",
        "test": data_directory / "internal_test_25.csv",
    }

    for path in required_data_files.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prerequisite file: {path}. "
                "Run 01_data_and_model_pipeline.py first."
            )

    parameter_path = (
        configuration_directory / "selected_hyperparameters.json"
    )
    if not parameter_path.exists():
        raise FileNotFoundError(parameter_path)

    primary = pd.read_csv(required_data_files["primary"])
    external = pd.read_csv(required_data_files["external"])
    internal_test = pd.read_csv(required_data_files["test"])

    selected_parameters = json.loads(
        parameter_path.read_text(encoding="utf-8")
    )

    same_fit_models = load_model_collection(
        model_directory / "same_fit"
    )
    final_models = load_model_collection(
        model_directory / "final_refit"
    )

    evaluate_same_fit_models(
        same_fit_models,
        internal_test,
        external,
        metrics_directory,
    )
    evaluate_final_refit_models(
        final_models,
        external,
        metrics_directory,
    )
    run_five_fold_cross_validation(
        primary,
        selected_parameters,
        metrics_directory,
    )
    export_gp_uncertainty(
        final_models,
        external,
        metrics_directory,
    )
    export_shap_values(
        final_models,
        primary,
        shap_directory,
    )

    print("Validation, uncertainty, and SHAP exports completed.")
    print(f"Metrics: {metrics_directory}")
    print(f"SHAP data: {shap_directory}")
    print("Next: run 03_gp_nsga2_optimization.py.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate micromixer models and export SHAP data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory created by script 01.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_analysis(arguments.output_dir.expanduser().resolve())
