#!/usr/bin/env python3
"""
01_data_and_model_pipeline.py

Data preparation and surrogate-model development for the sinusoidal
micromixer study.

This script:
1. loads the 169-case primary CFD dataset and 49-case external dataset;
2. creates the model features [A/h, sin(phi), cos(phi), lambda];
3. creates the deterministic 119/25/25 train/validation/test split by greedy
   maximin selection on the standardised sinusoidal feature space;
4. selects Polynomial Ridge, GPR, and XGBoost hyperparameters using validation
   RMSE (MAE as a tie-break);
5. fits and saves the same-fit models on the 119-case training set;
6. fits and saves the final-refit models on all 169 primary cases;
7. saves processed data, split files, fitted models, and configuration metadata.

The pressure drop in both CSVs is already expressed in Pa; the pressure targets
are modelled as log(dP) and reported in Pa. No graphs or manuscript figures are
generated.

Output layout (consumed by scripts 02, 03, and 04):
    <output-dir>/data/primary_processed_169.csv
    <output-dir>/data/external_processed_49.csv
    <output-dir>/data/internal_train_119.csv
    <output-dir>/data/internal_validation_25.csv
    <output-dir>/data/internal_test_25.csv
    <output-dir>/config/selected_hyperparameters.json
    <output-dir>/config/validation_hyperparameter_search.csv
    <output-dir>/models/same_fit/<model>_<target>.pkl
    <output-dir>/models/final_refit/<model>_<target>.pkl
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42
L_SIN_UM = 2400.0

FEATURE_COLUMNS = ["A_h", "sin_phi", "cos_phi", "lambda_um"]
TARGET_COLUMNS = ["MI", "dP"]
MODEL_NAMES = ["Linear", "Polynomial", "Gaussian Process", "XGBoost"]

# Fitting effort. Same-fit and final-refit use ten restarts; the validation
# hyperparameter search uses three (mirrors the master pipeline).
SAME_FIT_GP_RESTARTS = 10
FINAL_REFIT_GP_RESTARTS = 10
TUNING_GP_RESTARTS = 3


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
        100.0 * absolute_error / np.maximum(np.abs(true), 1e-15)
    )

    return {
        "R2": float(r2_score(true, pred)),
        "RMSE": float(mean_squared_error(true, pred) ** 0.5),
        "MAE": float(mean_absolute_error(true, pred)),
        "MaxAE": float(np.max(absolute_error)),
        "Bias": float(np.mean(pred - true)),
        "MedianAPE_pct": float(np.median(relative_error)),
    }


def standardise_column_names(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()

    aliases = {
        "case": ["case", "case_id", "case_folder", "folder"],
        "A_h": ["A_h", "A/H", "A_over_h", "A_by_h", "amplitude_ratio"],
        "phi_deg": ["phi_deg", "phi", "phase", "phase_deg"],
        "n_waves": ["n_waves", "n", "N", "waves", "wave_count"],
        "lambda_um": ["lambda_um", "lambda", "wavelength", "lambda_microns"],
        "MI": ["MI", "mixing_index", "Mixing_Index"],
        "dP": ["dP", "DeltaP", "deltaP", "delta_p", "pressure_drop"],
    }

    lookup = {str(column).strip().lower(): column for column in output.columns}

    for canonical, candidates in aliases.items():
        if canonical in output.columns:
            continue
        for candidate in candidates:
            source = lookup.get(candidate.lower())
            if source is not None:
                output.rename(columns={source: canonical}, inplace=True)
                break

    return output


def prepare_dataset(
    raw_data: pd.DataFrame,
    dataset_label: str,
) -> pd.DataFrame:
    output = standardise_column_names(raw_data)

    if "case" not in output.columns:
        output["case"] = [
            f"{dataset_label}_{index + 1:03d}"
            for index in range(len(output))
        ]

    required = ["A_h", "phi_deg", "n_waves", "lambda_um", "MI", "dP"]
    missing = [column for column in required if column not in output.columns]
    if missing:
        raise KeyError(f"{dataset_label}: missing required columns: {missing}")

    for column in required:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    if output[required].isna().any().any():
        raise ValueError(
            f"{dataset_label}: missing or non-numeric required values found."
        )

    if (output["A_h"] < 0).any():
        raise ValueError(f"{dataset_label}: negative A/h values found.")

    if (output["dP"] <= 0).any():
        raise ValueError(f"{dataset_label}: pressure drop must be positive.")

    flat_mask = np.isclose(
        output["A_h"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    sinusoidal_mask = ~flat_mask

    # Canonical numerical encoding of the straight-channel reference.
    output.loc[flat_mask, ["phi_deg", "n_waves", "lambda_um"]] = 0.0

    if (output.loc[sinusoidal_mask, "n_waves"] <= 0).any():
        raise ValueError(
            f"{dataset_label}: sinusoidal cases require n_waves > 0."
        )

    output.loc[sinusoidal_mask, "lambda_um"] = (
        L_SIN_UM / output.loc[sinusoidal_mask, "n_waves"]
    )
    output.loc[flat_mask, "lambda_um"] = 0.0

    output["phi_rad"] = np.deg2rad(output["phi_deg"])
    output["sin_phi"] = np.sin(output["phi_rad"])
    output["cos_phi"] = np.cos(output["phi_rad"])
    output["geometry_class"] = np.where(
        flat_mask,
        "flat_reference",
        "sinusoidal",
    )
    output["is_flat_reference"] = flat_mask
    output["log_dP"] = np.log(output["dP"])

    model_values = output[
        FEATURE_COLUMNS + ["MI", "dP", "log_dP"]
    ].to_numpy(dtype=float)
    if not np.isfinite(model_values).all():
        raise ValueError(f"{dataset_label}: non-finite model values found.")

    output.reset_index(drop=True, inplace=True)

    if dataset_label == "primary":
        if len(output) != 169:
            raise ValueError(
                f"Primary dataset must contain 169 cases; found {len(output)}."
            )
        if int(output["is_flat_reference"].sum()) != 1:
            raise ValueError(
                "Primary dataset must contain one A/h = 0 reference case."
            )
        if int((output["A_h"] > 0).sum()) != 168:
            raise ValueError(
                "Primary dataset must contain 168 sinusoidal cases."
            )

    if dataset_label == "external":
        if len(output) != 49:
            raise ValueError(
                f"External dataset must contain 49 cases; found {len(output)}."
            )
        if output["is_flat_reference"].any():
            raise ValueError(
                "External dataset must not contain the flat reference case."
            )

    return output


def greedy_maximin_indices(
    feature_matrix: np.ndarray,
    number_to_select: int,
) -> np.ndarray:
    matrix = np.asarray(feature_matrix, dtype=float)

    if number_to_select >= len(matrix):
        raise ValueError(
            "number_to_select must be smaller than the candidate count."
        )

    centroid = matrix.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(matrix - centroid, axis=1)))]
    minimum_distance = np.linalg.norm(matrix - matrix[selected[0]], axis=1)

    for _ in range(1, number_to_select):
        minimum_distance[selected] = -np.inf
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(matrix - matrix[next_index], axis=1),
        )

    return np.asarray(selected, dtype=int)


def create_internal_split(
    primary: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sinusoidal = (
        primary.loc[primary["geometry_class"].eq("sinusoidal")]
        .copy()
        .reset_index(drop=True)
    )
    flat_reference = primary.loc[
        primary["geometry_class"].eq("flat_reference")
    ].copy()

    scaled_geometry = StandardScaler().fit_transform(
        sinusoidal[FEATURE_COLUMNS]
    )
    holdout_indices = greedy_maximin_indices(
        scaled_geometry,
        number_to_select=50,
    )

    validation_indices = holdout_indices[::2][:25]
    test_indices = holdout_indices[1::2][:25]
    holdout_set = set(validation_indices.tolist() + test_indices.tolist())
    training_indices = [
        index
        for index in range(len(sinusoidal))
        if index not in holdout_set
    ]

    training = pd.concat(
        [flat_reference, sinusoidal.iloc[training_indices]],
        ignore_index=True,
    )
    validation = (
        sinusoidal.iloc[validation_indices].copy().reset_index(drop=True)
    )
    test = sinusoidal.iloc[test_indices].copy().reset_index(drop=True)

    if (len(training), len(validation), len(test)) != (119, 25, 25):
        raise RuntimeError("Unexpected internal split sizes.")

    if int(training["is_flat_reference"].sum()) != 1:
        raise RuntimeError("The flat reference case must be in the training set.")

    if validation["is_flat_reference"].any() or test[
        "is_flat_reference"
    ].any():
        raise RuntimeError(
            "The flat reference case must not be in validation or test."
        )

    return training, validation, test


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
                ("model", Ridge(alpha=float(parameters["alpha"]))),
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
                        kernel=gp_kernel(float(parameters["noise_init"])),
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


def fit_target_model(model: Any, target: str, data: pd.DataFrame) -> None:
    response = data["MI"] if target == "MI" else data["log_dP"]
    model.fit(data[FEATURE_COLUMNS], response)


def predict_target_model(
    model: Any,
    target: str,
    data: pd.DataFrame,
) -> np.ndarray:
    prediction = np.asarray(model.predict(data[FEATURE_COLUMNS]), dtype=float)
    return prediction if target == "MI" else np.exp(prediction)


def validation_record(
    model_name: str,
    target: str,
    parameters: Mapping[str, Any],
    truth: pd.Series,
    prediction: np.ndarray,
) -> Dict[str, Any]:
    return {
        "Model": model_name,
        "Target": target,
        "Parameters_JSON": json.dumps(parameters, sort_keys=True),
        **compute_metrics(truth, prediction),
    }


def select_hyperparameters(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], pd.DataFrame]:
    selected: Dict[str, Dict[str, Dict[str, Any]]] = {
        "Linear": {"MI": {}, "dP": {}},
        "Polynomial": {},
        "Gaussian Process": {},
        "XGBoost": {},
    }
    records: List[Dict[str, Any]] = []

    for target in TARGET_COLUMNS:
        truth = validation[target]

        # Linear baseline, evaluated on the validation set for completeness.
        linear = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        fit_target_model(linear, target, training)
        records.append(
            validation_record(
                "Linear",
                target,
                {},
                truth,
                predict_target_model(linear, target, validation),
            )
        )

        # Polynomial ridge search.
        polynomial_records: List[Dict[str, Any]] = []
        for degree, alpha in itertools.product(
            [2, 3],
            [1e-6, 1e-4, 1e-2, 1.0],
        ):
            parameters = {"degree": degree, "alpha": alpha}
            model = Pipeline(
                [
                    (
                        "polynomial",
                        PolynomialFeatures(degree=degree, include_bias=False),
                    ),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=alpha)),
                ]
            )
            fit_target_model(model, target, training)
            record = validation_record(
                "Polynomial",
                target,
                parameters,
                truth,
                predict_target_model(model, target, validation),
            )
            polynomial_records.append(record)
            records.append(record)

        best_polynomial = min(
            polynomial_records,
            key=lambda row: (row["RMSE"], row["MAE"]),
        )
        selected["Polynomial"][target] = json.loads(
            best_polynomial["Parameters_JSON"]
        )

        # Gaussian process initial-noise search.
        gp_records: List[Dict[str, Any]] = []
        for noise_init in [1e-8, 1e-6, 1e-4]:
            parameters = {"noise_init": noise_init}
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GaussianProcessRegressor(
                            kernel=gp_kernel(noise_init),
                            normalize_y=True,
                            optimizer="fmin_l_bfgs_b",
                            n_restarts_optimizer=TUNING_GP_RESTARTS,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            fit_target_model(model, target, training)
            record = validation_record(
                "Gaussian Process",
                target,
                parameters,
                truth,
                predict_target_model(model, target, validation),
            )
            gp_records.append(record)
            records.append(record)

        best_gp = min(gp_records, key=lambda row: (row["RMSE"], row["MAE"]))
        selected["Gaussian Process"][target] = json.loads(
            best_gp["Parameters_JSON"]
        )

        # XGBoost search.
        xgb_records: List[Dict[str, Any]] = []
        for (
            n_estimators,
            max_depth,
            learning_rate,
            min_child_weight,
        ) in itertools.product([300, 600], [2, 3], [0.02, 0.05], [1, 3]):
            parameters = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_child_weight": min_child_weight,
                "subsample": 0.9,
                "reg_lambda": 1.0,
            }
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
                colsample_bytree=1.0,
                reg_alpha=0.0,
                **parameters,
            )
            fit_target_model(model, target, training)
            record = validation_record(
                "XGBoost",
                target,
                parameters,
                truth,
                predict_target_model(model, target, validation),
            )
            xgb_records.append(record)
            records.append(record)

        best_xgb = min(xgb_records, key=lambda row: (row["RMSE"], row["MAE"]))
        selected["XGBoost"][target] = json.loads(best_xgb["Parameters_JSON"])

    validation_results = pd.DataFrame(records).sort_values(
        ["Target", "Model", "RMSE", "MAE"]
    )
    return selected, validation_results


def fit_and_save_model_collection(
    training_data: pd.DataFrame,
    selected_parameters: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gp_restarts: int,
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_NAMES:
        model_slug = slugify_model_name(model_name)
        for target in TARGET_COLUMNS:
            model = build_model(
                model_name,
                target,
                selected_parameters,
                gp_restarts=gp_restarts,
            )
            fit_target_model(model, target, training_data)
            with (destination / f"{model_slug}_{target}.pkl").open("wb") as handle:
                pickle.dump(model, handle)


def run(
    primary_csv: Path,
    external_csv: Path,
    output_directory: Path,
) -> None:
    data_directory = output_directory / "data"
    config_directory = output_directory / "config"
    model_directory = output_directory / "models"

    data_directory.mkdir(parents=True, exist_ok=True)
    config_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    print("--- Data ingestion and feature construction ---")
    primary = prepare_dataset(pd.read_csv(primary_csv), "primary")
    external = prepare_dataset(pd.read_csv(external_csv), "external")

    primary.to_csv(data_directory / "primary_processed_169.csv", index=False)
    external.to_csv(data_directory / "external_processed_49.csv", index=False)

    print(
        f"Primary: {len(primary)} cases "
        f"({int((primary['A_h'] > 0).sum())} sinusoidal + "
        f"{int(primary['is_flat_reference'].sum())} flat reference)."
    )
    print(f"External holdout: {len(external)} cases.")
    print(
        f"Primary dP range: {primary['dP'].min():.6g} to "
        f"{primary['dP'].max():.6g} Pa"
    )

    print("\n--- Deterministic 119/25/25 geometry-space split ---")
    training, validation, test = create_internal_split(primary)
    training.to_csv(data_directory / "internal_train_119.csv", index=False)
    validation.to_csv(
        data_directory / "internal_validation_25.csv", index=False
    )
    test.to_csv(data_directory / "internal_test_25.csv", index=False)

    print("\n--- Validation-set hyperparameter selection ---")
    selected_parameters, validation_results = select_hyperparameters(
        training, validation
    )
    (config_directory / "selected_hyperparameters.json").write_text(
        json.dumps(selected_parameters, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_results.to_csv(
        config_directory / "validation_hyperparameter_search.csv", index=False
    )
    print(json.dumps(selected_parameters, indent=2, sort_keys=True))

    print("\n--- Same-fit models (119-case internal training set) ---")
    fit_and_save_model_collection(
        training,
        selected_parameters,
        gp_restarts=SAME_FIT_GP_RESTARTS,
        destination=model_directory / "same_fit",
    )

    print("--- Final-refit models (all 169 primary cases) ---")
    fit_and_save_model_collection(
        primary,
        selected_parameters,
        gp_restarts=FINAL_REFIT_GP_RESTARTS,
        destination=model_directory / "final_refit",
    )

    print("\nData preparation and model fitting completed.")
    print(f"Processed data: {data_directory}")
    print(f"Configuration:  {config_directory}")
    print(f"Fitted models:  {model_directory}")
    print("Next: run 02_model_validation_and_shap.py.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare data and fit micromixer surrogate models."
    )
    parser.add_argument(
        "--primary-csv",
        type=Path,
        default=Path("primary_final.csv"),
        help="169-case primary CFD dataset (dP already in Pa).",
    )
    parser.add_argument(
        "--external-csv",
        type=Path,
        default=Path("external_final.csv"),
        help="49-case external holdout dataset (dP already in Pa).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for processed data, models, and configuration.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(
        primary_csv=arguments.primary_csv.expanduser().resolve(),
        external_csv=arguments.external_csv.expanduser().resolve(),
        output_directory=arguments.output_dir.expanduser().resolve(),
    )
