#!/usr/bin/env python3
"""
04_pareto_and_results_export.py

Pareto-front construction, compromise selection, and final numerical exports.

Prerequisites:
    Run 01_data_and_model_pipeline.py first.
    Run 03_gp_nsga2_optimization.py second.

This script:
- extracts the CFD Pareto front from the 168 sinusoidal primary cases;
- loads the central GP-NSGA-II Pareto front;
- calculates the CFD and GP geometric knees using common normalization bounds
  obtained from the union of the two fronts;
- calculates the equal-weight ideal-point compromise;
- calculates pressure-constrained optima at the reported pressure limits;
- exports numerical tables and a machine-readable summary.

No graphs or images are generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


# Pressure limits reported in the manuscript (Pa). The 500 Pa case was removed
# from the confirmation table, so it is excluded here to keep the exported
# constrained optima consistent with the reported results.
DEFAULT_PRESSURE_LIMITS_PA = [100, 150, 200, 300, 400]


def get_pareto_front(
    data: pd.DataFrame,
    mi_column: str,
    pressure_column: str,
) -> pd.DataFrame:
    ordered = data.sort_values(
        [pressure_column, mi_column],
        ascending=[True, False],
    ).copy()

    retained_indices: List[int] = []
    best_mixing_index = -np.inf

    for index, row in ordered.iterrows():
        current_mixing_index = float(row[mi_column])
        if current_mixing_index > best_mixing_index + 1e-12:
            retained_indices.append(index)
            best_mixing_index = current_mixing_index

    return (
        ordered.loc[retained_indices]
        .copy()
        .sort_values(pressure_column)
        .reset_index(drop=True)
    )


def common_objective_bounds(
    front_definitions: Iterable[
        Tuple[pd.DataFrame, str, str]
    ],
) -> Dict[str, float]:
    mixing_values: List[np.ndarray] = []
    pressure_values: List[np.ndarray] = []

    for data, mi_column, pressure_column in front_definitions:
        mixing_values.append(
            data[mi_column].to_numpy(dtype=float)
        )
        pressure_values.append(
            data[pressure_column].to_numpy(dtype=float)
        )

    all_mixing = np.concatenate(mixing_values)
    all_pressure = np.concatenate(pressure_values)

    return {
        "mi_min": float(np.min(all_mixing)),
        "mi_max": float(np.max(all_mixing)),
        "pressure_min": float(np.min(all_pressure)),
        "pressure_max": float(np.max(all_pressure)),
    }


def normalized_coordinates(
    data: pd.DataFrame,
    mi_column: str,
    pressure_column: str,
    bounds: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    mi_range = bounds["mi_max"] - bounds["mi_min"]
    pressure_range = (
        bounds["pressure_max"] - bounds["pressure_min"]
    )

    if mi_range <= 0 or pressure_range <= 0:
        raise ValueError(
            "Objective normalization requires non-zero ranges."
        )

    normalized_mixing = (
        data[mi_column].to_numpy(dtype=float)
        - bounds["mi_min"]
    ) / mi_range

    normalized_pressure = (
        data[pressure_column].to_numpy(dtype=float)
        - bounds["pressure_min"]
    ) / pressure_range

    return normalized_pressure, normalized_mixing


def geometric_knee(
    pareto_front: pd.DataFrame,
    mi_column: str,
    pressure_column: str,
    bounds: Mapping[str, float],
) -> pd.Series:
    if pareto_front.empty:
        raise ValueError("Cannot select a knee from an empty front.")

    if len(pareto_front) < 3:
        return pareto_front.iloc[0].copy()

    ordered = (
        pareto_front.sort_values(pressure_column)
        .reset_index(drop=True)
    )
    normalized_pressure, normalized_mixing = normalized_coordinates(
        ordered,
        mi_column,
        pressure_column,
        bounds,
    )

    start = np.array(
        [normalized_pressure[0], normalized_mixing[0]],
        dtype=float,
    )
    end = np.array(
        [normalized_pressure[-1], normalized_mixing[-1]],
        dtype=float,
    )
    chord = end - start
    chord_norm = np.linalg.norm(chord)

    if chord_norm <= 0:
        return ordered.iloc[0].copy()

    chord_unit = chord / chord_norm
    distances: List[float] = []

    for pressure_value, mixing_value in zip(
        normalized_pressure,
        normalized_mixing,
    ):
        point_vector = (
            np.array(
                [pressure_value, mixing_value],
                dtype=float,
            )
            - start
        )
        perpendicular = (
            point_vector
            - np.dot(point_vector, chord_unit)
            * chord_unit
        )
        distances.append(
            float(np.linalg.norm(perpendicular))
        )

    selected = ordered.iloc[int(np.argmax(distances))].copy()
    selected["geometric_knee_distance"] = float(
        np.max(distances)
    )
    return selected


def ideal_point_compromise(
    pareto_front: pd.DataFrame,
    mi_column: str,
    pressure_column: str,
    bounds: Mapping[str, float],
) -> pd.Series:
    normalized_pressure, normalized_mixing = normalized_coordinates(
        pareto_front,
        mi_column,
        pressure_column,
        bounds,
    )

    distance_to_ideal = np.sqrt(
        normalized_pressure**2
        + (1.0 - normalized_mixing) ** 2
    )

    selected = pareto_front.iloc[
        int(np.argmin(distance_to_ideal))
    ].copy()
    selected["distance_to_ideal"] = float(
        np.min(distance_to_ideal)
    )
    return selected


def selection_record(
    selection_name: str,
    evidence: str,
    row: pd.Series,
    mi_column: str,
    pressure_column: str,
) -> Dict[str, object]:
    return {
        "Selection": selection_name,
        "Evidence": evidence,
        "A_h": float(row["A_h"]),
        "phi_deg": float(row["phi_deg"]),
        "n_waves": int(round(float(row["n_waves"]))),
        "lambda_um": float(row["lambda_um"]),
        "MI_value": float(row[mi_column]),
        "dP_value_Pa": float(row[pressure_column]),
        "MI_pred_gp": float(
            row.get("MI_pred_gp", np.nan)
        ),
        "dP_pred_gp_Pa": float(
            row.get("dP_pred_gp", np.nan)
        ),
    }


def pressure_constrained_optima(
    gp_pareto: pd.DataFrame,
    pressure_limits: Iterable[float],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for pressure_limit in pressure_limits:
        feasible = gp_pareto.loc[
            gp_pareto["dP_pred_gp"]
            <= float(pressure_limit)
        ].copy()

        if feasible.empty:
            records.append(
                {
                    "pressure_limit_Pa": float(
                        pressure_limit
                    ),
                    "status": "no feasible candidate",
                }
            )
            continue

        selected = feasible.loc[
            feasible["MI_pred_gp"].idxmax()
        ]

        records.append(
            {
                "pressure_limit_Pa": float(
                    pressure_limit
                ),
                "status": "feasible",
                "A_h": float(selected["A_h"]),
                "phi_deg": float(
                    selected["phi_deg"]
                ),
                "n_waves": int(
                    round(float(selected["n_waves"]))
                ),
                "lambda_um": float(
                    selected["lambda_um"]
                ),
                "MI_pred_gp": float(
                    selected["MI_pred_gp"]
                ),
                "dP_pred_gp_Pa": float(
                    selected["dP_pred_gp"]
                ),
            }
        )

    return pd.DataFrame(records)


def run(
    output_directory: Path,
    pressure_limits: List[float],
) -> None:
    data_directory = output_directory / "data"
    optimization_directory = output_directory / "optimization"
    summary_directory = output_directory / "summary"
    summary_directory.mkdir(parents=True, exist_ok=True)

    primary_path = (
        data_directory / "primary_processed_169.csv"
    )
    gp_pareto_path = (
        optimization_directory
        / "nsga2_global_pareto_front.csv"
    )
    convergence_path = (
        optimization_directory
        / "nsga2_convergence_history.csv"
    )

    for path in [
        primary_path,
        gp_pareto_path,
        convergence_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prerequisite file: {path}"
            )

    primary = pd.read_csv(primary_path)
    gp_pareto = pd.read_csv(gp_pareto_path)
    convergence = pd.read_csv(convergence_path)

    cfd_sinusoidal = primary.loc[
        primary["geometry_class"].eq("sinusoidal")
    ].copy()

    if len(cfd_sinusoidal) != 168:
        raise ValueError(
            "Expected 168 sinusoidal primary CFD cases."
        )

    cfd_pareto = get_pareto_front(
        cfd_sinusoidal,
        mi_column="MI",
        pressure_column="dP",
    )

    common_bounds = common_objective_bounds(
        [
            (cfd_pareto, "MI", "dP"),
            (
                gp_pareto,
                "MI_pred_gp",
                "dP_pred_gp",
            ),
        ]
    )

    cfd_knee = geometric_knee(
        cfd_pareto,
        mi_column="MI",
        pressure_column="dP",
        bounds=common_bounds,
    )
    gp_knee = geometric_knee(
        gp_pareto,
        mi_column="MI_pred_gp",
        pressure_column="dP_pred_gp",
        bounds=common_bounds,
    )
    gp_ideal = ideal_point_compromise(
        gp_pareto,
        mi_column="MI_pred_gp",
        pressure_column="dP_pred_gp",
        bounds=common_bounds,
    )

    selections = pd.DataFrame(
        [
            selection_record(
                "CFD geometric knee",
                "direct CFD case",
                cfd_knee,
                "MI",
                "dP",
            ),
            selection_record(
                "GP-NSGA-II geometric knee",
                "central posterior-mean GP prediction",
                gp_knee,
                "MI_pred_gp",
                "dP_pred_gp",
            ),
            selection_record(
                "GP-NSGA-II equal-weight ideal-point compromise",
                "central posterior-mean GP prediction",
                gp_ideal,
                "MI_pred_gp",
                "dP_pred_gp",
            ),
        ]
    )

    constrained = pressure_constrained_optima(
        gp_pareto,
        pressure_limits,
    )

    cfd_pareto.to_csv(
        summary_directory / "cfd_pareto_front_168_cases.csv",
        index=False,
    )
    gp_pareto.to_csv(
        summary_directory / "gp_nsga2_pareto_front.csv",
        index=False,
    )
    selections.to_csv(
        summary_directory / "pareto_compromise_selections.csv",
        index=False,
    )
    constrained.to_csv(
        summary_directory / "pressure_constrained_optima.csv",
        index=False,
    )

    convergence_summary = (
        convergence.groupby("Generation")[
            "Hypervolume_Relative_To_Final"
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    convergence_summary.to_csv(
        summary_directory / "nsga2_convergence_summary.csv",
        index=False,
    )

    gp_knee_record = selections.loc[
        selections["Selection"].eq(
            "GP-NSGA-II geometric knee"
        )
    ].iloc[0]

    summary = {
        "cfd_pareto_case_pool": 168,
        "cfd_pareto_points": len(cfd_pareto),
        "gp_nsga2_pareto_points": len(gp_pareto),
        "shared_normalization_bounds": common_bounds,
        "recommended_gp_nsga2_knee": {
            "A_h": float(gp_knee_record["A_h"]),
            "phi_deg": float(
                gp_knee_record["phi_deg"]
            ),
            "n_waves": int(
                gp_knee_record["n_waves"]
            ),
            "lambda_um": float(
                gp_knee_record["lambda_um"]
            ),
            "MI_pred_gp": float(
                gp_knee_record["MI_pred_gp"]
            ),
            "dP_pred_gp_Pa": float(
                gp_knee_record["dP_pred_gp_Pa"]
            ),
            "objective_formulation": (
                "central GP posterior-mean MI and "
                "central back-transformed GP pressure"
            ),
        },
        "note": (
            "Independent confirmation CFD results are not "
            "automatically inferred by this script. Add them "
            "to the repository as a separate verified table."
        ),
    }

    (
        summary_directory / "paper_results_summary.json"
    ).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    text_summary = (
        "Micromixer Pareto and optimization summary\n"
        "==========================================\n"
        f"CFD sinusoidal case pool: {len(cfd_sinusoidal)}\n"
        f"CFD Pareto points: {len(cfd_pareto)}\n"
        f"GP-NSGA-II Pareto points: {len(gp_pareto)}\n\n"
        "Recommended central GP-NSGA-II geometric knee\n"
        f"A/h: {gp_knee_record['A_h']:.8f}\n"
        f"phi: {gp_knee_record['phi_deg']:.8f} deg\n"
        f"n: {int(gp_knee_record['n_waves'])}\n"
        f"lambda: {gp_knee_record['lambda_um']:.8f} um\n"
        f"GP MI: {gp_knee_record['MI_pred_gp']:.10f}\n"
        f"GP dP: {gp_knee_record['dP_pred_gp_Pa']:.10f} Pa\n"
    )
    (
        summary_directory / "paper_results_summary.txt"
    ).write_text(
        text_summary,
        encoding="utf-8",
    )

    print("Pareto and result exports completed.")
    print(f"Summary files: {summary_directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Pareto fronts and compromise selections."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory created by scripts 01 and 03.",
    )
    parser.add_argument(
        "--pressure-limits",
        type=float,
        nargs="+",
        default=DEFAULT_PRESSURE_LIMITS_PA,
        help="Pressure limits in Pa for constrained selections.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(
        output_directory=arguments.output_dir.expanduser().resolve(),
        pressure_limits=list(arguments.pressure_limits),
    )
