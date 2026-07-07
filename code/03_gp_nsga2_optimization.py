#!/usr/bin/env python3
"""
03_gp_nsga2_optimization.py

Central posterior-mean GP-assisted NSGA-II optimization.

Prerequisite:
    Run 01_data_and_model_pipeline.py first.

The optimization:
- maximizes the final-refit GP posterior-mean mixing index;
- minimizes exp(final-refit GP posterior-mean log pressure);
- treats wave count as an exact integer by performing separate searches for
  n = 2, 3, 4, 5, 6, 7, and 8;
- uses A/h in [0.3, 0.9] and phase offset in [0, 180] degrees;
- merges all fixed-n solutions and applies a global non-dominance filter.

No graphs or images are generated.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


FEATURE_COLUMNS = ["A_h", "sin_phi", "cos_phi", "lambda_um"]
L_SIN_UM = 2400.0
WAVE_COUNTS = [2, 3, 4, 5, 6, 7, 8]

DEFAULT_SEEDS = [11, 42, 101]
DEFAULT_POPULATION = 120
DEFAULT_GENERATIONS = 200
DEFAULT_CROSSOVER_PROBABILITY = 0.90
DEFAULT_CROSSOVER_ETA = 15
DEFAULT_MUTATION_ETA = 20


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fitted model: {path}. "
            "Run 01_data_and_model_pipeline.py first."
        )

    with path.open("rb") as handle:
        return pickle.load(handle)


def build_feature_frame(
    amplitude: np.ndarray,
    phase_deg: np.ndarray,
    wave_count: int,
) -> pd.DataFrame:
    amplitude_array = np.asarray(amplitude, dtype=float)
    phase_array = np.asarray(phase_deg, dtype=float)

    if amplitude_array.shape != phase_array.shape:
        raise ValueError(
            "Amplitude and phase arrays must have identical shapes."
        )

    phase_radians = np.deg2rad(phase_array)

    return pd.DataFrame(
        {
            "A_h": amplitude_array,
            "sin_phi": np.sin(phase_radians),
            "cos_phi": np.cos(phase_radians),
            "lambda_um": np.full(
                amplitude_array.size,
                L_SIN_UM / float(wave_count),
                dtype=float,
            ),
        }
    )


class GPObjectiveEvaluator:
    def __init__(
        self,
        mi_pipeline: Any,
        pressure_pipeline: Any,
    ) -> None:
        self.mi_scaler = mi_pipeline.named_steps["scaler"]
        self.mi_model = mi_pipeline.named_steps["model"]
        self.pressure_scaler = pressure_pipeline.named_steps["scaler"]
        self.pressure_model = pressure_pipeline.named_steps["model"]

    def predict(
        self,
        amplitude: np.ndarray,
        phase_deg: np.ndarray,
        wave_count: int,
    ) -> Dict[str, np.ndarray]:
        features = build_feature_frame(
            amplitude,
            phase_deg,
            wave_count,
        )

        mi_scaled = self.mi_scaler.transform(
            features[FEATURE_COLUMNS]
        )
        pressure_scaled = self.pressure_scaler.transform(
            features[FEATURE_COLUMNS]
        )

        mi_mean, mi_std = self.mi_model.predict(
            mi_scaled,
            return_std=True,
        )
        log_pressure_mean, log_pressure_std = self.pressure_model.predict(
            pressure_scaled,
            return_std=True,
        )

        return {
            "MI_pred_gp": np.asarray(mi_mean, dtype=float),
            "MI_std_gp": np.asarray(mi_std, dtype=float),
            "log_dP_pred_gp": np.asarray(
                log_pressure_mean,
                dtype=float,
            ),
            "log_dP_std_gp": np.asarray(
                log_pressure_std,
                dtype=float,
            ),
            "dP_pred_gp": np.asarray(
                np.exp(log_pressure_mean),
                dtype=float,
            ),
        }


class FixedWaveProblem(Problem):
    def __init__(
        self,
        wave_count: int,
        evaluator: GPObjectiveEvaluator,
    ) -> None:
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.array([0.3, 0.0], dtype=float),
            xu=np.array([0.9, 180.0], dtype=float),
        )
        self.wave_count = int(wave_count)
        self.evaluator = evaluator

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        candidates = np.atleast_2d(
            np.asarray(x, dtype=float)
        )
        predictions = self.evaluator.predict(
            candidates[:, 0],
            candidates[:, 1],
            self.wave_count,
        )

        # pymoo minimizes all objectives.
        out["F"] = np.column_stack(
            [
                -predictions["MI_pred_gp"],
                predictions["dP_pred_gp"],
            ]
        )


def evaluate_candidate_table(
    decision_values: np.ndarray,
    wave_count: int,
    evaluator: GPObjectiveEvaluator,
) -> pd.DataFrame:
    values = np.atleast_2d(
        np.asarray(decision_values, dtype=float)
    )
    predictions = evaluator.predict(
        values[:, 0],
        values[:, 1],
        wave_count,
    )

    table = pd.DataFrame(
        {
            "A_h": values[:, 0],
            "phi_deg": values[:, 1],
            "n_waves": np.full(
                len(values),
                int(wave_count),
                dtype=int,
            ),
            "lambda_um": np.full(
                len(values),
                L_SIN_UM / float(wave_count),
                dtype=float,
            ),
        }
    )

    for column, values_array in predictions.items():
        table[column] = values_array

    return table


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


def run_optimization(
    evaluator: GPObjectiveEvaluator,
    seeds: Iterable[int],
    population: int,
    generations: int,
    crossover_probability: float,
    crossover_eta: float,
    mutation_eta: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_frames: List[pd.DataFrame] = []
    convergence_records: List[Dict[str, float]] = []

    for wave_count in WAVE_COUNTS:
        for seed in seeds:
            problem = FixedWaveProblem(
                wave_count,
                evaluator,
            )
            algorithm = NSGA2(
                pop_size=int(population),
                sampling=FloatRandomSampling(),
                crossover=SBX(
                    prob=float(crossover_probability),
                    eta=float(crossover_eta),
                ),
                mutation=PM(eta=float(mutation_eta)),
                eliminate_duplicates=True,
            )

            result = minimize(
                problem,
                algorithm,
                termination=("n_gen", int(generations)),
                seed=int(seed),
                save_history=True,
                verbose=False,
            )

            if result.X is None:
                raise RuntimeError(
                    f"No solution for n={wave_count}, seed={seed}."
                )

            run_candidates = evaluate_candidate_table(
                result.X,
                wave_count,
                evaluator,
            )
            run_candidates["Seed"] = int(seed)
            candidate_frames.append(run_candidates)

            generation_objectives: List[np.ndarray] = []
            for generation_state in result.history:
                objective_values = np.asarray(
                    generation_state.pop.get("F"),
                    dtype=float,
                )
                if objective_values.size:
                    generation_objectives.append(objective_values)

            if generation_objectives:
                all_values = np.vstack(generation_objectives)
                span = np.ptp(all_values, axis=0)
                reference_point = (
                    np.max(all_values, axis=0)
                    + 0.05 * np.where(span > 0.0, span, 1.0)
                )
                hypervolume_indicator = HV(
                    ref_point=reference_point
                )
                hypervolumes = [
                    float(
                        hypervolume_indicator.do(
                            generation_values
                        )
                    )
                    for generation_values in generation_objectives
                ]
                final_hypervolume = max(
                    hypervolumes[-1],
                    1e-30,
                )

                for generation_number, (
                    generation_values,
                    hypervolume,
                ) in enumerate(
                    zip(
                        generation_objectives,
                        hypervolumes,
                    ),
                    start=1,
                ):
                    non_dominated_indices = (
                        NonDominatedSorting().do(
                            generation_values,
                            only_non_dominated_front=True,
                        )
                    )

                    convergence_records.append(
                        {
                            "n_waves": int(wave_count),
                            "Seed": int(seed),
                            "Generation": int(generation_number),
                            "Population": int(
                                len(generation_values)
                            ),
                            "Non_Dominated_Count": int(
                                len(non_dominated_indices)
                            ),
                            "Hypervolume": float(hypervolume),
                            "Hypervolume_Relative_To_Final": float(
                                hypervolume
                                / final_hypervolume
                            ),
                            "Best_MI_Objective": float(
                                -np.min(
                                    generation_values[:, 0]
                                )
                            ),
                            "Lowest_dP_Objective_Pa": float(
                                np.min(
                                    generation_values[:, 1]
                                )
                            ),
                        }
                    )

    all_candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    all_candidates["_A_round"] = all_candidates["A_h"].round(7)
    all_candidates["_phi_round"] = all_candidates["phi_deg"].round(5)

    unique_candidates = (
        all_candidates.sort_values(
            ["n_waves", "_A_round", "_phi_round", "Seed"]
        )
        .drop_duplicates(
            subset=[
                "n_waves",
                "_A_round",
                "_phi_round",
            ],
            keep="first",
        )
        .drop(
            columns=["_A_round", "_phi_round"]
        )
        .reset_index(drop=True)
    )

    pareto_front = get_pareto_front(
        unique_candidates,
        mi_column="MI_pred_gp",
        pressure_column="dP_pred_gp",
    )
    convergence = pd.DataFrame(convergence_records)

    return unique_candidates, pareto_front, convergence


def run(
    output_directory: Path,
    seeds: List[int],
    population: int,
    generations: int,
) -> None:
    final_model_directory = (
        output_directory / "models" / "final_refit"
    )
    optimization_directory = output_directory / "optimization"
    optimization_directory.mkdir(parents=True, exist_ok=True)

    mi_pipeline = load_pickle(
        final_model_directory / "gaussian_process_MI.pkl"
    )
    pressure_pipeline = load_pickle(
        final_model_directory / "gaussian_process_dP.pkl"
    )

    evaluator = GPObjectiveEvaluator(
        mi_pipeline,
        pressure_pipeline,
    )

    candidates, pareto, convergence = run_optimization(
        evaluator=evaluator,
        seeds=seeds,
        population=population,
        generations=generations,
        crossover_probability=DEFAULT_CROSSOVER_PROBABILITY,
        crossover_eta=DEFAULT_CROSSOVER_ETA,
        mutation_eta=DEFAULT_MUTATION_ETA,
    )

    candidates.to_csv(
        optimization_directory / "nsga2_all_unique_candidates.csv",
        index=False,
    )
    pareto.to_csv(
        optimization_directory / "nsga2_global_pareto_front.csv",
        index=False,
    )
    convergence.to_csv(
        optimization_directory / "nsga2_convergence_history.csv",
        index=False,
    )

    settings = {
        "algorithm": "NSGA-II",
        "objectives": [
            "maximize GP posterior-mean MI",
            "minimize exp(GP posterior-mean log(dP_Pa))",
        ],
        "amplitude_bounds": [0.3, 0.9],
        "phase_bounds_deg": [0.0, 180.0],
        "wave_counts": WAVE_COUNTS,
        "integer_treatment": (
            "separate fixed-n runs followed by global merging"
        ),
        "seeds": seeds,
        "population": population,
        "generations": generations,
        "crossover_probability": DEFAULT_CROSSOVER_PROBABILITY,
        "crossover_eta": DEFAULT_CROSSOVER_ETA,
        "mutation_eta": DEFAULT_MUTATION_ETA,
    }
    (
        optimization_directory / "nsga2_settings.json"
    ).write_text(
        json.dumps(settings, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Central GP-assisted NSGA-II optimization completed.")
    print(f"Optimization outputs: {optimization_directory}")
    print(
        "Next: run 04_pareto_and_results_export.py."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run central GP-assisted NSGA-II optimization."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory created by script 01.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Independent NSGA-II random seeds.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(
        output_directory=arguments.output_dir.expanduser().resolve(),
        seeds=list(arguments.seeds),
        population=int(arguments.population),
        generations=int(arguments.generations),
    )
