# surrogate-micromixer-optimization
Data, surrogate models, and multi-objective optimization code for the study
"Data-Driven Surrogate Modeling of Non-Newtonian Micromixing in Sinusoidal
Converging-Diverging Microchannels" (Sharma and Mahapatra).

The pipeline builds Gaussian Process Regression (GPR) surrogates for the outlet
mixing index (MI) and pressure drop (dP) of a Carreau-Yasuda fluid from a database
of 218 CFD simulations, then couples the GPR surrogate with NSGA-II to reconstruct
the CFD Pareto front and identify optimal micromixer geometries.

# contents

primary_final.csv: 169 primary CFD cases (168 sinusoidal plus 1 straight reference), with dP in Pa.
external_final.csv: 49 independent external-holdout cases, including n = 7 geometries.
01_data_and_model_pipeline.py: data preparation, deterministic 119/25/25 split, hyperparameter selection, and same-fit and final-refit model fitting.
02_model_validation_and_shap.py: internal and external metrics, five-fold cross-validation, GP interval coverage, and TreeSHAP export.
03_gp_nsga2_optimization.py: central posterior-mean GP-assisted NSGA-II optimization.
04_pareto_and_results_export.py: CFD and GP Pareto fronts, knee, ideal-point, and pressure-constrained selections, and summary export.

# Usage

Run the scripts in order. Each one reads the previous stage's outputs from the
directory given by --output-dir.
python 01_data_and_model_pipeline.py --primary-csv primary_final.csv --external-csv external_final.csv --output-dir outputs
python 02_model_validation_and_shap.py --output-dir outputs
python 03_gp_nsga2_optimization.py --output-dir outputs
python 04_pareto_and_results_export.py --output-dir outputs

# Notes

The model features are A/h, sin(phi), cos(phi), and lambda, and the pressure drop is
modeled as log(dP). NSGA-II uses a population of 120 over 200 generations with seeds
11, 42, and 101, running each wave count separately and merging the results under a
global non-dominance filter. Exact optimizer outputs, such as the knee phase offset,
can vary slightly with library versions; the reported values correspond to the
environment described above.

# Citation

If you use this repository, please cite the associated paper.






