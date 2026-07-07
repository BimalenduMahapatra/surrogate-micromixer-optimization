# surrogate-micromixer-optimization
Data, surrogate models, and multi-objective optimization code for the study
"Data-Driven Surrogate Modeling of Non-Newtonian Micromixing in Sinusoidal
Converging-Diverging Microchannels" (Sharma and Mahapatra).

The pipeline builds Gaussian Process Regression (GPR) surrogates for the outlet
mixing index (MI) and pressure drop (dP) of a Carreau-Yasuda fluid from a database
of 218 CFD simulations, then couples the GPR surrogate with NSGA-II to reconstruct
the CFD Pareto front and identify optimal micromixer geometries.

# Contents


primary_final.csv: 169 primary CFD cases (168 sinusoidal plus 1 straight reference), with dP in Pa.
external_final.csv: 49 independent external-holdout cases, including n = 7 geometries.
01_data_and_model_pipeline.py: data preparation, deterministic 119/25/25 split, hyperparameter selection, and same-fit and final-refit model fitting.
02_model_validation_and_shap.py: internal and external metrics, five-fold cross-validation, GP interval coverage, and TreeSHAP export.
03_gp_nsga2_optimization.py: central posterior-mean GP-assisted NSGA-II optimization.
04_pareto_and_results_export.py: CFD and GP Pareto fronts, knee, ideal-point, and pressure-constrained selections, and summary export.


