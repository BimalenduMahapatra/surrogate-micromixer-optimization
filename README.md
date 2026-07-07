# surrogate-micromixer-optimization
This repository contains the datasets and source code accompanying our study on surrogate-assisted design optimization of two-dimensional sinusoidal converging-diverging micromixers for shear-thinning (Carreau–Yasuda) fluids.

The repository includes:

CFD-derived datasets, Machine-learning surrogate models, Training and evaluation scripts, NSGA-II multi-objective optimization codes

Data, surrogate models, and multi-objective optimization code for the study
"Data-Driven Surrogate Modeling of Non-Newtonian Micromixing in Sinusoidal
Converging–Diverging Microchannels" (Sharma and Mahapatra).

The pipeline builds Gaussian Process Regression (GPR) surrogates for the outlet
mixing index (MI) and pressure drop (ΔP) of a Carreau–Yasuda fluid from a database
of 218 CFD simulations, then couples the GPR surrogate with NSGA-II to reconstruct
the CFD Pareto front and identify optimal micromixer geometries.

Repository contents

FileDescriptionprimary_final.csv169 primary CFD cases (168 sinusoidal + 1 straight reference); ΔP in Pa.external_final.csv49 independent external-holdout cases (includes n = 7).01_data_and_model_pipeline.pyData prep, deterministic 119/25/25 split, hyperparameter selection, same-fit and final-refit model fitting.02_model_validation_and_shap.pyInternal/external metrics, 5-fold cross-validation, GP interval coverage, TreeSHAP export.03_gp_nsga2_optimization.pyCentral posterior-mean GP-assisted NSGA-II optimization.04_pareto_and_results_export.pyCFD and GP Pareto fronts, knee/ideal/pressure-constrained selections, summary export.

Requirements

Python 3.10+ with numpy, pandas, scikit-learn, xgboost, shap, and pymoo.

bashpip install numpy pandas scikit-learn xgboost shap pymoo

Usage

Run the scripts in order; each reads the previous stage's outputs from --output-dir.

bashpython 01_data_and_model_pipeline.py --primary-csv primary_final.csv --external-csv external_final.csv --output-dir outputs
python 02_model_validation_and_shap.py --output-dir outputs
python 03_gp_nsga2_optimization.py --output-dir outputs
python 04_pareto_and_results_export.py --output-dir outputs

Results (processed data, fitted models, metrics, SHAP values, Pareto fronts, and
optimal selections) are written under outputs/. The scripts export numerical
tables only; manuscript figures are generated separately.

Notes


Features are [A/h, sin(phi), cos(phi), lambda]; ΔP is modeled as log(ΔP).
NSGA-II uses population 120, 200 generations, seeds 11/42/101, with fixed-n runs
merged under a global non-dominance filter.
Exact optimizer outputs (for example the knee phase offset) can vary slightly with
library versions; the reported values correspond to the environment above.


Citation

If you use this repository, please cite the associated paper.

License

See LICENSE.
Project contentpaperCreated by youAdd PDFs, documents, or other text to reference in this project.Contentpdfimportant_one.pypyprimary_final.csvcsvexternal_final.csvcsvpdfI'm revising a manuscript for Physics of Fluids titled "Data-driven surrogate 
modeling and multi-objective optimization of a 2D sinusoidal converging–diverging 
micromixer for a Carreau–Yasuda (shear-thinning) fluid." I've been through several 
review rounds with an AI reviewer and am near submipasted01_data_and_model_pipeline.py1 linepy04_pareto_and_results_export.py1 linepy02_model_validation_and_shap.py1 linepy03_gp_nsga2_optimization.py1 linepy#!/usr/bin/env python3
"""
01_data_and_model_pipeline.py

Data preparation and surrogate-model development for the sinusoidal
micromixer study.

This script:
1. loads the 169-case primary CFD dataset and 49-case external dataset;
2. creates the model features [A/h, sin(phi), cos(phi), lambdpasted#!/usr/bin/env python3
"""
02_model_validation_and_shap.py

Numerical validation, cross-validation, GP uncertainty, and TreeSHAP export
for the fitted micromixer surrogate models.

Prerequisite:
    Run 01_data_and_model_pipeline.py first.

This script generates CSV tables only. It does nopasted#!/usr/bin/env python3
"""
03_gp_nsga2_optimization.py

Central posterior-mean GP-assisted NSGA-II optimization.

Prerequisite:
    Run 01_data_and_model_pipeline.py first.

The optimization:
- maximizes the final-refit GP posterior-mean mixing index;
- minimizes exp(final-refit GP posterpasted#!/usr/bin/env python3
"""
04_pareto_and_results_export.py

Pareto-front construction, compromise selection, and final numerical exports.

Prerequisites:
    Run 01_data_and_model_pipeline.py first.
    Run 03_gp_nsga2_optimization.py second.

This script:
- extracts the CFD Pareto front pasted












## License and Usage

This repository contains material associated with an unpublished research manuscript. Until the corresponding work is published, the code, datasets, figures, and results are provided solely for authorized use and may not be copied, redistributed, or publicly disclosed without the prior written permission of the authors.
