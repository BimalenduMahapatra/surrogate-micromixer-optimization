# surrogate-micromixer-optimization
Data, surrogate models, and multi-objective optimization code for the study
"Data-Driven Surrogate Modeling of Non-Newtonian Micromixing in Sinusoidal
Converging-Diverging Microchannels" (Sharma and Mahapatra).

The pipeline builds Gaussian Process Regression (GPR) surrogates for the outlet
mixing index (MI) and pressure drop (dP) of a Carreau-Yasuda fluid from a database
of 218 CFD simulations, then couples the GPR surrogate with NSGA-II to reconstruct
the CFD Pareto front and identify optimal micromixer geometries.

