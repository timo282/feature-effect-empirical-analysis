# Simulation Concept - ADEMP

## Aims

Empirical Analysis of the Error when estimating feature effects via various Machine Learning models and Feature Effect methods such as Partial Dependence (PD) or Accumulated Local Effects (ALE), quantified through different loss metrics.

## Data Generating Mechanisms

Strategy: Random sampling from a groundtruth. Groundtruth is the Friedman1 dataset (may be extended to other groundtruths later on), namely the function
y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + epsilon * N(0, 1), where epsilon is a noise parameter.
The following parameters of the Data Generating Mechanism will be varied factorially:
- n_train: Number of datapoints drawn to train models on
- epsilon: different strengths of noise in the data

## Estimands and other targets
Targets of the simulation study are
- marginal 1d effects of features on the target (partial dependence)
- local conditional 1d effects of features on the target (accumulated local effects)

## Methods
The methods encompass two parts: the Machine Learning model trained on the simulated data and the explanation method explaining the respective feature effects based on the model. 
The (ML) algorithms considered in the study are:
- GAM/GLM
- XGBoost
- Random Forest
- SVM with RBF-kernel.
Each ML model is tuned well (with the same number of iterations and same tuning algorithm) by means of their MSE, and their performance is reported using R^2-Score, MAE, and MSE.
Subsequently, PDP (for marginal) and ALE (for local conditional) are applied to estimate 1d feature effects.

## Performance Measures
To compare and evaluate the distance of the estimated feature effects to their respective groundtruths, the L2 loss at different “grid points” is calculated.

# Open Questions
- number of simulations: 100, 500, 1000?