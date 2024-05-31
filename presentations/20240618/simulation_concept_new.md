# Simulation Concept

## Aims

Quantification of the error between groundtruth (1D) feature effect and estimated feature effects via different Machine Learning models and Feature Effect methods (Partial Dependence (PD), Accumulated Local Effects (ALE)) for simple groundtruth functions.

## Data Generating Mechanisms

| | Additive:<br>$f(x) = x_1 + 0.5*x_2^2$ | Interaction:<br>$g(x) = x_1*x_2$ | Combined:<br>$h(x) = x_1*x_2 + 0.5*x_2^2$ |
|----------------------------------|----------------------------------|---------------------------------------|----------------------------------|
| $\rho=0$, uniform feature distributions | | | |
| $\rho=0.5$, multivariate normal feature distribution | | | |

- 1000 training samples
- SNRs: 20, 10
- X times on samples drawn with different random seeds

## Estimands and other targets
Targets of the simulation study are
- estimated model PDP
- estimated model ALE

## Methods
The (ML) algorithms:
- GAM (correctly specified + not)
- XGBoost (interactions correctly specified + normal)
- SVM with RBF-kernel
- (featureless model as baseline)
each tuned well (same n_iters and optimizer) w.r.t. their CV MSE.

Feature effect methods:
- PDP (1d)
- ALE (1d)

## Performance Measures
MSE, MAE and $R^2$-Score on holdout test set (10000 samples).

Average pointwise L2-loss between $\widehat{PD}_{\hat f,S}(x_S)$ and $\widehat{PD}_{f, S}(x_S)$, or $\widehat{ALE}_{f,S}(x_S)$ and $\widehat{ALE}_{\hat f,S}(x_S)$, respectively, at 100 equidistant grid points.