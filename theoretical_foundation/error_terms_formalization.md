# Mathematical Formalization of Feature Effect Errors

## Groundtruth Feature Effects

### 1. Groundtruth PD

#### 1.1 Theoretical Groundtruth PD

Theoretical Partial Dependence calculated directly on the known groundtruth function and feature distributions by integrating out the other features:

$PD_{GT, S}(x_S) = \mathbb{E}_{x_C}[f(x_S, x_C)] = \int f(x_S, x_C )d\mathbb{P}(x_C)$

where $f$ is the groundtruth function, $x_S$ are the features for which the partial dependence function is computed and $x_C$ are the
other features.

#### 1.2 Empirical (estimated) Groundtruth PD

Empirical Partial Dependence estimated on the sampled training data using the groundtruth function as model:

$\widehat{PD}_{GT, S}(x_S) = \frac{1}{n} \sum_{i=1}^n f(x_S, x_C^{(i)})$

where $x^{(i)}_C$ are actual feature values from the training sample for the features in which we are not interested, n is the number of instances in the sample.

### 2. Groundtruth ALE

#### 2.1 Theoretical Groundtruth ALE

Theoretical (centered) Accumulated Local Effects calculated directly on the known groundtruth function and the (conditional) feature distributions:

$ALE_{GT,S}(x_S) = \int_{z_{0,1}}^{x_S} \mathbb E_{X_C |X_S}[f^S(X_S, X_C)|X_S = z_S] dz_S − c = \int_{z_{0,1}}^{x_S} \int_{x_C}f^S(z_S,x_C)\mathbb P(x_C|z_S)dx_C dz_S - c$

where $f^S(x_s, x_c) = \frac{\delta f(x_S, x_C)}{\delta x_S}$ and $c$ some constant to center the feature effect.

#### 2.2 Empirical (estimated) Groundtruth ALE

Empirical (centered) Accumulated Local Effects estimated on the training sample using the groundtruth function as model:

$\widetilde{\widehat{ALE}}_{GT,S}(x_S) = \sum_{k=1}^{k_S(x_S)} \frac{1}{n_S(k)} \sum_{\{i:x^{(i)}_S \in N_S(k)\}} [f(z_{k,S}, x^{(i)}_C) − f(z_{k−1,S}, x^{(i)}_C)]$ 

This effect is centered so that the mean effect is zero:

$\widehat{ALE}_{GT,S}(x_S) = \widetilde{\widehat{ALE}}_{GT,S}(x_S) − \frac{1}{n} \sum_{i=1}^n\widetilde{\widehat{ALE}}_{GT,S}(x^{(i)}_S)$

Again $S$ is the feature for which the feature effect is computed (with observations $x_S$, $x_S^{(i)}$ for the ith observation), $C$ the remaining features. 
For each feature, $\{N_S(k) = (z_{k−1,S}, z_{k,S}] : k = 1, 2, \dots ,K\}$ describes a sufficiently fine partition of the sample range of $\{x^{(i)}_S :i=1, 2, \dots , n\}$ into $K$ intervals. 
For $k =1, 2, \dots ,K$, $n_S(k)$ denotes the number of training observations that fall into the kth interval N_S(k). For a particular value $x$ of the predictor $x_S$, $k_S(x) denotes the index of the interval into which x falls.


### 3. Groundtruth main effects

Use the main effects of the groundtruth function. Makes only sense in absence of interaction effects.

*Example (Friedman1):*

$GT_{3}(x_3) = 20x_3^2-20x_3$

Note that the offset of the feature effect in y-direction is ignored. When using this groundtruth effect for error calculation, the model feature effect should be centered.

## Model Feature Effects

### 1. Model PD
Partial dependence of the model estimated by calculating averages on the training sample as defined by *Friedman, 2001*:

$\widehat{PD}_{\hat f,S}(x_S) = \frac{1}{n} \sum_{i=1}^n \hat f(x_S, x_C^{(i)})$

where $\hat f$ is the trained model (also estimated on the training data).

### 2. Model ALE
Accumulated Local Effects of the model estimated on the training samlple as proposed by *Apley and Zhu, 2020*:

$\widetilde{\widehat{ALE}}_{\hat f,S}(x_S) = \sum_{k=1}^{k_S(x_S)} \frac{1}{n_S(k)} \sum_{\{i:x^{(i)}_S \in N_S(k)\}} [\hat f(z_{k,S}, x^{(i)}_C) − \hat f(z_{k−1,S}, x^{(i)}_C)]$ 

for the uncentered effect, where $\hat f$ is the estimated model. This effect is again centered so that the mean effect is zero:

$\widehat{ALE}_{\hat f,S}(x_S) = \widetilde{\widehat{ALE}}_{\hat f,S}(x_S) − \frac{1}{n} \sum_{i=1}^n\widetilde{\widehat{ALE}}_{\hat f,S}(x^{(i)}_S)$

### 3. Model Main Effects

Main effects of the estimated model, if they can directly by derived:

- $\hat f_S(x_S) = x_S^T \beta_S$ for a linear regression model and a single feature S
- corresponding spline terms for GAMs
- ...


## Error Metrics

From these definitions ($PD_{GT, S}(x_S)$,
$\widehat{PD}_{GT, S}(x_S)$,
$ALE_{GT,S}(x_S)$,
$\widehat{ALE}_{GT,S}(x_S)$,
$GT_{S}(x_S)$, and
$\widehat{PD}_{\hat f,S}(x_S)$,
$\widehat{ALE}_{\hat f,S}(x_S)$,
$\hat f_S(x_S)$), one can derive several error metrics:

Model PD w.r.t. DGP:
- $Err(\widehat{PD}_{\hat f,S}(x_S),PD_{GT, S}(x_S))$ or $Err_c(\widehat{PD}_{\hat f,S}(x_S),PD_{GT, S}(x_S))$
- $Err(\widehat{PD}_{\hat f,S}(x_S), \widehat{PD}_{GT, S}(x_S))$ or $Err_c(\widehat{PD}_{\hat f,S}(x_S), \widehat{PD}_{GT, S}(x_S))$
- $Err_c(\widehat{PD}_{\hat f,S}(x_S), GT_{S}(x_S))$

similarly, model ALE w.r.t. DGP:
- $Err(\widehat{ALE}_{\hat f,S}(x_S), ALE_{GT,S}(x_S)) = Err_c(\dots)$
- $Err(\widehat{ALE}_{\hat f,S}(x_S), \widehat{ALE}_{GT,S}(x_S)) = Err_c(\dots)$
- $Err_c(\widehat{ALE}_{\hat f,S}(x_S), GT_{S}(x_S))$

or, model main effect w.r.t. DGP:
- $Err_c(\hat f_S(x_S), GT_{S}(x_S))$

alternatively, w.r.t. model main effect:
- $Err_c(\widehat{PD}_{\hat f,S}(x_S), \hat f_S(x_S))$
- $Err_c(\widehat{ALE}_{\hat f,S}(x_S), \hat f_S(x_S))$

Of course, other combinations would also be possible (e.g., GT_PD_emp vs GT_PD_theor and GT_ALE_emp vs. GT_ALE_theor to assess the quality of the estimation), but are not necessarily always meaningful (e.g., Model_PD vs GT_ALE_theor).

$Err( \dots)$ can correspond to any pointwise error measure, such as the Mean Squared Error (MSE), estimated by averaging over the (L2-)loss between a grid of equidistant points of the compared effects.

Additionally, the curves can be centered around 0 before the comparison, here denoted by $Err_c( \dots)$ (which is sometimes even necessary for comparabillity, e.g. with groundtruth main effects).

## Examples

Functions used in *Liu et al., 2018*: x1+x2 and x1*x2

## References

Apley, D.W., Zhu, J., 2020. Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models. Journal of the Royal Statistical Society Series B: Statistical Methodology 82, 1059–1086. https://doi.org/10.1111/rssb.12377

Friedman, J.H., 2001. Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics 29, 1189–1232.

Liu, X., Chen, J., Vaughan, J., Nair, V., Sudjianto, A., 2018. Model Interpretation: A Unified Derivative-based Framework for Nonparametric Regression and Supervised Machine Learning. https://doi.org/10.48550/arXiv.1808.07216