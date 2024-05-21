# Mathematical Formalization of Feature Effect Errors

## Groundtruth Feature Effects

### 1. Groundtruth PD

#### 1.1 Theoretical Groundtruth PD

Theoretical Partial Dependence calculated directly on the known groundtruth function and feature distributions by integrating out the other features:

$PD_{f, S}(x_S) = \mathbb{E}_{x_C}[f(x_S, x_C)] = \int f(x_S, x_C )d\mathbb{P}(x_C)$

where $f$ is the groundtruth function, $x_S$ are the features for which the partial dependence function is computed and $x_C$ are the
other features.

#### 1.2 Empirical (estimated) Groundtruth PD

Empirical Partial Dependence estimated on the sampled training data using the groundtruth function as model:

$\widehat{PD}_{f, S}(x_S) = \frac{1}{n} \sum_{i=1}^n f(x_S, x_C^{(i)})$

where $x^{(i)}_C$ are actual feature values from the training sample for the features in which we are not interested, n is the number of instances in the sample.

### 2. Groundtruth ALE

#### 2.1 Theoretical Groundtruth ALE

Theoretical (centered) Accumulated Local Effects calculated directly on the known groundtruth function and the (conditional) feature distributions:

$ALE_{f,S}(x_S) = \int_{x_{min,S}}^{x_S} \mathbb E_{X_C |X_S}[f^S(X_S, X_C)|X_S = z_S] dz_S − c = \int_{x_{min,S}}^{x_S} \int_{x_C}f^S(z_S,x_C)\mathbb P(x_C|z_S)dx_C dz_S - c$

where $f^S(x_s, x_c) = \frac{\delta f(x_S, x_C)}{\delta x_S}$ and $c$ some constant to center the feature effect.

#### 2.2 Empirical (estimated) Groundtruth ALE

Empirical (centered) Accumulated Local Effects estimated on the training sample using the groundtruth function as model:

$\widetilde{\widehat{ALE}}_{f,S}(x_S) = \sum_{k=1}^{k_S(x_S)} \frac{1}{n_S(k)} \sum_{\{i:x^{(i)}_S \in N_S(k)\}} [f(z_{k,S}, x^{(i)}_C) − f(z_{k−1,S}, x^{(i)}_C)]$ 

This effect is centered so that the mean effect is zero:

$\widehat{ALE}_{f,S}(x_S) = \widetilde{\widehat{ALE}}_{f,S}(x_S) − \frac{1}{n} \sum_{i=1}^n\widetilde{\widehat{ALE}}_{f,S}(x^{(i)}_S)$

Again $S$ is the feature for which the feature effect is computed (with observations $x_S$, $x_S^{(i)}$ for the ith observation), $C$ the remaining features. 
For each feature, $\{N_S(k) = (z_{k−1,S}, z_{k,S}] : k = 1, 2, \dots ,K\}$ describes a sufficiently fine partition of the sample range of $\{x^{(i)}_S :i=1, 2, \dots , n\}$ into $K$ intervals. 
For $k =1, 2, \dots ,K$, $n_S(k)$ denotes the number of training observations that fall into the kth interval $N_S(k)$. For a particular value $x$ of the predictor $x_S$, $k_S(x)$ denotes the index of the interval into which $x$ falls.


### 3. Groundtruth main effects

Use the main effects of the groundtruth function. Makes only sense in absence of interaction effects.

*Example (Friedman1):*

$f_{3}(x_3) = 20x_3^2-20x_3$

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

From these definitions ($PD_{f, S}(x_S)$,
$\widehat{PD}_{f, S}(x_S)$,
$ALE_{f,S}(x_S)$,
$\widehat{ALE}_{f,S}(x_S)$,
$f_{S}(x_S)$, and
$\widehat{PD}_{\hat f,S}(x_S)$,
$\widehat{ALE}_{\hat f,S}(x_S)$,
$\hat f_S(x_S)$), one can derive several error metrics:

Model PD w.r.t. DGP:
- $Err(\widehat{PD}_{\hat f,S}(x_S),PD_{f, S}(x_S))$ or $Err_c(\widehat{PD}_{\hat f,S}(x_S),PD_{f, S}(x_S))$
- $Err(\widehat{PD}_{\hat f,S}(x_S), \widehat{PD}_{f, S}(x_S))$ or $Err_c(\widehat{PD}_{\hat f,S}(x_S), \widehat{PD}_{f, S}(x_S))$
- $Err_c(\widehat{PD}_{\hat f,S}(x_S), f_{S}(x_S))$

similarly, model ALE w.r.t. DGP:
- $Err(\widehat{ALE}_{\hat f,S}(x_S), ALE_{f,S}(x_S)) = Err_c(\dots)$
- $Err(\widehat{ALE}_{\hat f,S}(x_S), \widehat{ALE}_{f,S}(x_S)) = Err_c(\dots)$
- $Err_c(\widehat{ALE}_{\hat f,S}(x_S), f_{S}(x_S))$

or, model main effect w.r.t. DGP:
- $Err_c(\hat f_S(x_S), f_{S}(x_S))$

alternatively, w.r.t. model main effect:
- $Err_c(\widehat{PD}_{\hat f,S}(x_S), \hat f_S(x_S))$
- $Err_c(\widehat{ALE}_{\hat f,S}(x_S), \hat f_S(x_S))$

Of course, other combinations would also be possible (e.g., $\widehat{PD}_{f, S}(x_S)$ vs. $PD_{f, S}(x_S)$, and $\widehat{ALE}_{f,S}(x_S)$ vs. $ALE_{f,S}(x_S)$ to assess the quality of the estimation), but are not necessarily always meaningful (e.g., $\widehat{PD}_{\hat f,S}(x_S)$ vs. $ALE_{f,S}(x_S)$).

$Err( \dots)$ can correspond to any pointwise error measure, such as the Mean Squared Error (MSE), estimated by averaging over the (L2-)loss between a grid of equidistant points of the compared effects.

Additionally, the curves can be centered around 0 before the comparison, here denoted by $Err_c( \dots)$ (which is sometimes even necessary for comparabillity, e.g. with groundtruth main effects).

## Examples

Using the examples from *Liu et al., 2018* for three different exemplary groundtruth functions $f$, we get the following groundtruth feature effects:

|  | $f(x_1, x_2) = x_1 + x_2$ | $f(x_1, x_2) = x_1 x_2$ | $f(x_1, x_2) = x_1^2 + x_1 x_2$
| --- | --- | --- | --- |
| $PD_{f, 1}(x_1)$ | $x_1 + \mathbb{E}[X_2]$ | $\mathbb{E}[X_2]x_1$ | $x_1^2 + \mathbb{E}[X_2]x_1$ |
| $ALE_{f, 1}(x_1)$ | $x_1 - x_{min,1} - c$ | $\int_{x_{min,1}}^{x_1} \mathbb{E}[X_2 ∣ X_1=z_1]dz_1−c$ | $x_1^2 - x_{min,1}^2 + \int_{x_{min,1}}^{x_1} \mathbb{E}[X_2 ∣ X_1=z_1]dz_1−c$ |
| $f_1(x_1)$ | $x_1$ | - | $x_1^2$ |
| $PD_{f, 2}(x_2)$ | $x_2 + \mathbb{E}[X_1]$ | $\mathbb{E}[X_1]x_2$ | $\mathbb{E}[X_1^2] + \mathbb{E}[X_1]x_2$ |
| $ALE_{f, 2}(x_2)$ | $x_2 - x_{min,2} - c$ | $\int_{x_{min,2}}^{x_2} \mathbb{E}[X_1 ∣ X_2=z_2]dz_2−c$ | $\int_{x_{min,2}}^{x_2} \mathbb{E}[X_1 ∣ X_2=z_2]dz_2−c$ |
| $f_2(x_2)$ | $x_2$ | - | - |

In order to make this more concrete, let us use the following example feature distribution from *Liu et al., 2018*

Let $X = (X_1, X_2)^T$ be bivariate normal with parameters 

$$
\mathbb{E}[X_1]=\mu_1, Var[X_1] = \sigma_1^2, \mathbb{E}[X_2]=\mu_2, Var[X_2] = \sigma_2^2, Corr(X_1, X_2) = \rho
$$.

We can then model $X_1$ as a linear function of $X_2$: 

$$
X_1 − \mu_1 = \beta_{1|2}(X_2 − \mu_2) + \epsilon_2
$$, 

where $\beta_{1|2} = \rho \frac{\sigma_1}{\sigma_2}$.
Similarly, we have

$$
X_2 − \mu_2 = \beta_{2|1}(X_1 − \mu_1) + \epsilon_1
$$ 

with $\beta_{2|1} = \rho \frac{\sigma_2}{\sigma_1}$. 

This leads to the following results for the feature effects above:

|  | $f(x_1, x_2) = x_1 + x_2$ | $f(x_1, x_2) = x_1 x_2$ | $f(x_1, x_2) = x_1^2 + x_1 x_2$
| --- | --- | --- | --- |
| $PD_{f, 1}(x_1)$ | $x_1 + \mu_2$ | $\mu_2x_1$ | $x_1^2 + \mu_2x_1$ |
| $ALE_{f, 1}(x_1)$ | $x_1 - x_{min,1} - c$ | $\frac{\beta_{2∣1}}{2}(x_1^2-x_{min,1}^2)+(\mu_2 - \beta_{2∣1} \mu_1) (x_1 - x_{min,1}) - c$ | $(1 + \frac{\beta_{2∣1}}{2})(x_1^2-x_{min,1}^2)+(\mu_2 - \beta_{2∣1} \mu_1) (x_1 - x_{min,1}) - c$ |
| $f_1(x_1)$ | $x_1$ | - | $x_1^2$ |
| $PD_{f, 2}(x_2)$ | $x_2 + \mu_1$ | $\mu_1x_2$ | $\mu_1^2 + \sigma_1^2 + \mu_1x_2$ |
| $ALE_{f, 2}(x_2)$ | $x_2 - x_{min,2} - c$ | $\frac{\beta_{1∣2}}{2}(x_2^2-x_{min,2}^2)+(\mu_1 - \beta_{1∣2} \mu_2) (x_2 - x_{min,2}) - c$ | $\frac{\beta_{1∣2}}{2}(x_2^2-x_{min,2}^2)+(\mu_1 - \beta_{1∣2} \mu_2) (x_2 - x_{min,2}) - c$ |
| $f_2(x_2)$ | $x_2$ | - | - |

Note that
- if additive constants are ignored (e.g., by centering the curve), the feature effects for the purely additive example conform.
- if additive constants are ignored and the correlation between the features is $\rho=0$, the PD and ALE are equivalent for all functions.

## References

Apley, D.W., Zhu, J., 2020. Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models. Journal of the Royal Statistical Society Series B: Statistical Methodology 82, 1059–1086. https://doi.org/10.1111/rssb.12377

Friedman, J.H., 2001. Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics 29, 1189–1232.

Liu, X., Chen, J., Vaughan, J., Nair, V., Sudjianto, A., 2018. Model Interpretation: A Unified Derivative-based Framework for Nonparametric Regression and Supervised Machine Learning. https://doi.org/10.48550/arXiv.1808.07216