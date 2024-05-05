# Mathematical Formalization of Feature Effect Errors

## Groundtruth Feature Effects

### 1. Groundtruth PD

#### 1.1 Theoretical Groundtruth PD

Theoretical Partial Dependence calculated directly on the known groundtruth function and feature distributions by integrating out the other features:

$PD_{GT, theor., S}(x_S) = \mathbb{E}_{x_C}[f(x_S, x_C)] = \int f(x_S, x_C )d\mathbb{P}(x_C)$

where $f$ is the groundtruth function, $x_S$ are the features for which the partial dependence function is computed and $x_C$ are the
other features.

#### 1.2 Empirical (estimated) Groundtruth PD

Empirical Partial Dependence estimated on the sampled training data using the groundtruth function as model:

$\hat{PD}_{GT, emp., S}(x_S) = \frac{1}{n} \sum_{i=1}^n f(x_S, x_C^{(i)})$

where $x^{(i)}_C$ are actual feature values from the training sample for the features in which we are not interested, n is the number of instances in the sample.

### 2. Groundtruth ALE

#### 2.1 Theoretical Groundtruth ALE

Theoretical (centered) Accumulated Local Effects calculated directly on the known groundtruth function and the (conditional) feature distributions:

$ALE_{GT,theoret.,x_S}(x_S) = \int_{z_{0,1}}^{x_S} \mathbb E_{X_C |X_S}[f^S(X_S, X_C)|X_S = z_S] dz_S − c = \int_{z_{0,1}}^{x_S} \int_{x_C}f^S(z_S,x_C)\mathbb P(x_C|z_S)dx_C dz_S - c$

where $f^S(x_s, x_c) = \frac{\delta f(x_S, x_C)}{\delta x_S}$ and $c$ some constant to center the feature effect.

#### 2.2 Empirical (estimated) Groundtruth ALE

Empirical (centered) Accumulated Local Effects estimated on the training sample using the groundtruth function as model:

fj,ALE(x) =
k
∑j (x)
k=1
1
nj (k)
∑
i:x
(i)
j ∈Nj (k)
[
f(zk,j , x
(i)
\j
) − f(zk−1,j , x
(i)
\j


### 3. Groundtruth main effects

Use the main effects of the groundtruth function. Makes only sense in absence of interaction effects.

Example (Friedman1):
$GT_{x_3}(x_3) = 20x_3^2-20x_3$

Note that the offset of the feature effect in y-direction is ignored. When using this groundtruth effect for error calculation, the model feature effect should be centered.

## Model Feature Effects

### 1. Model PD
Partial dependence of the model estimated by calculating averages on the training sample as defined by Friedman (2001):

$\hat{PD}_{\hat f,S}(x_S) = \frac{1}{n} \sum_{i=1}^n \hat f(x_S, x_C^{(i)})$

where $\hat f$ is the trained model (also estimated on the training data).

### 2. Model ALE
Accumulated Local Effects of the model estimated on the training samlple as proposed by [SOURCE].

fj,ALE(x) =
k
∑j (x)
k=1
1
nj (k)
∑
i:x
(i)
j ∈Nj (k)
[
f(zk,j , x
(i)
\j
) − f(zk−1,j , x
(i)
\j


## Error Metrics

## Examples

Functions from paper x1+x2 and x1*x2
## References
[Friedman 2001] Friedman, Jerome H.: Greedy Function Approximation: A Gradient
Boosting Machine. In: The Annals of Statistics 29 (2001), Nr. 5, pp. 1189–1232


