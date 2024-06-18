# Results Meeting 2024/06/18

## Changes to current simulation set-up
- inlcude Random Forest, Decision Tree and Linear Model (not tune them, but choose good default values, e.g. 500 trees for RF, remaining to default)
- use R2 for results presentation and create an equivalent version of the R2 for MAE (model MAE vs. constant model MAE)

## Next research objectives/questions

### (1) Should we compute PDP (and ALE) on training or test data?
- compute PDP on training data (already done)
- compute PDP on entire test set
- compute PDP in a Cross-Validated manner, i.e. compute it on small subsets of the training data several times (not used to fit the model, refit model) and average curves
- compute the feature effect error for each and compare
- do the same for ALE
- potential result: in contrast to the currently widely adopted standard approach of using the training set, rather use validation data / CV to compute a representative feature effect estimate

### (2) Compare PDP and ALE
- only for the additive scenario
- use the true underlying feature effect from the groundtruth function (theoretical/analytical) as baseline in the error computation
- check if these errors differ for certain scenarios (especially with correlation)
- potential result: recommendations when to use which or when to be careful?

### (3) Library (Python/R) for groundtruth functions
- collect a set of interesting groundtruth functions
- look into the literature and find out what people want for groundtruth/simulation functions
- search for papers: good benchmark functions to evaluate feature effects
- potential papers: https://arxiv.org/abs/1705.04977, https://arxiv.org/pdf/2404.19756
- maybe create a data generator package: choose function and covariate structure
  - no/1/2/...-way interactions
  - Toeplitz covariance matrix *

*In a Toeplitz covariance matrix, each descending diagonal from left to right is constant, meaning that the elements within the same diagonal are the same, and elements between the diagonals can differ but follow the rule that if ∣𝑖−𝑗∣ is the same, then the correlation is the same.

#### ((4) Are there models within the Rashomon set with significantly different feature effects?)
- find equally well performing models with different feature effects (e.g., through pairwise comparison)
