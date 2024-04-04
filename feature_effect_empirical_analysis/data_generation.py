from numbers import Integral, Real
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, validate_params


class Groundtruth(BaseEstimator):
    """
    A wrapper class for the groundtruth (Friedman1) wrapped as fitted sklearn
    regression estimator adhering to the standard scikit-learn estimator
    interface.

    Attributes
    ----------
    _is_fitted__ : bool
        Indicates whether the estimator has been 'fitted'. This is a mock
        attribute and is set to True by default.
    _estimator_type : str
        Defines the type of the estimator as 'regressor'.
    """

    def __init__(self):
        self._is_fitted__ = True
        self._estimator_type = "regressor"

    def __sklearn_is_fitted__(self):
        return self._is_fitted__

    def fit(self, X, y):
        """
        Mocks fit method for the groundtruth (does not perform any operation).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Not used.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Not used.
        """

    def predict(self, X):
        """
        Returns target value (y) of the groundtruth (Friedman1) for each
        sample in X. The output `y` is created according to the formula::

        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
        + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        return (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )


@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 5, None, closed="left")],
        "noise": [Interval(Real, 0.0, None, closed="left")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def _make_friedman1(
    n_samples=100, n_features=10, *, noise=0.0, random_state=None
):
    """Reimplementation of sklearn.datasets.make_friedman1 to
    generate the "Friedman #1" regression problem.

    This dataset is described in Friedman [1] and Breiman [2].

    Inputs `X` are independent features uniformly distributed on the interval
    [0, 1]. The output `y` is created according to the formula::

        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
    + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).

    Out of the `n_features` features, only 5 are actually used to compute
    `y`. The remaining features are independent of `y`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=10
        The number of features. Should be at least 5.

    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.

    y : ndarray of shape (n_samples,)
        The output values.

    References
    ----------
    .. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
           of Statistics 19 (1), pages 1-67, 1991.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
           pages 123-140, 1996.
    """
    generator = check_random_state(random_state)
    groundtruth = Groundtruth()

    X = generator.uniform(size=(n_samples, n_features))
    y = groundtruth.predict(X) + noise * generator.standard_normal(
        size=(n_samples)
    )

    return X, y


def generate_data(
    n_train: int,
    n_test: int,
    noise_sd: float,
    seed: int,
    n_features: int = 5,
):
    """Generate data for the Friedman1 dataset.

    Parameters
    ----------
    n_train : int
        Number of training samples to generate.
    n_test : int
        Number of test samples to generate.
    noise_sd : float
        Standard deviation of the noise to add to the data.
    seed : int
        Random seed to use for reproducibility.
    n_features : int, optional
        Number of features to generate (all feature >5 are independet from y),
        by default 5.

    Returns
    -------
    _type_
        _description_
    """
    X, y = _make_friedman1(
        n_samples=n_train + n_test,
        n_features=n_features,
        noise=noise_sd,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42
    )

    return X_train, y_train, X_test, y_test
