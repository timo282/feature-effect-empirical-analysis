from numbers import Integral, Real
from typing import Literal, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, validate_params
from scipy.integrate import quad


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
        + 10 * X[:, 3] + 5 * X[:, 4].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]

    def get_theoretical_partial_dependence(
        self, feature: Literal["x_1", "x_2", "x_3", "x_4", "x_5"], feature_distribution: str
    ) -> Callable:
        if feature_distribution != "uniform":
            raise ValueError("Only uniform feature distribution is supported.")

        integrand12_over_x2 = lambda x2, x1: 10 * np.sin(np.pi * x1 * x2)  # noqa: E731
        integrand12_over_x1 = lambda x1, x2: 10 * np.sin(np.pi * x1 * x2)  # noqa: E731
        integrand3_over_x3 = lambda x3: 20 * (x3 - 0.5) ** 2  # noqa: E731

        outer_integral = lambda x1: quad(integrand12_over_x2, 0, 1, args=(x1,))[0]  # noqa: E731
        complete_integral_12, _ = quad(outer_integral, 0, 1)
        integral_3, _ = quad(integrand3_over_x3, 0, 1)

        if feature == "x_1":

            def partial_dependence(x1):
                integral_12, _ = quad(integrand12_over_x2, 0, 1, args=(x1,))
                return integral_12 + integral_3 + 10 * 0.5 + 5 * 0.5

        elif feature == "x_2":

            def partial_dependence(x2):
                integral_12, _ = quad(integrand12_over_x1, 0, 1, args=(x2,))
                return integral_12 + integral_3 + 10 * 0.5 + 5 * 0.5

        elif feature == "x_3":

            def partial_dependence(x3):
                return complete_integral_12 + 20 * (x3 - 0.5) ** 2 + 10 * 0.5 + 5 * 0.5

        elif feature == "x_4":

            def partial_dependence(x4):
                return complete_integral_12 + integral_3 + 10 * x4 + 2.5

        elif feature == "x_5":

            def partial_dependence(x5):
                return complete_integral_12 + integral_3 + 5 + 5 * x5

        return partial_dependence


@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 5, None, closed="left")],
        "noise": [Interval(Real, 0.0, None, closed="left")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def _make_friedman1(n_samples=100, n_features=10, *, noise=0.0, random_state=None):
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
    y = groundtruth.predict(X) + noise * generator.standard_normal(size=(n_samples))

    return X, y


def generate_data(
    n_train: int,
    n_test: int,
    snr: float,
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
    snr : float
        Signal-to-noise-ratio defining the amount of noise to add to the data.
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
        noise=_get_noise_sd_from_snr(snr, _make_friedman1),
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)

    return X_train, y_train, X_test, y_test


def _get_noise_sd_from_snr(snr: int, generate_data_func: Callable) -> float:
    """Calculate noise standard deviation from signal-to-noise ratio.

    Parameters
    ----------
    snr : int
        Signal-to-noise ratio.
    generate_data_func : Callable
        Function to generate data with argument n_samples
        and returning features X and target values y.

    Returns
    -------
    float
        Noise standard deviation.
    """
    _, y = generate_data_func(100000)

    signal_std = np.std(y)
    noise_std = signal_std / snr

    return noise_std
