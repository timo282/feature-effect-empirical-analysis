from abc import abstractmethod, ABC
from typing import Callable, Literal, List, Optional, Tuple
from scipy.stats import norm, uniform
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


class Groundtruth(ABC, BaseEstimator):
    """
    A wrapper class for a groundtruth function wrapped as fitted sklearn
    regression estimator adhering to the standard scikit-learn estimator
    interface.

    Attributes
    ----------
    _is_fitted__ : bool
        Indicates whether the estimator has been 'fitted'. This is a mock
        attribute and is set to True by default.
    _estimator_type : str
        Defines the type of the estimator as 'regressor'.
    marginal_distributions : List[Tuple[Literal["normal", "uniform"], Tuple]]
        Marginal distributions of the features. Each tuple contains the
        distribution type and its parameters. Supported distributions and
        parameters:
        - 'normal': (mean, std)
        - 'uniform': (low, high)

    correlation_matrix : np.ndarray
        Correlation matrix of the features. If None, features are independent.
    n_features : int
        Number of features.
    feature_names : List[str]
        Names of the features.
    """

    def __init__(self):
        self._is_fitted__ = True
        self._estimator_type = "regressor"

    def __sklearn_is_fitted__(self):
        return self._is_fitted__

    @property
    @abstractmethod
    def marginal_distributions(self) -> List[Tuple[Literal["normal", "uniform"], Tuple]]:
        """Marginal distributions of the features."""

    @property
    @abstractmethod
    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Correlation matrix of the features."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of features."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Names of the features."""

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

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """

    @abstractmethod
    def get_theoretical_partial_dependence(self, feature: str) -> Callable:
        """Get the theoretical partial dependence function for a feature.

        Parameters
        ----------
        feature : str
            The feature for which to compute the partial dependence function.

        Returns
        -------
        Callable
            The theoretical partial dependence function for the feature.
        """

    @abstractmethod
    def __str__(self):
        """Return dataset name as string."""


def _transform_to_target_distribution(
    data: np.ndarray, dist_type: Literal["normal", "uniform"], params: Tuple
) -> np.ndarray:
    """
    Transform standard normal data to a target distribution using inverse CDF.

    Parameters
    ----------
    data : np.ndarray
        The data to transform.
    dist_type : str
        The target distribution type. Supported distributions are 'normal' and 'uniform'.
    params : tuple
        Tuple containing the distribution parameters.

    Returns
    -------
    np.ndarray
        The transformed data.
    """
    if dist_type == "normal":
        mean, std = params
        return norm.ppf(norm.cdf(data)) * std + mean
    elif dist_type == "uniform":
        low, high = params
        # Transform standard normal data to uniform
        return uniform.ppf(norm.cdf(data), loc=low, scale=high - low)
    else:
        raise ValueError(f"Unsupported distribution type {dist_type}")


def _generate_samples(
    groundtruth: Groundtruth,
    n_samples: int,
    noise_sd: float,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for specified groundtruth and noise standard deviation.

    Parameters
    ----------
    groundtruth : Groundtruth
        A Groundtruth object from which to predict the response variable.
    n_samples : int
        Number of samples to generate.
    noise_sd : float
        Standard deviation of the Gaussian noise added to the output.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        The generated features matrix with shape (n_samples, groundtruth.n_features).
    y : np.ndarray
        The generated response variable with added noise.
    """
    generator = check_random_state(random_state)
    X = generator.normal(0, 1, size=(n_samples, groundtruth.n_features))

    # Apply the correlation matrix using Cholesky decomposition
    if groundtruth.correlation_matrix is not None:
        L = np.linalg.cholesky(groundtruth.correlation_matrix)
        X = X @ L.T

    # Transform each column to the desired distribution
    for i in range(groundtruth.n_features):
        dist_type, params = groundtruth.marginal_distributions[i]
        X[:, i] = _transform_to_target_distribution(X[:, i], dist_type, params)

    y = groundtruth.predict(X) + noise_sd * generator.standard_normal(n_samples)
    return X, y


def generate_data(
    groundtruth: Groundtruth,
    n_train: int,
    n_test: int,
    snr: float,
    seed: int,
):
    """Generate data for training and testing based on the specified groundtruth,
    and signal-to-noise ratio for noise standard deviation.

    Parameters
    ----------
    groundtruth : Groundtruth
        Groundtruth object to generate the response variable.
    n_train : int
        Number of training samples to generate.
    n_test : int
        Number of test samples to generate.
    snr : float
        Signal-to-noise-ratio defining the amount of noise to add to the data.
    seed : int
        Random seed to use for reproducibility.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        The generated training and test data.
    """

    X, y = _generate_samples(
        groundtruth=groundtruth,
        n_samples=n_train + n_test,
        noise_sd=_get_noise_sd_from_snr(snr, groundtruth),
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=42)

    return X_train, y_train, X_test, y_test


def _get_noise_sd_from_snr(snr: int, groundtruth: Groundtruth) -> float:
    """Calculate noise standard deviation from signal-to-noise ratio.

    Parameters
    ----------
    snr : int
        Signal-to-noise ratio.
    groundtruth : Groundtruth
        Groundtruth object to generate the response variable
        (only for range of the response variable for signal to noise ratio).

    Returns
    -------
    float
        Noise standard deviation.
    """
    _, y = _generate_samples(n_samples=100000, groundtruth=groundtruth, noise_sd=0.0)

    signal_std = np.std(y)
    noise_std = signal_std / snr

    return noise_std
