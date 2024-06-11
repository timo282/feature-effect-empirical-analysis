from abc import abstractmethod
from typing import Callable, Literal, List, Tuple, Optional
import pandas as pd
import numpy as np

from feature_effect_empirical_analysis.data_generating.data_generation import Groundtruth


class Simple2FGroundtruth(Groundtruth):
    """
    An abstract class for simple 2-feature groundtruths.
    """

    def __init__(
        self, marginal_distributions: List[Tuple[Literal["normal", "uniform"], Tuple]], correlation_matrix: np.ndarray
    ):
        super().__init__()
        if len(marginal_distributions) != 2:
            raise ValueError("Simple2FGroundtruth requires exactly 2 marginal distributions.")
        if correlation_matrix.shape != (2, 2):
            raise ValueError("Correlation matrix must be of shape (2, 2).")
        self._marginal_distributions = marginal_distributions
        self._correlation_matrix = correlation_matrix
        self._n_features = 2
        self._feature_names = ["x_1", "x_2"]

    @property
    def marginal_distributions(self) -> List[Tuple[Literal["normal", "uniform"], Tuple]]:
        """Marginal distributions of the features."""
        return self._marginal_distributions

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Correlation matrix of the features."""
        return self._correlation_matrix

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    @property
    def feature_names(self) -> List[str]:
        """Names of the features."""
        return self._feature_names

    @abstractmethod
    def predict(self, X) -> float:
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
    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2"]) -> Callable:
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


class SimpleAdditiveGroundtruth(Simple2FGroundtruth):
    """
    A simple additive groundtruth specified by the formula::
    `g(x) = x_1 + 0.5*x_2^2`
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.shape[1] != 2:
            raise ValueError("SimpleAdditiveGroundtruth requires exactly 2 features.")
        return X[:, 0] + 0.5 * X[:, 1] ** 2

    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2"]) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleAdditiveGroundtruth.")


class SimpleInteractionGroundtruth(Simple2FGroundtruth):
    """
    A simple interaction groundtruth specified by the formula::
    `g(x) = x_1 * x_2$`
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.shape[1] != 2:
            raise ValueError("SimpleInteractionGroundtruth requires exactly 2 features.")
        return X[:, 0] * X[:, 1]

    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2"]) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleInteractionGroundtruth.")


class SimpleCombinedGroundtruth(Simple2FGroundtruth):
    """
    A simple combined additive and interaction groundtruth specified by the formula::
    `g(x) = x_1 + 0.5*x_2^2 + x_1*x_2`
    """

    def predict(self, X) -> np.ndarray:
        """
        Returns target value (y) of the groundtruth for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.shape[1] != 2:
            raise ValueError("SimpleCombinedGroundtruth requires exactly 2 features.")
        return X[:, 0] * 0.5 * X[:, 1] ** 2 + X[:, 0] * X[:, 1]

    def get_theoretical_partial_dependence(self, feature: Literal["x_1", "x_2"]) -> Callable:
        raise NotImplementedError("Theoretical partial dependence not implemented for SimpleCombinedGroundtruth.")
