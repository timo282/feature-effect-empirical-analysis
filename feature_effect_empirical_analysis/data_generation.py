from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split


def generate_data(
    n_train: int,
    n_test: int,
    noise_sd: float,
    n_features: int = 5,
    seed: int = 0,
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
    n_features : int, optional
        Number of features to generate (all feature >5 are independet from y),
        by default 5.
    seed : int, optional
        Random seed to use for reproducibility, by default 0.

    Returns
    -------
    _type_
        _description_
    """
    X, y = make_friedman1(
        n_samples=n_train + n_test,
        n_features=n_features,
        noise=noise_sd,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42
    )

    return X_train, y_train, X_test, y_test
