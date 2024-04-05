from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
import optuna


def map_modelname_to_estimator(model_name: str) -> BaseEstimator:
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor()
    raise NotImplementedError("Base estimator not implemented yet")


def suggested_hps_for_model(
    model: BaseEstimator, trial: optuna.trial.Trial
) -> dict:
    if isinstance(model, RandomForestRegressor):
        return _suggest_hps_rf(trial)
    raise NotImplementedError("Base estimator not implemented yet")


def _suggest_hps_rf(trial: optuna.trial.Trial):
    # using the values from https://www.jmlr.org/papers/v20/18-444.html
    hyperparams = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1750),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_float(
            "min_samples_split", 0.01, 0.5
        ),
        "max_samples": trial.suggest_float("max_samples", 0.3, 0.975),
        "max_features": trial.suggest_float("max_features", 0.035, 0.7),
    }

    return hyperparams
