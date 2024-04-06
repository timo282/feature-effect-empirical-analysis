from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna


def map_modelname_to_estimator(model_name: str) -> BaseEstimator:
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor()
    elif model_name == "XGBRegressor":
        return XGBRegressor()
    raise NotImplementedError("Base estimator not implemented yet")


def suggested_hps_for_model(
    model: BaseEstimator, trial: optuna.trial.Trial
) -> dict:
    if isinstance(model, RandomForestRegressor):
        return _suggest_hps_rf(trial)
    elif isinstance(model, XGBRegressor):
        return _suggest_hps_xgboost(trial)
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


def _suggest_hps_xgboost(trial: optuna.trial.Trial):
    hyperparams = {
        "n_estimators": trial.suggest_int("n_estimators", 920, 4550),
        "max_depth": trial.suggest_int("max_depth", 5, 14),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.002, 0.355, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.545, 0.958),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.295, 6.984, log=True
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.419, 0.864
        ),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.335, 0.886
        ),
        "lambda": trial.suggest_float("lambda", 0.008, 29.755, log=True),
        "alpha": trial.suggest_float("alpha", 0.002, 6.105, log=True),
    }

    return hyperparams
