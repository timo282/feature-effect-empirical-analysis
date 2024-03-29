from typing_extensions import Literal
from datetime import datetime
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def objective(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    trial: optuna.trial.Trial,
    cv: int,
    metric: str,
) -> float:
    if isinstance(model, RandomForestRegressor):
        # using the values from https://www.jmlr.org/papers/v20/18-444.html
        hyperparams = dict()
        hyperparams["n_estimators"] = trial.suggest_int(
            "n_estimators", 200, 1750
        )
        hyperparams["max_depth"] = trial.suggest_int("max_depth", 2, 30)
        hyperparams["min_samples_split"] = trial.suggest_float(
            "min_samples_split", 0.01, 0.5
        )
        hyperparams["bootstrap"] = trial.suggest_categorical(
            "bootstrap", [True, False]
        )
        hyperparams["max_samples"] = (
            trial.suggest_float("max_samples", 0.3, 0.975)
            if hyperparams["bootstrap"]
            else trial.suggest_categorical("max_samples", [None])
        )
        hyperparams["max_features"] = trial.suggest_float(
            "max_features", 0.035, 0.7
        )

        model.set_params(**hyperparams, random_state=42)
    else:
        raise NotImplementedError("Base estimator not implemented yet")

    score = cross_val_score(
        model, X_train, y_train, cv=cv, scoring=metric
    ).mean()

    return score


def optimize(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
    cv: int,
    metric: str,
    direction: Literal["maximize", "minimize"],
) -> optuna.study.Study:
    model_name = model.__class__.__name__
    study = optuna.create_study(
        storage=f"sqlite:///tuning/{model_name}.db",
        study_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction=direction,
    )

    study.optimize(
        lambda trial: objective(model, X_train, y_train, trial, cv, metric),
        n_trials=n_trials,
    )

    def obj(trial: optuna.trial.Trial):
        return objective(model, X_train, y_train, trial, cv, metric)

    study.optimize(obj, n_trials=n_trials, n_jobs=-1)

    return study


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
    cv: int,
    metric: str,
    direction: Literal["maximize", "minimize"],
) -> BaseEstimator:
    study = optimize(
        model=model,
        X_train=X_train,
        y_train=y_train,
        n_trials=n_trials,
        cv=cv,
        metric=metric,
        direction=direction,
    )

    hyperparams = study.best_params
    model.set_params(**hyperparams)

    model.fit(X_train, y_train)

    return model
