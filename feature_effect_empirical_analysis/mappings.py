from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor


def map_modelname_to_estimator(model_name: str) -> BaseEstimator:
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor()
    else:
        raise NotImplementedError("Base estimator not implemented yet")


def suggested_hps_for_model(model: BaseEstimator) -> dict:
    # TODO: Implement this function
    if isinstance(model, RandomForestRegressor):
        return dict()
