from configparser import ConfigParser
from typing_extensions import List, Dict, Callable
import pandas as pd
import numpy as np
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator


def compute_pdps(
    model: BaseEstimator, X_train: np.ndarray, config: ConfigParser
) -> List[Dict]:
    pdp = []
    percentiles = (
        float(config.get("pdp", "percentiles").split(",")[0]),
        float(config.get("pdp", "percentiles").split(",")[1]),
    )
    grid_resolution = config.getint("pdp", "grid_resolution")
    for feature in range(X_train.shape[1]):
        pdp_feature = partial_dependence(
            model,
            X_train,
            [feature],
            kind="average",
            percentiles=percentiles,
            grid_resolution=grid_resolution,
        )
        pdp.append(
            {
                "feature": feature,
                "grid_values": pdp_feature["grid_values"][0],
                "average": pdp_feature["average"][0],
            }
        )

    return pdp


def compare_pdps(
    pdp_groundtruth: List[Dict], pdp_model: List[Dict], metric: Callable
) -> pd.DataFrame:
    pdp_comparison = {}
    for i, pdp_model_feature in enumerate(pdp_model):
        pdp_groundtruth_feature = pdp_groundtruth[i]
        if pdp_groundtruth_feature["feature"] != pdp_model_feature["feature"]:
            raise ValueError(
                "Features in groundtruth and model PDPs do not match"
            )
        if not np.array_equal(
            pdp_groundtruth_feature["grid_values"],
            pdp_model_feature["grid_values"],
        ):
            raise ValueError(
                "Grid values in groundtruth and model PDPs do not match"
            )
        pdp_comparison[f"x_{pdp_model_feature['feature']+1}"] = metric(
            pdp_groundtruth_feature["average"],
            pdp_model_feature["average"],
        )

    return pd.DataFrame(pdp_comparison, index=[0])
