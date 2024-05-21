from configparser import ConfigParser
from pathlib import Path
import os
from datetime import datetime
from typing_extensions import List
from joblib import dump
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from feature_effect_empirical_analysis.data_generation import (
    generate_data,
    Groundtruth,
)
from feature_effect_empirical_analysis.model_training import train_model
from feature_effect_empirical_analysis.model_eval import eval_model
from feature_effect_empirical_analysis.utils import (
    parse_sim_params,
    create_and_set_sim_dir,
    parse_storage_and_sim_metadata,
)
from feature_effect_empirical_analysis.feature_effects import (
    compute_pdps,
    # compute_ales,
    compare_effects,
)

sim_config = ConfigParser()
sim_config.read("config.ini")


def simulate(
    models: List[BaseEstimator],
    n_sim: int,
    n_trains: List[int],
    snrs: List[float],
    config: ConfigParser,
):
    np.random.seed(42)
    sim_metatadata = parse_storage_and_sim_metadata(config)
    engine_model_results = create_engine(f"sqlite://{sim_metatadata['model_results_storage']}")
    engine_effects_results = create_engine(f"sqlite://{sim_metatadata['effects_results_storage']}")

    for i in range(n_sim):
        for n_train in n_trains:
            for snr in snrs:
                # generate data
                X_train, y_train, X_test, y_test = generate_data(
                    n_train=n_train,
                    n_test=sim_metatadata["n_test"],
                    snr=snr,
                    seed=i,
                )

                # calulate feature effects of groundtruth
                groundtruth = Groundtruth()
                feature_names = ["x_1", "x_2", "x_3", "x_4", "x_5"]
                if sim_metatadata["groundtruth_feature_effect"] == "theoretical":
                    # ugly workaround to get the grid values
                    grid = [
                        compute_pdps(groundtruth, X_train, feature_names, config)[i]["grid_values"]
                        for i in range(len(feature_names))
                    ]
                    pdp_groundtruth_functions = [
                        groundtruth.get_theoretical_partial_dependence(x, feature_distribution="uniform")
                        for x in feature_names
                    ]
                    pdp_groundtruth = [
                        {
                            "feature": feature_names[i],
                            "grid_values": grid[i],
                            "effect": [pdp_groundtruth_functions[i](p) for p in grid[i]],
                        }
                        for i in range(len(feature_names))
                    ]
                    # ale_groundtruth_functions = []
                    # ale_groundtruth = []
                elif sim_metatadata["groundtruth_feature_effect"] == "empirical":
                    pdp_groundtruth = compute_pdps(groundtruth, X_train, feature_names, config)
                    # ale_groundtruth = compute_ales(groundtruth, X_train, feature_names, config)
                else:
                    raise ValueError(
                        f"Unknown groundtruth feature effect: {sim_metatadata['groundtruth_feature_effect']}"
                    )

                for model in models:
                    model_str = model.__class__.__name__
                    date = datetime.now().strftime("%Y%m%d")
                    model_name = f"{model_str}_{date}_{i+1}_{n_train}_{snr}"

                    # train and tune model
                    model = train_model(
                        model,
                        X_train,
                        y_train,
                        n_trials=sim_metatadata["n_trials"],
                        cv=sim_metatadata["cv"],
                        metric=sim_metatadata["metric"],
                        direction=sim_metatadata["direction"],
                        tuning_studies_folder=sim_metatadata["tuning_studies_folder"],
                        model_name=model_name,
                    )

                    # save model
                    os.makedirs(sim_metatadata["model_folder"], exist_ok=True)
                    dump(
                        model,
                        Path(os.getcwd()) / sim_metatadata["model_folder"] / f"{model_name}.joblib",
                    )

                    # evaluate model
                    model_results = eval_model(model, X_train, y_train, X_test, y_test)
                    df_model_result = pd.DataFrame(
                        {
                            "model_id": [model_name],
                            "model": [model_str],
                            "simulation": [i + 1],
                            "n_train": [n_train],
                            "snr": [snr],
                            "mse_train": [model_results[0]],
                            "mse_test": [model_results[1]],
                            "mae_train": [model_results[2]],
                            "mae_test": [model_results[3]],
                            "r2_train": [model_results[4]],
                            "r2_test": [model_results[5]],
                        }
                    )

                    # save model results
                    os.makedirs("results", exist_ok=True)
                    df_model_result.to_sql(
                        "model_results",
                        con=engine_model_results,
                        if_exists="append",
                    )

                    # calculate and compare pdps to groundtruth
                    pdp = compute_pdps(model, X_train, feature_names, config)
                    pdp_comparison = compare_effects(pdp_groundtruth, pdp, mean_squared_error)
                    df_pdp_result = pd.concat(
                        (
                            pd.DataFrame(
                                {
                                    "model_id": [model_name],
                                    "model": [model_str],
                                    "simulation": [i + 1],
                                    "n_train": [n_train],
                                    "snr": [snr],
                                }
                            ),
                            pdp_comparison,
                        ),
                        axis=1,
                    )

                    # save pdp results
                    df_pdp_result.to_sql(
                        "pdp_results",
                        con=engine_effects_results,
                        if_exists="append",
                    )

                    # # calculate ales and compare to groundtruth
                    # ale = compute_ales(model, X_train, feature_names, config)
                    # ale_comparison = compare_effects(ale_groundtruth, ale, mean_squared_error)
                    # df_ale_result = pd.concat(
                    #     (
                    #         pd.DataFrame(
                    #             {
                    #                 "model_id": [model_name],
                    #                 "model": [model_str],
                    #                 "simulation": [i + 1],
                    #                 "n_train": [n_train],
                    #                 "snr": [snr],
                    #             }
                    #         ),
                    #         ale_comparison,
                    #     ),
                    #     axis=1,
                    # )

                    # # save ale results
                    # df_ale_result.to_sql(
                    #     "ale_results",
                    #     con=engine_effects_results,
                    #     if_exists="append",
                    # )


if __name__ == "__main__":
    sim_params = parse_sim_params(sim_config)

    create_and_set_sim_dir(sim_config)

    simulate(
        models=sim_params["models_config"],
        n_sim=sim_params["n_sim"],
        n_trains=sim_params["n_train"],
        snrs=sim_params["snr"],
        config=sim_config,
    )
