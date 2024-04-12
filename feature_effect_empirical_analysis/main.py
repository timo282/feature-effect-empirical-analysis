from configparser import ConfigParser
from pathlib import Path
import os
import shutil
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
from feature_effect_empirical_analysis.mappings import (
    map_modelname_to_estimator,
)
from feature_effect_empirical_analysis.feature_effects import (
    compute_pdps,
    compare_pdps,
)

sim_config = ConfigParser()
sim_config.read("config.ini")


def simulate(
    models: List[BaseEstimator],
    n_sim: int,
    n_trains: List[int],
    noise_sds: List[float],
    config: ConfigParser,
):
    np.random.seed(42)
    model_folder = config.get("storage", "models")
    model_results_storage = config.get("storage", "model_results")
    pdp_results_storage = config.get("storage", "pdp_results")
    engine_model_results = create_engine(f"sqlite://{model_results_storage}")
    engine_pdp_results = create_engine(f"sqlite://{pdp_results_storage}")
    n_trials = config.getint("simulation_metadata", "n_tuning_trials")
    cv = config.getint("simulation_metadata", "n_tuning_folds")
    metric = config.get("simulation_metadata", "tuning_metric")
    direction = config.get("simulation_metadata", "tuning_direction")
    tuning_studies_folder = config.get("storage", "tuning_studies_folder")
    n_test = config.getint("simulation_metadata", "n_test")

    for i in range(n_sim):
        for n_train in n_trains:
            for noise_sd in noise_sds:
                # generate data
                X_train, y_train, X_test, y_test = generate_data(
                    n_train=n_train,
                    n_test=n_test,
                    noise_sd=noise_sd,
                    seed=i,
                )

                # calulate pdps of groundtruth
                groundtruth = Groundtruth()
                pdp_groundtruth = compute_pdps(groundtruth, X_train, config)

                for model in models:
                    model_str = model.__class__.__name__
                    date = datetime.now().strftime("%Y%m%d")
                    model_name = (
                        f"{model_str}_{date}_{i+1}_{n_train}_{noise_sd}"
                    )

                    # train and tune model
                    model = train_model(
                        model,
                        X_train,
                        y_train,
                        n_trials=n_trials,
                        cv=cv,
                        metric=metric,
                        direction=direction,
                        tuning_studies_folder=tuning_studies_folder,
                        model_name=model_name,
                    )

                    # save model
                    os.makedirs(model_folder, exist_ok=True)
                    dump(
                        model,
                        Path(os.getcwd())
                        / model_folder
                        / f"{model_name}.joblib",
                    )

                    # evaluate model
                    model_results = eval_model(
                        model, X_train, y_train, X_test, y_test
                    )

                    df_model_result = pd.DataFrame(
                        {
                            "model_id": [model_name],
                            "model": [model_str],
                            "simulation": [i + 1],
                            "n_train": [n_train],
                            "noise_sd": [noise_sd],
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

                    # calculate pdps
                    pdp = compute_pdps(model, X_train, config)

                    # compare pdps to groundtruth
                    pdp_comparison = compare_pdps(
                        pdp_groundtruth, pdp, mean_squared_error
                    )

                    df_pdp_result = pd.concat(
                        (
                            pd.DataFrame(
                                {
                                    "model_id": [model_name],
                                    "model": [model_str],
                                    "simulation": [i + 1],
                                    "n_train": [n_train],
                                    "noise_sd": [noise_sd],
                                }
                            ),
                            pdp_comparison,
                        ),
                        axis=1,
                    )

                    # save model results
                    df_pdp_result.to_sql(
                        "pdp_results",
                        con=engine_pdp_results,
                        if_exists="append",
                    )


if __name__ == "__main__":
    model_names = sim_config.get("simulation_params", "models").split(",")
    n_sim_config = sim_config.getint("simulation_params", "n_sim")
    n_train_config = [
        int(x)
        for x in sim_config.get("simulation_params", "n_train").split(",")
    ]
    noise_sd_config = [
        float(x)
        for x in sim_config.get("simulation_params", "noise_sd").split(",")
    ]
    models_config = [
        map_modelname_to_estimator(model_name) for model_name in model_names
    ]

    simulation_name = sim_config.get("storage", "simulation_name")

    base_dir = (
        Path(sim_config.get("storage", "simulations_dir")) / simulation_name
    )
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        shutil.copy2("config.ini", base_dir / f"config_{simulation_name}")
        os.chdir(base_dir)
    else:
        raise ValueError(f"Simulation {base_dir} already exists.")

    simulate(
        models=models_config,
        n_sim=n_sim_config,
        n_trains=n_train_config,
        noise_sds=noise_sd_config,
        config=sim_config,
    )
