from typing_extensions import List
from configparser import ConfigParser
from joblib import dump
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator

from feature_effect_empirical_analysis.data_generation import generate_data
from feature_effect_empirical_analysis.model_training import train_model
from feature_effect_empirical_analysis.model_eval import eval_model
from feature_effect_empirical_analysis.mappings import (
    map_modelname_to_estimator,
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
    df_model_results = pd.DataFrame(
        columns=[
            "model",
            "mse_train",
            "mse_test",
            "mae_train",
            "mae_test",
            "r2_train",
            "r2_test",
        ]
    )

    for i in range(n_sim):
        for n_train in n_trains:
            for noise_sd in noise_sds:
                # generate data
                X_train, y_train, X_test, y_test = generate_data(
                    n_train=n_train,
                    n_test=config.getint("simulation_metadata", "n_test"),
                    noise_sd=noise_sd,
                    seed=i,
                )

                for model in models:
                    # train and tune model
                    model = train_model(
                        model,
                        X_train,
                        y_train,
                        n_trials=config.getint(
                            "simulation_metadata", "n_trials"
                        ),
                        cv=config.getint(
                            "simulation_metadata", "n_tuning_folds"
                        ),
                        metric=config.get(
                            "simulation_metadata", "tuning_metric"
                        ),
                        direction=config.get(
                            "simulation_metadata", "tuning_direction"
                        ),
                    )

                    # save model
                    model_folder = config.get("storage", "models")
                    model_name = model.__class__.__name__
                    dump(
                        model,
                        f"{model_folder}/{model_name}_{datetime.now()}.joblib",
                    )

                    # evaluate model
                    model_results = eval_model(
                        model, X_train, y_train, X_test, y_test
                    )
                    df_model_results = df_model_results.append(
                        {
                            "model": model_name,
                            "simulation": i,
                            "mse_train": model_results[0],
                            "mse_test": model_results[1],
                            "mae_train": model_results[2],
                            "mae_test": model_results[3],
                            "r2_train": model_results[4],
                            "r2_test": model_results[5],
                        },
                        ignore_index=True,
                    )

                    # save model results
                    model_results_storage = config.get(
                        "storage", "model_results"
                    )
                    engine = create_engine(f"sqlite://{model_results_storage}")
                    df_model_results.to_sql(
                        "model_results", con=engine, if_exists="append"
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

    simulate(
        models=models_config,
        n_sim=n_sim_config,
        n_trains=n_train_config,
        noise_sds=noise_sd_config,
        config=sim_config,
    )
