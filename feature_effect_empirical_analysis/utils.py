from configparser import ConfigParser
from typing import Dict
from pathlib import Path
import os
import shutil

from feature_effect_empirical_analysis.mappings import map_modelname_to_estimator


def parse_sim_params(sim_config: ConfigParser) -> Dict:
    """Parse simulation parameters from configuration file.

    Parameters
    ----------
    sim_config : ConfigParser
        Config containing simulation parameters.

    Returns
    -------
    Dict
        Dictionary of simulation parameters.
    """
    param_dict = {}
    model_names = sim_config.get("simulation_params", "models").split(",")
    param_dict["n_sim"] = sim_config.getint("simulation_params", "n_sim")
    param_dict["n_train"] = [int(x) for x in sim_config.get("simulation_params", "n_train").split(",")]
    param_dict["noise_sd"] = [float(x) for x in sim_config.get("simulation_params", "noise_sd").split(",")]
    param_dict["models_config"] = [map_modelname_to_estimator(model_name) for model_name in model_names]

    return param_dict


def parse_storage_and_sim_metadata(config: ConfigParser) -> Dict:
    """Parse storage and simulation metadata from configuration file.

    Parameters
    ----------
    sim_config : ConfigParser
        Config containing storage and simulation metadata.

    Returns
    -------
    Dict
        Dictionary of storage and simulation metadata.
    """
    metadata_dict = {}
    metadata_dict["model_folder"] = config.get("storage", "models")
    metadata_dict["model_results_storage"] = config.get("storage", "model_results")
    metadata_dict["effects_results_storage"] = config.get("storage", "effects_results")

    metadata_dict["n_trials"] = config.getint("simulation_metadata", "n_tuning_trials")
    metadata_dict["cv"] = config.getint("simulation_metadata", "n_tuning_folds")
    metadata_dict["metric"] = config.get("simulation_metadata", "tuning_metric")
    metadata_dict["direction"] = config.get("simulation_metadata", "tuning_direction")
    metadata_dict["tuning_studies_folder"] = config.get("storage", "tuning_studies_folder")
    metadata_dict["n_test"] = config.getint("simulation_metadata", "n_test")
    metadata_dict["groundtruth_feature_effect"] = config.get("simulation_metadata", "groundtruth_feature_effects")

    return metadata_dict


def create_and_set_sim_dir(sim_config: ConfigParser) -> None:
    """Create and set simulation directory.

    Parameters
    ----------
    sim_config : ConfigParser
        Config containing simulation name and base directory.
    """
    simulation_name = sim_config.get("storage", "simulation_name")

    base_dir = Path(sim_config.get("storage", "simulations_dir")) / simulation_name
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        shutil.copy2("config.ini", base_dir / f"config_{simulation_name}.ini")
        os.chdir(base_dir)
    else:
        raise ValueError(f"Simulation {base_dir} already exists.")
