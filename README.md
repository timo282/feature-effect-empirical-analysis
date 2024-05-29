# Empirical Analysis of Feature Effects

Explainable AI / Interpretable ML Research Project at LMU Munich: Quantifying the the feature effect errors of PDP and ALE empirically through simulation studies.

## Installation
To install and use/contribute to this project, it is recommended to follow the following steps:

1. Clone or fork this repository

2. Install pipx:
```
pip install --user pipx
```

3. Install Poetry:
```
pipx install poetry
```

3a. *Optionally (if you want the env-folder to be created in your project)*:
```
poetry config virtualenvs.in-project true
```

4. Install this project:
```
poetry install
```

4a. If the specified Python Version is not found:

If the required Python Version is not installed on your device, install Python from the official [python.org](https://www.python.org/downloads) website.

Then run
```
poetry env use <path-to-your-python-version>
```

## Usage & Helpful Information

To run a simulation, specify your configuration/parameters for the simulation in the `config.ini` file. Then go to the `feature_effect_empirical_analysis` directory, which is the python project and therefore the core of the repository. Run the `main.py` script, which is the entry point of the project. It will automatically run a simulation based on the parameters you have provided in the `config.ini`. It will create a new subdirectory in `simulations` based on the simulation name you specified in the config, it will copy the config-file to this directory, and store all simulation results (trained models as joblib-files, model results, tuning trials, and feature effects results in database-files). To analyze the results after your simulation is completed, you can copy the template notebooks from the `notebook_templates` directory into your simulation results and fill in the missing information.

Random seeds are used throughout the entire simulation to make all results reproducible. Datasets created in each simulation run are not stored, but can simply be recreated afterwards if needed by using the number of the simulation run as seed for the dataset generation.

During development, to add new packages to the project, use:
```
poetry add <package-name>
```

If you want to convert your notebooks to presentations (html), use the following command (you may need to install the required extensions first):
```
python -m jupyter nbconvert \notebooks\my_notebook.ipynb --to slides --output-dir \presentations
```
