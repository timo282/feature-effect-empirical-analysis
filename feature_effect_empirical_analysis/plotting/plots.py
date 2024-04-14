from typing_extensions import Literal, List
from configparser import ConfigParser
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import seaborn as sns
from sklearn.base import BaseEstimator

from feature_effect_empirical_analysis.plotting.utils import (
    set_style,
    get_boxplot_style,
    get_feature_effect_plot_style,
)
from feature_effect_empirical_analysis.feature_effects import compute_pdps, compute_ales


def boxplot_model_results(metric: Literal["mse", "mae", "r2"], df: pd.DataFrame) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharey=True)
    fig.suptitle("Model evaluation", fontsize=16, fontweight="bold")
    ax[0].set_title(f"{metric} on train set")
    sns.boxplot(
        x="noise_sd",
        y=f"{metric}_train",
        hue="model",
        data=df,
        ax=ax[0],
        **get_boxplot_style(),
    )
    ax[0].legend().set_visible(False)
    sns.boxplot(
        x="noise_sd",
        y=f"{metric}_test",
        hue="model",
        data=df,
        ax=ax[1],
        **get_boxplot_style(),
    )
    ax[1].set_title(f"{metric} on test set")
    ax[1].legend(title="Learner", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    return fig


def boxplot_feature_effect_results(
    features: List[Literal["x_1", "x_2", "x_3", "x_4", "x_5"]],
    df: pd.DataFrame,
    effect_type: Literal["PDP", "ALE"],
) -> plt.Figure:
    set_style()
    fig = plt.figure(
        figsize=(min(6 * len(features), 18), math.ceil(len(features) / 3) * 5),
        dpi=300,
    )
    fig.suptitle(
        f"Feature effect evaluation {effect_type} with {df['metric'].iloc[0]}",
        fontsize=16,
        fontweight="bold",
    )
    for i, feature in enumerate(features):
        if i == 0:
            plt.subplot(
                math.ceil(len(features) / 3),
                min(len(features), 3),
                i + 1,
            )
        else:
            ax = plt.gca()
            plt.subplot(
                math.ceil(len(features) / 3),
                min(len(features), 3),
                i + 1,
                sharey=ax,
            )
        plt.title(f"{effect_type} of {feature}")
        sns.boxplot(
            x="noise_sd",
            y=feature,
            hue="model",
            data=df,
            **get_boxplot_style(),
        )
        plt.legend().set_visible(False)
    plt.legend(title="Learner", bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
    fig.tight_layout()

    return fig


def plot_effect_comparison(
    model: BaseEstimator,
    groundtruth: BaseEstimator,
    X_train: np.ndarray,
    effect: Literal["PDP", "ALE"],
    features: List[Literal["x_1", "x_2", "x_3", "x_4", "x_5"]],
    config: ConfigParser,
) -> plt.Figure:
    set_style()
    if effect == "PDP":
        effect_func = compute_pdps
        title = "Partial dependence"
    elif effect == "ALE":
        effect_func = compute_ales
        title = "Accumulated local effects"
    effects = effect_func(model, X_train, features, config)
    effects_gt = effect_func(groundtruth, X_train, features, config)
    fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 6), dpi=300, sharey=True)
    fig.suptitle(f"{title} comparison", fontsize=16, fontweight="bold")
    for i in range(len(features)):
        if effects[i]["feature"] != features[i]:
            raise ValueError(f"Feature {features[i]} does not match {effects[i]['feature']}")
        axes[i].plot(
            effects[i]["grid_values"],
            effects[i]["effect"],
            label=model.__class__.__name__,
            **get_feature_effect_plot_style(),
        )
        axes[i].plot(
            effects_gt[i]["grid_values"],
            effects_gt[i]["effect"],
            label="Groundtruth",
            **get_feature_effect_plot_style(),
        )
        axes[i].set_xlabel(f"${effects[i]['feature']}$")
        axes[i].set_ylabel(title)
        deciles = np.percentile(X_train[:, 0], np.arange(10, 101, 10))
        trans = transforms.blended_transform_factory(axes[i].transData, axes[i].transAxes)
        axes[i].vlines(deciles, 0, 0.045, transform=trans, color="k", linewidth=1)
        axes[i].legend()

    return fig
