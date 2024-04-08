from typing_extensions import Literal, List
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import seaborn as sns
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator

from feature_effect_empirical_analysis.plotting.utils import (
    set_style,
    get_boxplot_style,
    get_feature_effect_plot_style,
)


def boxplot_model_results(
    metric: Literal["mse", "mae", "r2"], df: pd.DataFrame
) -> plt.Figure:
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
    plt.legend(
        title="Learner", bbox_to_anchor=(1.05, 1), loc="upper left"
    ).set_visible(True)
    fig.tight_layout()

    return fig


def plot_pdp_comparison(
    model: BaseEstimator,
    groundtruth: BaseEstimator,
    X_train: np.ndarray,
    features: List[Literal["x_1", "x_2", "x_3", "x_4", "x_5"]],
) -> plt.Figure:
    set_style()
    feature_indices = [int(feature.split("_")[1]) - 1 for feature in features]
    fig, axes = plt.subplots(
        1, len(features), figsize=(6 * len(features), 6), dpi=300, sharey=True
    )
    fig.suptitle(
        "Partial dependence comparison", fontsize=16, fontweight="bold"
    )
    for i in range(len(features)):
        feature, feature_index = features[i], feature_indices[i]
        pd_model = partial_dependence(
            model,
            X_train,
            features=[feature_index],
            kind="average",
            percentiles=(0, 1),
            grid_resolution=100,
        )
        pd_gt = partial_dependence(
            groundtruth,
            X_train,
            features=[feature_index],
            kind="average",
            percentiles=(0, 1),
            grid_resolution=100,
        )
        axes[i].plot(
            pd_model["grid_values"][0],
            pd_model["average"][0],
            label=model.__class__.__name__,
            **get_feature_effect_plot_style(),
        )
        axes[i].plot(
            pd_gt["grid_values"][0],
            pd_gt["average"][0],
            label="Groundtruth",
            **get_feature_effect_plot_style(),
        )
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Partial dependence")
        deciles = np.percentile(X_train[:, 0], np.arange(10, 101, 10))
        trans = transforms.blended_transform_factory(
            axes[i].transData, axes[i].transAxes
        )
        axes[i].vlines(
            deciles, 0, 0.045, transform=trans, color="k", linewidth=1
        )
        axes[i].legend()

    return fig
