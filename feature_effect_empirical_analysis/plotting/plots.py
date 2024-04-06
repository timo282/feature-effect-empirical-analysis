from typing_extensions import Literal, List
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_effect_empirical_analysis.plotting.utils import (
    set_style,
    get_boxplot_style,
)


def boxplot_model_results(
    metric: Literal["mse", "mae", "r2"], df: pd.DataFrame
) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100, sharey=True)
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
        figsize=(min(6 * len(features), 18), math.ceil(len(features) / 3) * 6),
        dpi=100,
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
