from typing_extensions import Literal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_effect_empirical_analysis.plotting.utils import (
    set_style,
    get_boxplot_style,
)


def boxplot_model_results(
    metric: Literal["mse", "mae", "r2"], df: pd.DataFrame
):
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
