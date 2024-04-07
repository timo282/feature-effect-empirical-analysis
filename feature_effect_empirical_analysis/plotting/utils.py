import matplotlib.pyplot as plt


def set_style():
    plt.style.use("fivethirtyeight")

    plt.rcParams.update(
        {
            "axes.titlesize": 12,  # Smaller title size
            "axes.labelsize": 10,  # Smaller axis labels size
            "xtick.labelsize": 8,  # Smaller x-axis tick labels size
            "ytick.labelsize": 8,  # Smaller y-axis tick labels size
            "legend.fontsize": 10,  # Smaller legend font size
            "font.size": 10,  # This sets the overall default font size
            "grid.linewidth": 0.5,  # Thin grid lines
            "figure.facecolor": "white",  # White background color
            "axes.facecolor": "white",  # White background color
            "axes.edgecolor": "white",  # White background edge color
        }
    )


def get_boxplot_style():
    style = dict(
        boxprops=dict(edgecolor="black"),  # Box properties
        whiskerprops=dict(color="black"),  # Whisker properties
        capprops=dict(color="black"),  # Cap properties
        medianprops=dict(color="black", linewidth=1.5),  # Median properties
        flierprops=dict(
            marker="o", markeredgecolor="black", markersize=5, linestyle="none"
        ),
        palette="Set2",
    )

    return style


def get_feature_effect_plot_style():
    style = dict(linewidth=2, marker="+", markeredgewidth=1, markersize=5)

    return style
