import matplotlib.pyplot as plt
import warnings
import math
import pandas as pd

# Importing centralized style constants
from .style import (
    UNIFORM_BLUE,
    PALE_PINK,
    WHITE,
    GREY_DARK,
    DEFAULT_FIGSIZE,
    GENDER_PALETTE
)

warnings.filterwarnings("ignore")

#---Function: plot_binary_distribution---
def plot_binary_distribution(df, binary_cols, figsize=None):
    """
    Plot binary distributions as pie charts.
    Each binary variable gets two plots (proportion, counts).
    Two variables are displayed per row (4 plots total).
    If an odd number of variables is provided, the last row is half-filled.
    """

    # Keep only existing columns to prevent errors
    binary_cols = [col for col in binary_cols if col in df.columns]
    n_cols = len(binary_cols)

    if n_cols == 0:
        print("No valid binary columns provided.")
        return

    # Calculate grid dimensions
    n_rows = math.ceil(n_cols / 2)
    plot_figsize = figsize or (14, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, 4, figsize=plot_figsize)

    # Ensure axes is always a 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(binary_cols):
        row = i // 2
        base_col = (i % 2) * 2

        # Data preparation
        series = df[col].dropna()
        counts = series.value_counts().sort_index()
        labels = [str(val) for val in counts.index]
        sizes = counts.values
        total = sizes.sum()

        # Semantic color mapping (prevents inversion)
        if col.lower() == "gender":
            colors = [GENDER_PALETTE[label] for label in labels]
        else:
            colors = [UNIFORM_BLUE, PALE_PINK]

        wedge_style = {'edgecolor': WHITE, 'linewidth': 1.5}

        # Proportion plot
        axes[row, base_col].pie(
            sizes,
            labels=labels,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=140,
            colors=colors,
            wedgeprops=wedge_style,
            textprops={'color': GREY_DARK, 'weight': 'bold'}
        )
        axes[row, base_col].set_title(f"{col}\nProportion", color=GREY_DARK, pad=10)

        # Count plot
        axes[row, base_col + 1].pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{int(round(p / 100 * total))}",
            startangle=140,
            colors=colors,
            wedgeprops=wedge_style,
            textprops={'color': GREY_DARK, 'weight': 'bold'}
        )
        axes[row, base_col + 1].set_title(f"{col}\nCounts", color=GREY_DARK, pad=10)

    # Turn off unused axes
    for j in range(n_cols * 2, n_rows * 4):
        axes.flatten()[j].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
