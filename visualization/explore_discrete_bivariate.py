import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --- Helper function to sort counts based on type ---
def get_counts(series, ascending_numeric=True):
    """
    Returns a Series with counts:
    - Numeric series are sorted ascending by value
    - Non-numeric series are sorted ascending by frequency
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.value_counts().sort_index(ascending=ascending_numeric)
    else:
        return series.value_counts().sort_values(ascending=True)

# --- Function: Bar plot for bivariate discrete ---
def plot_discrete_bivariate(df, category_col, hue_col, figsize=(8,4)):
    """
    Bar plot for two categorical variables.
    category_col: main x-axis variable
    hue_col: grouping variable (different colors)
    Labels centered inside bars, colors fixed: blue and green.
    """
    if category_col not in df.columns or hue_col not in df.columns:
        print("Columns not found. Skipping.")
        return

    series = df[[category_col, hue_col]].dropna()
    counts = series.groupby([category_col, hue_col]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.35
    indices = np.arange(len(counts))

    colors = ["#ADD8E6", "#90EE90"]  # Blue and green

    for i, hue in enumerate(counts.columns):
        ax.bar(indices + i*bar_width, counts[hue], width=bar_width, color=colors[i], edgecolor="black")
        # Labels inside bars
        for idx, val in enumerate(counts[hue]):
            ax.text(indices[idx] + i*bar_width + bar_width/2,
                    val/2,
                    str(val),
                    ha="center", va="center", fontsize=9, color="black")

    ax.set_xticks(indices + bar_width/2)
    ax.set_xticklabels(counts.index.astype(str), rotation=45)
    ax.set_xlabel(category_col)
    ax.set_ylabel("Count")
    ax.set_title(f"{category_col} by {hue_col}")

    ax.legend(counts.columns)
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function: Grid bar plot for multiple bivariate discrete ---
def plot_discrete_bivariate_grid(df, category_cols, hue_col, n_cols=2, figsize=(12,8)):
    """
    Grid of bar plots for multiple categorical columns by a single hue variable.
    """
    cols = [col for col in category_cols if col in df.columns]
    if not cols or hue_col not in df.columns:
        print("Columns not found. Skipping.")
        return

    n_rows = math.ceil(len(cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(cols) > 1 else [axes]

    for ax, col in zip(axes, cols):
        series = df[[col, hue_col]].dropna()
        counts = series.groupby([col, hue_col]).size().unstack(fill_value=0)
        bar_width = 0.35
        indices = np.arange(len(counts))
        colors = ["#ADD8E6", "#90EE90"]

        for i, hue in enumerate(counts.columns):
            ax.bar(indices + i*bar_width, counts[hue], width=bar_width, color=colors[i], edgecolor="black")
            # Labels inside bars
            for idx, val in enumerate(counts[hue]):
                ax.text(indices[idx] + i*bar_width + bar_width/2,
                        val/2,
                        str(val),
                        ha="center", va="center", fontsize=9, color="black")

        ax.set_xticks(indices + bar_width/2)
        ax.set_xticklabels(counts.index.astype(str), rotation=45)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"{col} by {hue_col}")
        ax.legend(counts.columns)

    # Remove empty axes
    for i in range(len(cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function: Dot plot for bivariate discrete ---
def plot_discrete_dot_bivariate(df, category_col, hue_col, normalize=True, figsize=(8,4)):
    """
    Dot plot for two categorical variables.
    Numeric main variables sorted ascending, categorical by frequency.
    """
    if category_col not in df.columns or hue_col not in df.columns:
        print("Columns not found. Skipping.")
        return

    series = df[[category_col, hue_col]].dropna()
    counts = series.groupby([category_col, hue_col]).size().unstack(fill_value=0)

    if normalize:
        counts = counts / counts.sum()
        xlabel = "Proportion"
    else:
        xlabel = "Count"

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#ADD8E6", "#90EE90"]
    for i, hue in enumerate(counts.columns):
        ax.plot(counts[hue].values, counts.index.astype(str), 'o', color=colors[i])
        for x, y in zip(counts[hue].values, counts.index.astype(str)):
            ax.text(x, y, f" {x:.2f}" if normalize else f" {int(x)}",
                    va="center", fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(category_col)
    ax.set_title(f"{category_col} by {hue_col}")
    ax.legend(counts.columns)
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function: Lollipop plot for bivariate discrete ---
def plot_discrete_lollipop_bivariate(df, category_col, hue_col, normalize=True, figsize=(8,4)):
    """
    Lollipop plot for two categorical variables.
    Numeric main variables sorted ascending, categorical by frequency.
    """
    if category_col not in df.columns or hue_col not in df.columns:
        print("Columns not found. Skipping.")
        return

    series = df[[category_col, hue_col]].dropna()
    counts = series.groupby([category_col, hue_col]).size().unstack(fill_value=0)

    if normalize:
        counts = counts / counts.sum()
        xlabel = "Proportion"
        fmt = "{:.2f}"
    else:
        xlabel = "Count"
        fmt = "{:.0f}"

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#ADD8E6", "#90EE90"]
    bar_width = 0.2
    indices = np.arange(len(counts))

    for i, hue in enumerate(counts.columns):
        ax.hlines(y=indices + i*bar_width, xmin=0, xmax=counts[hue], color="gray", linewidth=1)
        ax.plot(counts[hue], indices + i*bar_width, 'o', color=colors[i])
        for idx, val in enumerate(counts[hue]):
            ax.text(val, indices[idx] + i*bar_width, f" {fmt.format(val)}", va="center", fontsize=9)

    ax.set_yticks(indices + bar_width/2)
    ax.set_yticklabels(counts.index.astype(str))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(category_col)
    ax.set_title(f"{category_col} by {hue_col}")
    ax.legend(counts.columns)
    plt.tight_layout()
    plt.show()
    plt.close()
