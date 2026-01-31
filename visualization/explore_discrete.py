import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from .style import UNIFORM_BLUE

#--- Helper function to sort counts based on type ---
def get_counts(series, ascending_numeric=True):
    """
    Returns a Series with counts:
    - Numeric series are sorted ascending by value
    - Non-numeric series are sorted ascending by frequency
    """
    if pd.api.types.is_numeric_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return series.value_counts().sort_index(ascending=ascending_numeric)
    else:
        return series.value_counts().sort_values(ascending=True)

#--- Function : plot_discrete_distribution ---
def plot_discrete_distribution(df, discrete_cols, figsize=(10,4)):
    """
    Simple bar plot for discrete variables with automatic orientation.
    """
    for col in discrete_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        counts = get_counts(series)
        n_cat = len(counts)

        fig, ax = plt.subplots(figsize=figsize)

        if n_cat > 8:
            bars = ax.barh(counts.index.astype(str), counts.values, color=UNIFORM_BLUE, 
                          edgecolor="black", linewidth=1)
            ax.set_title(f"Distribution of {col}", fontweight='bold')
            ax.set_xlabel("Count")
            for bar in bars:
                ax.text(bar.get_width() + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{int(bar.get_width())}", va="center", fontsize=9)
        else:
            bars = ax.bar(counts.index.astype(str), counts.values, color=UNIFORM_BLUE, 
                         edgecolor="black", linewidth=1)
            ax.set_title(f"Distribution of {col}", fontweight='bold')
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f"{int(bar.get_height())}", ha="center", va="center", fontsize=9)

        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : plot_discrete_distribution_grid ---
def plot_discrete_distribution_grid(df, discrete_cols, n_cols=2, figsize=(12,8)):
    """
    Bar plots for multiple discrete variables arranged in a grid.
    """
    cols = [col for col in discrete_cols if col in df.columns]
    if not cols: return

    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        series = df[col].dropna()
        counts = get_counts(series)
        n_cat = len(counts)

        if n_cat >= 8:
            bars = ax.barh(counts.index.astype(str), counts.values, color=UNIFORM_BLUE, 
                          edgecolor="black", linewidth=1)
            for bar in bars:
                ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        f"{int(bar.get_width())}", ha="center", va="center", fontsize=9)
        else:
            bars = ax.bar(counts.index.astype(str), counts.values, color=UNIFORM_BLUE, 
                         edgecolor="black", linewidth=1)
            ax.tick_params(axis="x", rotation=45)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f"{int(bar.get_height())}", ha="center", va="center", fontsize=9)

        ax.set_title(col, fontweight='bold')

    for j in range(len(cols), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()
    plt.close()

#---Function : plot_discrete_dot_distribution---
def plot_discrete_dot_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):
    """
    Dot plot for discrete variables.
    """
    for col in discrete_cols:
        if col not in df.columns: continue
        series = df[col].dropna()
        counts = get_counts(series)
        if normalize:
            counts = counts / counts.sum()
            label = "Proportion"
        else:
            label = "Count"
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(counts.values, counts.index.astype(str), 'o', color=UNIFORM_BLUE)
        for x, y in zip(counts.values, counts.index.astype(str)):
            ax.text(x, y, f" {x:.2f}" if normalize else f" {int(x)}", va="center", fontsize=9)
        ax.set_title(f"Dot plot of {col}", fontweight='bold')
        ax.set_xlabel(label)
        plt.tight_layout(); plt.show(); plt.close()

#---Function : plot_discrete_lollipop_distribution---
def plot_discrete_lollipop_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):
    """
    Lollipop plot for discrete variables.
    """
    for col in discrete_cols:
        if col not in df.columns: continue
        series = df[col].dropna()
        counts = get_counts(series)
        fmt = "{:.2f}" if normalize else "{:.0f}"
        if normalize:
            counts = counts / counts.sum()
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.hlines(y=counts.index.astype(str), xmin=0, xmax=counts.values, color="gray", linewidth=1)
        ax.plot(counts.values, counts.index.astype(str), 'o', color=UNIFORM_BLUE)
        for y, x in zip(counts.index.astype(str), counts.values):
            ax.text(x, y, f" {fmt.format(x)}", va="center", fontsize=9)
        ax.set_title(f"Lollipop plot of {col}", fontweight='bold')
        ax.set_xlabel("Proportion" if normalize else "Count")
        plt.tight_layout(); plt.show(); plt.close()
