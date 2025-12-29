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

#--- Function : plot_discrete_distribution ---
def plot_discrete_distribution(df, discrete_cols, figsize=(10,4)):
    """
    Simple bar plot for discrete (non-binary) variables.
    Displays counts on top of each bar with dynamic offset.
    """
    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found. Skipping.")
            continue

        series = df[col].dropna()
        counts = get_counts(series)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(counts.index.astype(str), counts.values, color="#ADD8E6",
                      edgecolor="black", linewidth=1)

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        # Dynamic offset to prevent labels from overlapping the chart border
        offset = counts.values.max() * 0.02
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + offset,
                    f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : plot_discrete_distribution_grid ---
def plot_discrete_distribution_grid(df, discrete_cols, n_cols=2, figsize=(12,8)):
    """
    Bar plots for multiple discrete variables arranged in a grid.
    Automatically orders numeric variables ascending.
    Displays counts on top of each bar.
    """
    cols = [col for col in discrete_cols if col in df.columns]
    if not cols:
        print("No valid discrete columns found.")
        return

    n_rows = math.ceil(len(cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(cols) > 1 else [axes]

    for ax, col in zip(axes, cols):
        series = df[col].dropna()
        counts = get_counts(series)

        bars = ax.bar(counts.index.astype(str), counts.values, color="#ADD8E6",
                      edgecolor="black", linewidth=1)

        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        offset = counts.values.max() * 0.02
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + offset,
                    f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)

    #Remove empty subplots
    for i in range(len(cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function : plot_discrete_dot_distribution ---
def plot_discrete_dot_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):
    """
    Dot plot for discrete variables.
    Numeric variables are ordered ascending, categorical by frequency.
    Displays values next to each dot.
    """
    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found. Skipping.")
            continue

        series = df[col].dropna()
        counts = get_counts(series)

        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
        else:
            xlabel = "Count"

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(counts.values, counts.index.astype(str), 'o', color="#1f77b4")

        #Labels next to dots
        for x, y in zip(counts.values, counts.index.astype(str)):
            ax.text(x, y, f" {x:.2f}" if normalize else f" {int(x)}",
                    va="center", fontsize=9)

        ax.set_title(f"Dot plot of {col}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : plot_discrete_lollipop_distribution ---
def plot_discrete_lollipop_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):
    """
    Lollipop plot for discrete variables.
    Numeric variables are ordered ascending, categorical by frequency.
    Displays values next to each lollipop with dynamic offset.
    """
    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found. Skipping.")
            continue

        series = df[col].dropna()
        counts = get_counts(series)

        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
            fmt = "{:.2f}"
        else:
            xlabel = "Count"
            fmt = "{:.0f}"

        fig, ax = plt.subplots(figsize=figsize)
        ax.hlines(y=counts.index.astype(str), xmin=0, xmax=counts.values,
                  color="gray", linewidth=1)
        ax.plot(counts.values, counts.index.astype(str), 'o', color="#1f77b4")

        offset = counts.values.max() * 0.02  # Offset for labels
        for y, x in zip(counts.index.astype(str), counts.values):
            ax.text(x + offset, y, f" {fmt.format(x)}", va="center", fontsize=9)

        ax.set_title(f"Lollipop plot of {col}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()
