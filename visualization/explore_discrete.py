import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#--- Function : plot_discrete_distribution ---
def plot_discrete_distribution(df, discrete_cols, bins=10, normalize=True):
    """
    Plot side-by-side visualizations for discrete (non-binary) variables.

    Left: Full discrete distribution (bar plot)
    Right: Binned distribution
    """

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Left: Full discrete distribution
        counts = series.value_counts()
        axes[0].bar(
            counts.index.astype(str),
            counts.values,
            color="#ADD8E6",
            edgecolor="black",
            linewidth=1
        )
        axes[0].set_title(f"Distribution of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=45)

        # Right: Binned distribution
        hist_counts, bin_edges = np.histogram(series, bins=bins)
        if normalize:
            hist_counts = hist_counts / hist_counts.sum()
            ylabel = "Proportion"
        else:
            ylabel = "Count"

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axes[1].bar(
            bin_centers,
            hist_counts,
            width=np.diff(bin_edges),
            color="#ADD8E6",
            edgecolor="black",
            linewidth=1,
            align="center"
        )
        axes[1].set_title(f"{col} (binned)")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel(ylabel)

        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : plot_discrete_dot_distribution ---
def plot_discrete_dot_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):
    """
    Dot plot for discrete variables.
    Each category is represented by a dot (count or proportion).
    """

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_values(ascending=True)

        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
        else:
            xlabel = "Count"

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            counts.values,
            counts.index.astype(str),
            'o',
            color="#1f77b4"
        )

        ax.set_title(f"Dot plot of {col}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#--- Function : plot_discrete_lollipop_distribution ---
def plot_discrete_lollipop_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):
    """
    Lollipop plot for discrete variables.
    Combines a thin line + dot for each category.
    """

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_values(ascending=True)

        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
        else:
            xlabel = "Count"

        fig, ax = plt.subplots(figsize=figsize)

        ax.hlines(
            y=counts.index.astype(str),
            xmin=0,
            xmax=counts.values,
            color="gray",
            linewidth=1
        )
        ax.plot(
            counts.values,
            counts.index.astype(str),
            'o',
            color="#1f77b4"
        )

        ax.set_title(f"Lollipop plot of {col}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()
