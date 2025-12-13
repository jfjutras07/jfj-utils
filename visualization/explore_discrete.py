import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#--- Function : plot_discrete_distribution ---
def plot_discrete_distribution(df, discrete_cols, top_k=10, bins=10, normalize=True):
    """
    Plot side-by-side visualizations for discrete (non-binary) variables.

    For each column:
    - Left: Top-k most frequent values (bar plot)
    - Right: Binned distribution
    """

    plt.style.use('seaborn-v0_8')

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        #Left: Top-k values
        top_counts = series.value_counts().head(top_k)
        axes[0].bar(top_counts.index.astype(str), top_counts.values)
        axes[0].set_title(f"Top {top_k} values of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=45)

        #Right: Binned distribution
        counts, bin_edges = np.histogram(series, bins=bins)

        if normalize:
            counts = counts / counts.sum()
            ylabel = "Proportion"
        else:
            ylabel = "Count"

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        axes[1].bar(
            bin_centers,
            counts,
            width=np.diff(bin_edges),
            align="center",
            facecolor="#ADD8E6",   # light blue (explicit, stable)
            edgecolor="black",
            linewidth=1
        )

        axes[1].set_title(f"Distribution of {col} (binned)")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel(ylabel)

        plt.tight_layout()
        plt.show()
        plt.close()
