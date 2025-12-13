import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#--- Function : plot_discrete_distribution ---
def plot_discrete_distribution(df, discrete_cols, top_k=10, bins=10, normalize=True):
    """
    Plot side-by-side visualizations for discrete (non-binary) variables.

    For each column:
    - Left: Top-k most frequent values
    - Right: Binned distribution
    """

    plt.style.use("seaborn-v0_8")

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        #Left: Top-k values
        top_counts = series.value_counts().head(top_k)
        sns.barplot(
            x=top_counts.index.astype(str),
            y=top_counts.values,
            color="#ADD8E6",
            edgecolor="black",
            linewidth=1,
            ax=axes[0]
        )
        axes[0].set_title(f"Top {top_k} values of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=45)

        #Right: Binned distribution
        sns.histplot(
            series,
            bins=bins,
            stat="probability" if normalize else "count",
            color="#ADD8E6",
            edgecolor="black",
            linewidth=1,
            ax=axes[1]
        )
        axes[1].set_title(f"Distribution of {col} (binned)")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Proportion" if normalize else "Count")

        plt.tight_layout()
        plt.show()
        plt.close()
