import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#--- Function : plot_discrete_distribution ---
def plot_discrete_distribution(df, discrete_cols, figsize=(10,4)):
    """
    Simple bar plot for discrete (non-binary) variables.
    Displays counts on top of each bar.
    """

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts()

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(
            counts.index.astype(str),
            counts.values,
            color="#ADD8E6",
            edgecolor="black",
            linewidth=1
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        # Data labels
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : plot_discrete_distribution_grid ---
def plot_discrete_distribution_grid(df, discrete_cols, n_cols=2, figsize=(14, 8)):
    """
    Bar plots for multiple discrete variables displayed in a grid (default 2x2).
    Displays counts on top of each bar.
    """

    # Keep only existing columns
    cols = [col for col in discrete_cols if col in df.columns]
    if not cols:
        print("No valid columns provided.")
        return

    n_rows = math.ceil(len(cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(cols) > 1 else [axes]

    for ax, col in zip(axes, cols):
        series = df[col].dropna()
        counts = series.value_counts()

        bars = ax.bar(
            counts.index.astype(str),
            counts.values,
            color="#ADD8E6",
            edgecolor="black",
            linewidth=1
        )

        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        # Data labels
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # Remove empty subplots
    for ax in axes[len(cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

#--- Function : plot_discrete_dot_distribution ---
def plot_discrete_dot_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_values(ascending=True)

        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
            fmt = "{:.2f}"
        else:
            xlabel = "Count"
            fmt = "{:.0f}"

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(counts.values, counts.index.astype(str), 'o', color="#1f77b4")

        for y, x in zip(counts.index.astype(str), counts.values):
            ax.text(x, y, f" {fmt.format(x)}", va="center", fontsize=9)

        ax.set_title(f"Dot plot of {col}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : plot_discrete_lollipop_distribution ---
def plot_discrete_lollipop_distribution(df, discrete_cols, normalize=True, figsize=(10,4)):

    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_values(ascending=True)

        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
            fmt = "{:.2f}"
        else:
            xlabel = "Count"
            fmt = "{:.0f}"

        fig, ax = plt.subplots(figsize=figsize)

        ax.hlines(
            y=counts.index.astype(str),
            xmin=0,
            xmax=counts.values,
            color="gray",
            linewidth=1
        )
        ax.plot(counts.values, counts.index.astype(str), 'o', color="#1f77b4")

        for y, x in zip(counts.index.astype(str), counts.values):
            ax.text(x, y, f" {fmt.format(x)}", va="center", fontsize=9)

        ax.set_title(f"Lollipop plot of {col}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()
