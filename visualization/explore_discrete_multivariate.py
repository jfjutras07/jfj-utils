import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --- Helper function ---
def get_counts(series, ascending_numeric=True):
    """Return counts for a series. Numeric: sorted ascending; categorical: sorted by frequency."""
    if pd.api.types.is_numeric_dtype(series):
        return series.value_counts().sort_index(ascending=ascending_numeric)
    else:
        return series.value_counts().sort_values(ascending=True)

# --- Function: plot_discrete_bivariate ---
def plot_discrete_bivariate(df, col, hue_col, figsize=(8,4), colors=None):
    """Bar plot of a discrete variable grouped by hue_col."""
    if colors is None:
        colors = ["#ADD8E6", "#90EE90"]
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    categories = sorted(df[col].dropna().unique())
    hue_values = sorted(df[hue_col].dropna().unique())
    x = np.arange(len(categories))
    width = 0.8 / len(hue_values)

    fig, ax = plt.subplots(figsize=figsize)
    for i, val in enumerate(hue_values):
        counts = df[df[hue_col]==val][col].value_counts().reindex(categories, fill_value=0)
        ax.bar(x + i*width, counts.values, width=width, label=str(val), color=colors[i % len(colors)], edgecolor="black")
        for xi, c in zip(x + i*width, counts.values):
            ax.text(xi, c/2, str(c), ha="center", va="center", fontsize=9, color="black")

    ax.set_xticks(x + width*(len(hue_values)-1)/2)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel("Count")
    ax.set_title(f"{col} by {hue_col}")
    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function: plot_discrete_bivariate_grid ---
def plot_discrete_bivariate_grid(df, discrete_cols, hue_col, n_cols=2, figsize=(12,8), colors=None):
    """Grid of bivariate bar plots for multiple discrete variables."""
    if colors is None:
        colors = ["#ADD8E6", "#90EE90"]
    cols = [c for c in discrete_cols if c in df.columns]
    if not cols:
        print("No valid columns found.")
        return

    n_rows = math.ceil(len(cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(cols) > 1 else [axes]

    for ax, col in zip(axes, cols):
        categories = sorted(df[col].dropna().unique())
        hue_values = sorted(df[hue_col].dropna().unique())
        x = np.arange(len(categories))
        width = 0.8 / len(hue_values)

        for i, val in enumerate(hue_values):
            counts = df[df[hue_col]==val][col].value_counts().reindex(categories, fill_value=0)
            ax.bar(x + i*width, counts.values, width=width, label=str(val), color=colors[i % len(colors)], edgecolor="black")
            for xi, c in zip(x + i*width, counts.values):
                ax.text(xi, c/2, str(c), ha="center", va="center", fontsize=9, color="black")

        ax.set_xticks(x + width*(len(hue_values)-1)/2)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_ylabel("Count")
        ax.set_title(f"{col} by {hue_col}")
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    for i in range(len(cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function : plot_discrete_dot_bivariate ---
def plot_discrete_dot_bivariate(df, col, hue_col, normalize=True, figsize=(8,4), colors=None):
    """
    Dot plot for discrete variable grouped by hue_col.
    Values displayed next to dots; legend outside.
    """
    if colors is None:
        colors = ["#ADD8E6", "#90EE90"]  # pale blue, pale green
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    categories = sorted(df[col].dropna().unique())
    hue_values = sorted(df[hue_col].dropna().unique())

    fig, ax = plt.subplots(figsize=figsize)

    for i, val in enumerate(hue_values):
        series = df[df[hue_col]==val][col].dropna()
        counts = get_counts(series)
        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
            fmt = "{:.2f}"
        else:
            xlabel = "Count"
            fmt = "{:.0f}"

        # Plot dots for all categories
        x_vals = [counts.get(cat,0) for cat in categories]
        ax.plot(x_vals, categories, 'o', color=colors[i % len(colors)], label=str(val))

        # Labels next to dots
        for x, cat in zip(x_vals, categories):
            ax.text(x, cat, f" {fmt.format(x)}", va="center", fontsize=9)

    ax.set_title(f"Dot plot of {col} by {hue_col}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(col)
    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function : plot_discrete_lollipop_bivariate ---
def plot_discrete_lollipop_bivariate(df, col, hue_col, normalize=True, figsize=(8,4), colors=None):
    """
    Lollipop plot for discrete variable grouped by hue_col.
    Lines not superimposed; legend outside.
    """
    if colors is None:
        colors = ["#ADD8E6", "#90EE90"]  # pale blue, pale green
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    categories = sorted(df[col].dropna().unique())
    hue_values = sorted(df[hue_col].dropna().unique())

    fig, ax = plt.subplots(figsize=figsize)

    for i, val in enumerate(hue_values):
        series = df[df[hue_col]==val][col].dropna()
        counts = get_counts(series)
        if normalize:
            counts = counts / counts.sum()
            xlabel = "Proportion"
            fmt = "{:.2f}"
        else:
            xlabel = "Count"
            fmt = "{:.0f}"

        y_pos = np.arange(len(categories)) + (i*0.2)  # small offset to avoid overlap
        x_vals = [counts.get(cat,0) for cat in categories]

        ax.hlines(y=y_pos, xmin=0, xmax=x_vals, color=colors[i % len(colors)], linewidth=2)
        ax.plot(x_vals, y_pos, 'o', color=colors[i % len(colors)], label=str(val))

        # Labels next to lollipops
        for x, y in zip(x_vals, y_pos):
            ax.text(x, y, f" {fmt.format(x)}", va="center", fontsize=9)

    ax.set_yticks(np.arange(len(categories)) + 0.1)
    ax.set_yticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(col)
    ax.set_title(f"Lollipop plot of {col} by {hue_col}")

# --- Function: plot_stacked_grid ---
def plot_stacked_grid(
    df,
    dependent_var,
    group_vars,
    n_rows=2,
    n_cols=2,
    palette='pastel'
):
    """
    Stacked bar plots for a categorical dependent variable across multiple independent variables.
    Layout adapts: single plot takes full space, multiple plots use grid (2x2 default).

    Parameters:
        df: pd.DataFrame
        dependent_var: categorical variable to stack
        group_vars: list of categorical independent variables
        n_rows, n_cols: rows and columns per figure grid
        palette: seaborn color palette
    """
    #Force list behavior
    if isinstance(group_vars, str):
        group_vars = [group_vars]

    plots_per_fig = n_rows * n_cols
    stack_levels = sorted(df[dependent_var].dropna().unique())
    palette_colors = sns.color_palette(palette, n_colors=len(stack_levels))

    for i in range(0, len(group_vars), plots_per_fig):
        batch = group_vars[i:i+plots_per_fig]

        if len(batch) == 1:
            # Single plot takes full figure
            fig, ax = plt.subplots(figsize=(10,6))
            col = batch[0]
            categories = sorted(df[col].dropna().unique())
            pivot = df.groupby([col, dependent_var]).size().unstack(fill_value=0).reindex(columns=stack_levels, index=categories, fill_value=0)
            bottom = np.zeros(len(categories))
            for j, level in enumerate(stack_levels):
                counts = pivot[level].values
                ax.bar(categories, counts, bottom=bottom, label=str(level), color=palette_colors[j], edgecolor='black')
                for xi, c, b in zip(categories, counts, bottom):
                    if c > 0:
                        ax.text(xi, b + c/2, str(c), ha='center', va='center', fontsize=9)
                bottom += counts
            ax.set_ylabel("Count")
            ax.set_title(f"{dependent_var} by {col}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.legend(title=dependent_var, bbox_to_anchor=(1.02,1), loc='upper left')
            plt.tight_layout()
            plt.show()

        else:
            # Multiple plots: use grid
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), sharey=True)
            axes = axes.flatten()
            for ax, col in zip(axes, batch):
                categories = sorted(df[col].dropna().unique())
                pivot = df.groupby([col, dependent_var]).size().unstack(fill_value=0).reindex(columns=stack_levels, index=categories, fill_value=0)
                bottom = np.zeros(len(categories))
                for j, level in enumerate(stack_levels):
                    counts = pivot[level].values
                    ax.bar(categories, counts, bottom=bottom, label=str(level), color=palette_colors[j], edgecolor='black')
                    for xi, c, b in zip(categories, counts, bottom):
                        if c > 0:
                            ax.text(xi, b + c/2, str(c), ha='center', va='center', fontsize=9)
                    bottom += counts
                ax.set_ylabel("Count")
                ax.set_title(f"{dependent_var} by {col}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.5)

            # Remove unused axes
            for j in range(len(batch), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.legend(title=dependent_var, bbox_to_anchor=(1.02,1), loc='upper left')
            plt.show()
