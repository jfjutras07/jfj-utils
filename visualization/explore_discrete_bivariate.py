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

# --- Function : plot_discrete_bivariate ---
def plot_discrete_bivariate(df, col, hue_col='Gender', figsize=(8,4)):
    """
    Bivariate bar plot for a discrete column vs. a binary categorical column (e.g., Gender).
    Bars side by side, counts inside bars centered. Two colors: blue and pale green.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    counts = df.groupby([col, hue_col]).size().unstack(fill_value=0)
    categories = counts.index.astype(str)
    genders = counts.columns

    bar_width = 0.35
    x = np.arange(len(categories))

    colors = ['#ADD8E6', '#90EE90']  # pale blue, pale green

    fig, ax = plt.subplots(figsize=figsize)

    for i, gender in enumerate(genders):
        ax.bar(x + i*bar_width, counts[gender], width=bar_width,
               color=colors[i], edgecolor='black', label=gender)

        #Labels inside bars, centered
        for j, val in enumerate(counts[gender]):
            ax.text(x[j] + i*bar_width, val/2, str(val),
                    ha='center', va='center', color='black', fontsize=9)

    ax.set_xticks(x + bar_width/2)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {col} by {hue_col}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function : plot_discrete_bivariate_grid ---
def plot_discrete_bivariate_grid(df, discrete_cols, hue_col='Gender', n_cols=2, figsize=(12,8)):
    """
    Plots multiple discrete columns bivariately by a binary categorical column (e.g., Gender)
    arranged in a grid.
    """
    cols = [col for col in discrete_cols if col in df.columns]
    if not cols:
        print("No valid discrete columns found.")
        return

    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(cols) > 1 else [axes]

    for ax, col in zip(axes, cols):
        counts = df.groupby([col, hue_col]).size().unstack(fill_value=0)
        categories = counts.index.astype(str)
        genders = counts.columns

        bar_width = 0.35
        x = np.arange(len(categories))

        colors = ['#ADD8E6', '#90EE90']

        for i, gender in enumerate(genders):
            ax.bar(x + i*bar_width, counts[gender], width=bar_width,
                   color=colors[i], edgecolor='black', label=gender)

            #Labels inside bars, centered
            for j, val in enumerate(counts[gender]):
                ax.text(x[j] + i*bar_width, val/2, str(val),
                        ha='center', va='center', color='black', fontsize=9)

        ax.set_xticks(x + bar_width/2)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_ylabel('Count')
        ax.set_title(col)

    #Remove empty subplots
    for i in range(len(cols), len(axes)):
        fig.delaxes(axes[i])

    #Add single legend for the figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function : plot_discrete_dot_bivariate ---
def plot_discrete_dot_bivariate(df, col, hue_col='Gender', normalize=True, figsize=(8,4)):
    """
    Dot plot for discrete variables bivariately.
    Numeric variables ordered ascending, categorical by frequency.
    Labels next to dots, small offset, colors by hue_col.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    categories = df[col].dropna().unique()
    colors = {'Male': '#ADD8E6', 'Female': '#90EE90'}

    fig, ax = plt.subplots(figsize=figsize)

    for gender in df[hue_col].unique():
        subset = df[df[hue_col] == gender]
        counts = subset[col].value_counts()
        if normalize:
            counts = counts / counts.sum()

        for i, cat in enumerate(counts.index):
            x = counts[cat]
            y = str(cat)
            ax.plot(x, y, 'o', color=colors[gender])
            ax.text(x, y, f" {x:.2f}" if normalize else f" {int(x)}",
                    va='center', fontsize=9, color='black')

    ax.set_title(f'Dot plot of {col} by {hue_col}')
    ax.set_xlabel("Proportion" if normalize else "Count")
    ax.set_ylabel(col)
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Function : plot_discrete_lollipop_bivariate ---
def plot_discrete_lollipop_bivariate(df, col, hue_col='Gender', normalize=True, figsize=(8,4)):
    """
    Lollipop plot for discrete variables bivariately.
    Numeric variables ordered ascending, categorical by frequency.
    Labels next to lollipops, colors by hue_col.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    colors = {'Male': '#ADD8E6', 'Female': '#90EE90'}
    fig, ax = plt.subplots(figsize=figsize)

    for gender in df[hue_col].unique():
        subset = df[df[hue_col] == gender]
        counts = subset[col].value_counts()
        if normalize:
            counts = counts / counts.sum()

        y_positions = np.arange(len(counts))
        for i, cat in enumerate(counts.index):
            val = counts[cat]
            ax.hlines(y=cat, xmin=0, xmax=val, color=colors[gender], linewidth=4)
            ax.plot(val, cat, 'o', color=colors[gender])
            ax.text(val, cat, f" {val:.2f}" if normalize else f" {int(val)}",
                    va='center', fontsize=9, color='black')

    ax.set_title(f'Lollipop plot of {col} by {hue_col}')
    ax.set_xlabel("Proportion" if normalize else "Count")
    ax.set_ylabel(col)
    plt.tight_layout()
    plt.show()
    plt.close()
