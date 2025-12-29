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

#--- Bivariate Bar Plot ---
def plot_discrete_bivariate_bar(df, col, hue_col='Gender', figsize=(8,4)):
    """
    Bar plot for discrete variable by hue (e.g., Gender).
    Displays counts inside each bar, centered, with separate colors.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    colors = {'Male': '#ADD8E6', 'Female': '#90EE90'}
    categories = df[col].dropna().unique()
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    for i, gender in enumerate(df[hue_col].unique()):
        counts = df[df[hue_col]==gender][col].value_counts()
        counts = counts.reindex(categories, fill_value=0)
        ax.bar(x + (i-0.5)*width, counts.values, width, label=gender, color=colors[gender], edgecolor='black')

        # Labels inside bars, centered
        for xi, val in zip(x + (i-0.5)*width, counts.values):
            ax.text(xi, val/2, str(val), ha='center', va='center', fontsize=9, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel('Count')
    ax.set_title(f'{col} by {hue_col}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

#--- Bivariate Dot Plot ---
def plot_discrete_dot_bivariate(df, col, hue_col='Gender', normalize=True, figsize=(8,4)):
    """
    Dot plot for discrete variables bivariately.
    Two dots per category (Male/Female), slightly offset vertically.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    colors = {'Male': '#ADD8E6', 'Female': '#90EE90'}
    offsets = {'Male': -0.15, 'Female': 0.15}

    categories = df[col].dropna().unique()
    fig, ax = plt.subplots(figsize=figsize)

    for gender in df[hue_col].unique():
        subset = df[df[hue_col]==gender]
        counts = subset[col].value_counts()
        counts = counts.reindex(categories, fill_value=0)
        if normalize:
            counts = counts / counts.sum()

        for i, cat in enumerate(counts.index):
            x = counts[cat]
            y = i + offsets[gender]
            ax.plot(x, y, 'o', color=colors[gender], label=gender if i==0 else "")
            ax.text(x, y, f"{x:.2f}" if normalize else f"{int(x)}", va='center', fontsize=9, color='black')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([str(c) for c in categories])
    ax.set_xlabel('Proportion' if normalize else 'Count')
    ax.set_ylabel(col)
    ax.set_title(f'Dot plot of {col} by {hue_col}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

#--- Bivariate Lollipop Plot ---
def plot_discrete_lollipop_bivariate(df, col, hue_col='Gender', normalize=True, figsize=(8,4)):
    """
    Lollipop plot for discrete variables bivariately.
    Two lines per category (Male/Female), slightly offset vertically.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    colors = {'Male': '#ADD8E6', 'Female': '#90EE90'}
    offsets = {'Male': -0.15, 'Female': 0.15}

    categories = df[col].dropna().unique()
    fig, ax = plt.subplots(figsize=figsize)

    for gender in df[hue_col].unique():
        subset = df[df[hue_col]==gender]
        counts = subset[col].value_counts()
        counts = counts.reindex(categories, fill_value=0)
        if normalize:
            counts = counts / counts.sum()
            fmt = "{:.2f}"
        else:
            fmt = "{:.0f}"

        for i, cat in enumerate(counts.index):
            val = counts[cat]
            y = i + offsets[gender]
            ax.hlines(y=y, xmin=0, xmax=val, color=colors[gender], linewidth=4)
            ax.plot(val, y, 'o', color=colors[gender])
            ax.text(val, y, fmt.format(val), va='center', fontsize=9, color='black')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([str(c) for c in categories])
    ax.set_xlabel('Proportion' if normalize else 'Count')
    ax.set_ylabel(col)
    ax.set_title(f'Lollipop plot of {col} by {hue_col}')
    ax.legend(df[hue_col].unique())
    plt.tight_layout()
    plt.show()
    plt.close()
