import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
from .style import UNIFORM_BLUE, PALE_PINK, GREY_DARK

warnings.filterwarnings("ignore")

def get_counts(series, ascending_numeric=True):
    """
    Return counts for a series. 
    Numeric: sorted ascending; categorical: sorted by frequency.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.value_counts().sort_index(ascending=ascending_numeric)
    else:
        return series.value_counts().sort_values(ascending=True)

#--- Function : plot_discrete_bivariate_grid ---
def plot_discrete_bivariate_grid(df, discrete_cols, hue_col, n_cols=2, figsize=(12,8), 
                             colors=None, show_proportion=True):
    """
    Grid of bivariate bar plots. Supports single or multiple columns.
    Switches to horizontal bars if a variable has 8 or more categories.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]
    
    if isinstance(discrete_cols, str):
        discrete_cols = [discrete_cols]

    cols = [c for c in discrete_cols if c in df.columns]
    if not cols:
        return

    n_plots = len(cols)
    actual_n_cols = min(n_cols, n_plots)
    n_rows = math.ceil(n_plots / actual_n_cols)
    
    fig, axes = plt.subplots(n_rows, actual_n_cols, figsize=figsize)
    axes = np.array([axes]).flatten() if n_plots > 1 else [axes]

    for ax, col in zip(axes, cols):
        ct = pd.crosstab(df[col], df[hue_col])
        if show_proportion:
            ct = ct.div(ct.sum(axis=1), axis=0)

        categories = ct.index.tolist()
        hue_values = ct.columns.tolist()
        n_cat = len(categories)

        if n_cat >= 8:
            y = np.arange(n_cat)
            height = 0.8 / len(hue_values)
            for i, val in enumerate(hue_values):
                widths = ct[val].values
                ax.barh(y + i * height, widths, height=height, label=str(val),
                        color=colors[i % len(colors)], edgecolor="black")
                for yi, w in zip(y + i * height, widths):
                    text_val = f"{w:.2f}" if show_proportion else str(int(w))
                    ax.text(w + 0.005, yi, text_val, va="center", ha="left", fontsize=9)
            ax.set_yticks(y + height * (len(hue_values) - 1) / 2)
            ax.set_yticklabels(categories)
            ax.set_xlabel("Proportion" if show_proportion else "Count")
        else:
            x = np.arange(n_cat)
            width = 0.8 / len(hue_values)
            for i, val in enumerate(hue_values):
                heights = ct[val].values
                ax.bar(x + i * width, heights, width=width, label=str(val),
                       color=colors[i % len(colors)], edgecolor="black")
                for xi, h in zip(x + i * width, heights):
                    text_val = f"{h:.2f}" if show_proportion else str(int(h))
                    ax.text(xi, h / 2, text_val, ha="center", va="center", fontsize=9, color="black")
            ax.set_xticks(x + width * (len(hue_values) - 1) / 2)
            ax.set_xticklabels(categories, rotation=45)
            ax.set_ylabel("Proportion" if show_proportion else "Count")

        ax.set_title(f"{col} by {hue_col}")
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")

    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

#--- Function : plot_discrete_dot_bivariate ---
def plot_discrete_dot_bivariate(df, col, hue_col, normalize=True, figsize=(8,4), colors=None):
    """Dot plot for discrete variable grouped by hue_col."""
    if colors is None: colors = [UNIFORM_BLUE, PALE_PINK]
    if col not in df.columns or hue_col not in df.columns: return

    categories = sorted(df[col].dropna().unique())
    hue_values = sorted(df[hue_col].dropna().unique())
    fig, ax = plt.subplots(figsize=figsize)

    for i, val in enumerate(hue_values):
        series = df[df[hue_col]==val][col].dropna()
        counts = get_counts(series)
        if normalize:
            counts = counts / counts.sum()
            xlabel, fmt = "Proportion", "{:.2f}"
        else:
            xlabel, fmt = "Count", "{:.0f}"

        x_vals = [counts.get(cat,0) for cat in categories]
        ax.plot(x_vals, categories, 'o', color=colors[i % len(colors)], label=str(val))
        for x, cat in zip(x_vals, categories):
            ax.text(x, cat, f" {fmt.format(x)}", va="center", fontsize=9)

    ax.set_title(f"Dot plot of {col} by {hue_col}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(col)
    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#--- Function : plot_discrete_lollipop_bivariate ---
def plot_discrete_lollipop_bivariate(df, col, hue_col, normalize=True, figsize=(8,4), colors=None):
    """Lollipop plot for discrete variable grouped by hue_col."""
    if colors is None: colors = [UNIFORM_BLUE, PALE_PINK]
    if col not in df.columns or hue_col not in df.columns: return

    categories = sorted(df[col].dropna().unique())
    hue_values = sorted(df[hue_col].dropna().unique())
    fig, ax = plt.subplots(figsize=figsize)

    for i, val in enumerate(hue_values):
        series = df[df[hue_col]==val][col].dropna()
        counts = get_counts(series)
        if normalize:
            counts = counts / counts.sum()
            xlabel, fmt = "Proportion", "{:.2f}"
        else:
            xlabel, fmt = "Count", "{:.0f}"

        y_pos = np.arange(len(categories)) + (i*0.2)
        x_vals = [counts.get(cat,0) for cat in categories]

        ax.hlines(y=y_pos, xmin=0, xmax=x_vals, color=colors[i % len(colors)], linewidth=2)
        ax.plot(x_vals, y_pos, 'o', color=colors[i % len(colors)], label=str(val))
        for x, y in zip(x_vals, y_pos):
            ax.text(x, y, f" {fmt.format(x)}", va="center", fontsize=9)

    ax.set_yticks(np.arange(len(categories)) + 0.1)
    ax.set_yticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(col)
    ax.set_title(f"Lollipop plot of {col} by {hue_col}")
    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#--- Function : plot_stacked_grid ---
def plot_stacked_grid(df, dependent_var, group_vars, n_rows=2, n_cols=2):
    """Stacked bar charts grid for multi-level analysis."""
    if isinstance(group_vars, str): group_vars = [group_vars]
    main_var = group_vars[0]
    sub_vars = group_vars[1:]
    unique_main = sorted(df[main_var].dropna().unique())
    
    colors = [UNIFORM_BLUE, PALE_PINK] + sns.color_palette("Blues_d", df[dependent_var].nunique())

    plots_per_fig = n_rows*n_cols
    for i in range(0, len(unique_main), plots_per_fig):
        batch = unique_main[i:i+plots_per_fig]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols,5*n_rows))
        axes = np.array([axes]).flatten()
        
        for ax, main_val in zip(axes, batch):
            df_subset = df[df[main_var]==main_val]
            counts = df_subset.groupby(sub_vars + [dependent_var]).size().unstack(fill_value=0)
            bottoms = np.zeros(len(counts))
            for j, val in enumerate(counts.columns):
                ax.bar(counts.index, counts[val], bottom=bottoms, 
                       color=colors[j % len(colors)], label=str(val))
                for xi, c, bottom in zip(range(len(counts)), counts[val].values, bottoms):
                    if c > 0: 
                        ax.text(xi, bottom + c/2, str(int(c)), ha="center", 
                                va="center", color="black", fontsize=9)
                bottoms += counts[val].values
            ax.set_title(f'{main_var}: {main_val}')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
        for j in range(len(batch), len(axes)): 
            fig.delaxes(axes[j])
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title=dependent_var, bbox_to_anchor=(1.01, 0.9), loc='upper left')
        plt.tight_layout()
        plt.show()

#--- Function: plot_faceted_countplot ---
def plot_faceted_countplot(df, x_col, hue_col, facet_col, 
                           n_cols=3, figsize_factor=(4, 4)):
    """
    Plots a faceted countplot using Seaborn catplot.
    """
    from .style import GENDER_PALETTE, BIVARIATE_PALETTE, UNIFORM_BLUE

    # Define palette and hue order
    if hue_col == "Gender":
        current_palette = GENDER_PALETTE
        h_order = ["Male", "Female"]
    elif hue_col:
        current_palette = BIVARIATE_PALETTE
        h_order = sorted(df[hue_col].dropna().unique().tolist())
    else:
        current_palette = [UNIFORM_BLUE]
        h_order = None

    g = sns.catplot(
        data=df,
        kind="count",
        x=x_col,
        hue=hue_col,
        col=facet_col,
        col_wrap=n_cols,
        order=sorted(df[x_col].unique().tolist()),
        hue_order=h_order,
        palette=current_palette,
        height=figsize_factor[1],
        aspect=figsize_factor[0]/figsize_factor[1],
        sharex=False,
        edgecolor="black",
        linewidth=0.5
    )

    g.set_axis_labels(x_col, "Count")
    g.set_titles(f"{facet_col}: {{col_name}}", fontweight='bold')
    
    g.fig.subplots_adjust(top=0.88, hspace=0.5, wspace=0.2)
    g.fig.suptitle(f'Faceted Analysis: {x_col} by {hue_col}', 
                   fontsize=14, fontweight='bold')

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelbottom=True, rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    plt.show()
