import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from .style import UNIFORM_BLUE, PALE_PINK, BIVARIATE_PALETTE
from visualization.style import SEQUENTIAL_CMAP

#--- Function: plot_box_grid ---
def plot_box_grid(df, value_cols, group_col='Economic_status', n_rows=2, n_cols=2, hue_col=None, figsize=None):
    """
    Plot a grid of boxplots. 
    Correctly handles multiple colors when hue_col is provided.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    
    x_axis = group_col[0] if isinstance(group_col, list) else group_col
    n_plots = len(value_cols)
    plots_per_fig = n_rows * n_cols

    for i in range(0, n_plots, plots_per_fig):
        batch = value_cols[i : i + plots_per_fig]
        
        current_rows = n_rows if len(batch) > 1 else 1
        current_cols = n_cols if len(batch) > 1 else 1
        
        if figsize is None:
            current_figsize = (16, 6) if len(batch) == 1 else (6 * n_cols, 5 * n_rows)
        else:
            current_figsize = figsize

        fig, axes = plt.subplots(current_rows, current_cols, figsize=current_figsize)
        axes_flat = np.atleast_1d(axes).flatten()

        for idx, y_col in enumerate(batch):
            ax = axes_flat[idx]
            
            # If hue_col is present, Seaborn uses BIVARIATE_PALETTE to map colors to categories.
            # If no hue, it falls back to a single UNIFORM_BLUE color.
            sns.boxplot(
                data=df, 
                x=x_axis, 
                y=y_col, 
                hue=hue_col, 
                palette=BIVARIATE_PALETTE if hue_col else None,
                color=UNIFORM_BLUE if hue_col is None else None,
                ax=ax,
                linewidth=1.5
            )
            
            ax.set_title(f'{y_col} by {x_axis}', fontweight='bold')
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_col)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')

        # Hide unused subplots in the grid
        for j in range(len(batch), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()
        
#--- Function: plot_correlation_heatmap ---
def plot_correlation_heatmap(df, numeric_cols=None, method='spearman', figsize=(12,8),
                             cmap=SEQUENTIAL_CMAP, annot=True, fmt=".2f"):
    """
    Plots a heatmap of correlations for selected numeric columns.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
    corr_matrix = df[numeric_cols].corr(method=method)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, fmt=fmt, cmap=cmap, linewidths=0.5, linecolor='white')
    plt.title(f"{method.capitalize()} Correlation Heatmap", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    plt.close()

#---Function: plot_heatmap_grid---
def plot_heatmap_grid(df, value_col, index_col, columns_col=None, aggfunc='median',
                      cmap=SEQUENTIAL_CMAP, fmt=".0f", figsize=(10,6)):
    """
    Plots a heatmap of aggregated values for one or two grouping variables.
    """
    if columns_col is not None:
        pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_col, aggfunc=aggfunc)
    else:
        pivot = df.groupby(index_col)[value_col].agg(aggfunc).to_frame()

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap, 
        linewidths=0.5, 
        linecolor='white'
    )
    
    title = f"{aggfunc.capitalize()} {value_col} by {index_col}" + (f" and {columns_col}" if columns_col else "")
    plt.title(title, fontweight='bold', pad=15)
    plt.ylabel(index_col)
    plt.xlabel(columns_col if columns_col else value_col)
    
    plt.tight_layout()
    plt.show()

#--- Function: plot_mi_vs_correlation ---
def plot_mi_vs_correlation(X, y, target_type='classification', method='spearman', figsize=(10, 12)):
    """
    Diagnostic plot comparing Mutual Information and Absolute Correlation scores.
    """
    # Compute MI Scores using your specific functions
    if target_type == 'classification':
        mi_scores = mi_classification(X, y)
    else:
        mi_scores = mi_regression(X, y)

    # Compute Correlation with target
    y_encoded = y.copy()
    if not pd.api.types.is_numeric_dtype(y_encoded):
        y_encoded, _ = y_encoded.factorize()

    correlations = {}
    for col in X.columns:
        x_col = X[col].copy()
        if not pd.api.types.is_numeric_dtype(x_col):
            x_col, _ = x_col.factorize()
        correlations[col] = abs(x_col.corr(y_encoded, method=method))

    corr_series = pd.Series(correlations).reindex(mi_scores.index)

    plot_df = pd.DataFrame({
        'Mutual Information': mi_scores,
        'Absolute Correlation': corr_series
    }).sort_values('Mutual Information', ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(plot_df))
    bar_height = 0.4

    ax.barh(y_pos + bar_height/2, plot_df['Mutual Information'], height=bar_height, 
            label='Mutual Information', color=UNIFORM_BLUE, edgecolor="black")

    ax.barh(y_pos - bar_height/2, plot_df['Absolute Correlation'], height=bar_height, 
            label=f'Abs {method.capitalize()} Corr', color=PALE_PINK, edgecolor="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel("Relationship Strength")
    ax.set_title(f"Diagnostic: MI vs {method.capitalize()} Correlation", fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    sns.despine()
    plt.tight_layout()
    plt.show()
    plt.close()

#---Function: plot_numeric_bivariate---
def plot_numeric_bivariate(df, numeric_cols, hue='Gender', bins=40):
    """
    Plots distributions and boxplots for numeric columns split by a binary group.
    """
    groups = df[hue].unique()
    if len(groups) != 2:
        print("Error: exactly 2 groups required.")
        return

    palette = {groups[0]: UNIFORM_BLUE, groups[1]: PALE_PINK}

    for col in numeric_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # Histogram Group 1
        sns.histplot(df[df[hue] == groups[0]], x=col, kde=True, bins=bins,
                     color=palette[groups[0]], alpha=0.6, ax=axes[0])
        axes[0].set_title(f"{col} - {groups[0]}", fontweight='bold')
        axes[0].set_ylabel("Count")

        # Histogram Group 2
        sns.histplot(df[df[hue] == groups[1]], x=col, kde=True, bins=bins,
                     color=palette[groups[1]], alpha=0.6, ax=axes[1])
        axes[1].set_title(f"{col} - {groups[1]}", fontweight='bold')
        axes[1].set_ylabel("Count")

        # Boxplot Comparison
        sns.boxplot(x=hue, y=col, data=df, palette=palette, ax=axes[2], linewidth=1.5)
        axes[2].set_title(f"{col} by {hue}", fontweight='bold')

        plt.tight_layout()
        plt.show()
        plt.close()

#---Function: plot_numeric_distribution---
def plot_numeric_distribution(df, numeric_cols, bins=40):
    """
    Plots histogram and boxplot for each numeric column.
    """
    for col in numeric_cols:
        if col not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(df[col].dropna(), kde=True, bins=bins, color=UNIFORM_BLUE, ax=axes[0], alpha=0.7)
        axes[0].set_title(f"Distribution: {col}", fontweight='bold')
        axes[0].set_ylabel("Count")

        sns.boxplot(x=df[col].dropna(), color=UNIFORM_BLUE, ax=axes[1], linewidth=1.5)
        axes[1].set_title(f"Outliers: {col}", fontweight='bold')

        plt.tight_layout()
        plt.show()
        plt.close()

#---Function: plot_pairplot---
def plot_pairplot(df, features, hue=None, diag_kind='kde', corner=True, alpha=0.5, figsize=(10,10)):
    """
    Plots pairwise relationships between selected features using seaborn's pairplot.
    """
    sns.set(style="ticks")
    
    # Ensure hue is defined in columns to avoid KeyError
    plot_cols = features + [hue] if hue is not None and hue not in features else features

    # FIX: Use BIVARIATE_PALETTE if hue is present to get blue/pink colors
    pairplot = sns.pairplot(
        df[plot_cols], 
        hue=hue, 
        diag_kind=diag_kind, 
        corner=corner, 
        palette=BIVARIATE_PALETTE if hue else None,
        plot_kws={'alpha': alpha, 's': 30, 'edgecolor': 'white', 'linewidth': 0.5}
    )
    pairplot.fig.set_size_inches(figsize)
    pairplot.fig.suptitle("Pairplot of Selected Features", fontweight='bold', y=1.02)
    plt.show()

#---Function: plot_scatter_grid---
def plot_scatter_grid(df: pd.DataFrame, x_cols: list, y_cols: list, group_col: str = None,
                      group_labels: dict = None, n_cols_per_row: int = 2, figsize=(14,6)):
    """
    Plots scatter plots of x vs y columns in a grid, optionally grouped by a categorical variable.
    """
    num_plots = len(x_cols)
    
    # Grid logic
    n_rows = (num_plots + n_cols_per_row - 1) // n_cols_per_row
    fig, axes = plt.subplots(
        n_rows, 
        n_cols_per_row if num_plots > 1 else 1, 
        figsize=(figsize[0], n_rows * figsize[1]) if num_plots > 1 else (10, 6)
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (x, y) in enumerate(zip(x_cols, y_cols)):
        ax = axes_flat[i]
        
        # Prepare data with group labels if provided
        plot_data = df.copy()
        if group_col and group_labels:
            plot_data[group_col] = plot_data[group_col].map(group_labels)

        # FIX: Use BIVARIATE_PALETTE for hue or UNIFORM_BLUE for single color
        sns.scatterplot(
            data=plot_data, 
            x=x, 
            y=y, 
            hue=group_col,
            palette=BIVARIATE_PALETTE if group_col else None,
            color=UNIFORM_BLUE if group_col is None else None,
            s=60, 
            ax=ax,
            alpha=0.7
        )
        
        ax.set_title(f'{y} vs {x}', fontweight='bold')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide unused subplots
    for j in range(num_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

#---Function: plot_swarm_grid---
def plot_swarm_grid(df, value_cols, group_col='Economic_status', hue_col=None,
                    n_rows=2, n_cols=2, color=UNIFORM_BLUE, hue_palette=BIVARIATE_PALETTE,
                    dodge=True, figsize_single=(10,6), figsize_grid=(6,5)):
    """
    Plots swarm plots for one or multiple grouping variables, optionally with hue.
    Ensures semantic color mapping for binary variables like Gender.
    """
    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = group_col

    y_col = value_cols[0]
    plots_per_fig = n_rows * n_cols

    # Prepare palette
    if hue_col == "Gender":
        from visualization.style import GENDER_PALETTE
        palette_to_use = GENDER_PALETTE
        hue_order = list(GENDER_PALETTE.keys())
    else:
        palette_to_use = hue_palette if hue_col else None
        hue_order = sorted(df[hue_col].dropna().unique().tolist()) if hue_col else None

    for i in range(0, len(group_cols), plots_per_fig):
        batch = group_cols[i:i + plots_per_fig]
        fig_size = figsize_single if len(batch) == 1 else (figsize_grid[0]*n_cols, figsize_grid[1]*n_rows)
        fig, axes = plt.subplots(
            nrows=1 if len(batch) == 1 else n_rows,
            ncols=1 if len(batch) == 1 else n_cols,
            figsize=fig_size,
            sharey=True
        )

        axes = np.atleast_1d(axes).flatten()

        for ax, grp in zip(axes, batch):
            group_order = sorted(df[grp].dropna().unique().tolist())

            sns.swarmplot(
                data=df,
                x=grp,
                y=y_col,
                hue=hue_col,
                order=group_order,
                hue_order=hue_order,
                dodge=dodge if hue_col else False,
                palette=palette_to_use,
                color=color if hue_col is None else None,
                ax=ax
            )
            ax.set_title(f'{y_col} by {grp}', fontweight='bold')
            ax.set_xlabel(grp)
            ax.set_ylabel(y_col)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        for j in range(len(batch), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

#---Function: plot_violin_grid---
def plot_violin_grid(df, value_cols, group_col='Economic_status', hue_col=None,
                     n_rows=2, n_cols=2, palette=BIVARIATE_PALETTE, dodge=True,
                     figsize_single=(10,6), figsize_grid=(6,5)):
    """
    Plots violin plots for one or multiple grouping variables, optionally with hue.
    Ensures semantic color mapping for binary variables like Gender.
    """
    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = group_col

    y_col = value_cols[0]
    plots_per_fig = n_rows * n_cols

    # Prepare palette
    if hue_col == "Gender":
        from visualization.style import GENDER_PALETTE
        palette_to_use = GENDER_PALETTE
        hue_order = list(GENDER_PALETTE.keys())
    else:
        palette_to_use = palette if hue_col else None
        hue_order = sorted(df[hue_col].dropna().unique().tolist()) if hue_col else None

    for i in range(0, len(group_cols), plots_per_fig):
        batch = group_cols[i:i + plots_per_fig]
        fig_size = figsize_single if len(batch) == 1 else (figsize_grid[0]*n_cols, figsize_grid[1]*n_rows)
        fig, axes = plt.subplots(
            nrows=1 if len(batch) == 1 else n_rows,
            ncols=1 if len(batch) == 1 else n_cols,
            figsize=fig_size,
            sharey=True
        )

        axes = np.atleast_1d(axes).flatten()

        for ax, grp in zip(axes, batch):
            group_order = sorted(df[grp].dropna().unique().tolist())

            sns.violinplot(
                data=df,
                x=grp,
                y=y_col,
                hue=hue_col,
                order=group_order,
                hue_order=hue_order,
                dodge=dodge if hue_col else False,
                palette=palette_to_use,
                color=UNIFORM_BLUE if hue_col is None else None,
                inner='quartile',
                ax=ax
            )
            ax.set_title(f'{y_col} by {grp}', fontweight='bold')
            ax.set_xlabel(grp)
            ax.set_ylabel(y_col)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        for j in range(len(batch), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()
