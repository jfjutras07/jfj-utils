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

#--- Function plot_box_grid ---
def plot_box_grid(df, value_cols, group_col='Economic_status', n_rows=2, n_cols=2, palette=BIVARIATE_PALETTE, hue_col=None):
    """
    Plot a grid of boxplots for the specified value columns against one or more group columns.
    Works whether hue is specified or not. Converts single hex palette to list if needed.
    
    Parameters:
    - df: DataFrame
    - value_cols: list of columns to plot on y-axis
    - group_col: column(s) to group on x-axis
    - n_rows, n_cols: grid layout
    - palette: hex string, list, dict, or seaborn palette name
    - hue_col: optional hue column
    """
    if isinstance(group_col, str):
        group_col = [group_col]
    
    y_col = value_cols[0]
    plots_per_fig = n_rows * n_cols

    #Convert single hex to list for Seaborn
    if isinstance(palette, str) and not hue_col:
        palette = [palette]

    for i in range(0, len(group_col), plots_per_fig):
        batch = group_col[i:i+plots_per_fig]
        if len(batch) == 1:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(data=df, x=batch[0], y=y_col, hue=hue_col, palette=palette, ax=ax)
            ax.set_title(f'{y_col} by {batch[0]}')
            ax.set_xlabel(batch[0])
            ax.set_ylabel(y_col)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols,5*n_rows), sharey=True)
            axes = axes.flatten()
            for ax, grp in zip(axes, batch):
                sns.boxplot(data=df, x=grp, y=y_col, hue=hue_col, palette=palette, ax=ax)
                ax.set_title(f'{y_col} by {grp}')
                ax.set_xlabel(grp)
                ax.set_ylabel(y_col)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            for j in range(len(batch), len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.show()

#--- Function plot_box_plot ---
def plot_box_plot(df, value_cols, category_col, hue_col=None,
                  palette=BIVARIATE_PALETTE, figsize=(16, 6)):
    """
    Boxplot for a single numeric column versus a categorical column, optionally split by hue.
    
    Parameters:
    - df: DataFrame
    - value_cols: list of numeric columns (only the first one is used)
    - category_col: column for x-axis
    - hue_col: optional hue column
    - palette: seaborn palette name, list, or dict mapping hue values
    - figsize: figure size tuple
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    y_col = value_cols[0]
    
    # Default palette
    if palette is None:
        if hue_col:
            # Create a simple palette for two categories
            unique_hue = df[hue_col].unique()
            default_colors = ["#1f77b4", "#ff69b4"]  # blue and pink
            palette = {k: default_colors[i % len(default_colors)] for i, k in enumerate(unique_hue)}
        else:
            palette = "#1f77b4"  # single color
    
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=category_col, y=y_col, hue=hue_col, palette=palette, dodge=True)
    plt.title(f'{y_col} by {category_col}' + (f' and {hue_col}' if hue_col else ''))
    plt.xlabel(category_col)
    plt.ylabel(y_col)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    if hue_col:
        plt.legend(title=hue_col)
    plt.tight_layout()
    plt.show()

#---Function: plot_correlation_heatmap---
def plot_correlation_heatmap(df, numeric_cols=None, method='spearman', figsize=(12,8),
                             cmap=SEQUENTIAL_CMAP, annot=True, fmt=".2f"):
    """
    Plots a heatmap of correlations for selected numeric columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    numeric_cols : list, optional
        Columns to include. Defaults to all numeric columns.
    method : str, optional
        Correlation method: 'pearson', 'spearman', or 'kendall'. Default is 'spearman'.
    figsize : tuple, optional
        Size of the figure. Default is (12,8).
    cmap : str or Colormap, optional
        Color map for heatmap. Default is SEQUENTIAL_CMAP.
    annot : bool, optional
        Whether to annotate cells with correlation values. Default is True.
    fmt : str, optional
        String format for annotation. Default is ".2f".
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    corr_matrix = df[numeric_cols].corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, fmt=fmt, cmap=cmap)
    plt.title(f"{method.capitalize()} Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    plt.close()

#---Function: plot_heatmap_grid---
def plot_heatmap_grid(df, value_col, index_col, columns_col=None, aggfunc='median',
                      cmap=SEQUENTIAL_CMAP, fmt=".0f", figsize=(10,6)):
    """
    Plots a heatmap of aggregated values for one or two grouping variables.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    value_col : str
        Column to aggregate.
    index_col : str
        Column to use for the rows of the heatmap.
    columns_col : str, optional
        Column to use for the columns of the heatmap. Default is None.
    aggfunc : str or function, optional
        Aggregation function to apply. Examples: 'median', 'mean'. Default is 'median'.
    cmap : str or Colormap, optional
        Colormap for the heatmap. Default is SEQUENTIAL_CMAP.
    fmt : str, optional
        Format string for annotations. Default is ".0f".
    figsize : tuple, optional
        Size of the figure. Default is (10,6).
    """
    if columns_col is not None:
        pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_col, aggfunc=aggfunc)
    else:
        pivot = df.groupby(index_col)[value_col].agg(aggfunc).to_frame()

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5)
    plt.title(f"{aggfunc.capitalize()} {value_col} by {index_col}" + (f" and {columns_col}" if columns_col else ""))
    plt.ylabel(index_col)
    plt.xlabel(columns_col if columns_col else value_col)
    plt.tight_layout()
    plt.show()
                       
#---Function: plot_numeric_bivariate---
def plot_numeric_bivariate(df, numeric_cols, hue='Gender', bins=40):
    """
    Plots distributions and boxplots for numeric columns split by a binary group.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    numeric_cols : list
        List of numeric columns to plot.
    hue : str, optional
        Binary grouping variable. Default is 'Gender'.
    bins : int, optional
        Number of bins for histograms. Default is 40.
    """
    groups = df[hue].unique()
    if len(groups) != 2:
        print("Error: exactly 2 groups required.")
        return

    palette = {groups[0]: UNIFORM_BLUE, groups[1]: PALE_PINK}

    for col in numeric_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        #Histogram for first group
        sns.histplot(df[df[hue] == groups[0]], x=col, kde=True, bins=bins,
                     color=palette[groups[0]], alpha=0.6, ax=axes[0])
        axes[0].set_title(f"{col} distribution - {groups[0]}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")

        #Histogram for second group
        sns.histplot(df[df[hue] == groups[1]], x=col, kde=True, bins=bins,
                     color=palette[groups[1]], alpha=0.6, ax=axes[1])
        axes[1].set_title(f"{col} distribution - {groups[1]}")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Count")

        #Boxplot comparing both groups
        sns.boxplot(x=hue, y=col, data=df, palette=palette, ax=axes[2])
        axes[2].set_title(f"{col} by {hue}")
        axes[2].set_xlabel(hue)
        axes[2].set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()

#---Function: plot_numeric_distribution---
def plot_numeric_distribution(df, numeric_cols, bins=40):
    """
    Plots histogram and boxplot for each numeric column to visualize distribution and outliers.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    numeric_cols : list
        List of numeric columns to plot.
    bins : int, optional
        Number of bins for histograms. Default is 40.
    """
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        #Histogram
        sns.histplot(df[col].dropna(), kde=True, bins=bins, color=UNIFORM_BLUE, ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")

        #Boxplot
        sns.boxplot(x=df[col].dropna(), color=UNIFORM_BLUE, ax=axes[1])
        axes[1].set_title(f"Outliers in {col}")
        axes[1].set_xlabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()

#---Function: plot_pairplot---
def plot_pairplot(df, features, hue=None, diag_kind='kde', corner=True, alpha=0.5, figsize=(10,10)):
    """
    Plots pairwise relationships between selected features using seaborn's pairplot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    features : list
        List of columns to include in the pairplot.
    hue : str, optional
        Column name for color grouping. Default is None.
    diag_kind : str, optional
        Kind of plot on the diagonal: 'kde' or 'hist'. Default is 'kde'.
    corner : bool, optional
        If True, shows only the lower triangle of plots. Default is True.
    alpha : float, optional
        Transparency of scatter points. Default is 0.5.
    figsize : tuple, optional
        Figure size. Default is (10,10).
    """
    sns.set(style="ticks")
    pairplot = sns.pairplot(df[features], hue=hue, diag_kind=diag_kind, corner=corner, plot_kws={'alpha': alpha})
    pairplot.fig.set_size_inches(figsize)
    pairplot.fig.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()

#---Function: plot_scatter_grid---
def plot_scatter_grid(df: pd.DataFrame, x_cols: list, y_cols: list, group_col: str = None,
                      group_labels: dict = None, n_cols_per_row: int = 2, figsize=(14,6)):
    """
    Plots scatter plots of x vs y columns in a grid, optionally grouped by a categorical variable.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    x_cols : list
        List of column names to use as x-axis variables.
    y_cols : list
        List of column names to use as y-axis variables.
    group_col : str, optional
        Column name for grouping and coloring points. Default is None.
    group_labels : dict, optional
        Dictionary to map group values to labels. Default is None.
    n_cols_per_row : int, optional
        Number of columns per row in the grid. Default is 2.
    figsize : tuple, optional
        Figure size (width, height). Default is (14,6).
    """
    num_plots = len(x_cols)
    if num_plots == 1:
        fig, ax = plt.subplots(figsize=(10,6))
        x, y = x_cols[0], y_cols[0]
        if group_col:
            plot_data = df.copy()
            plot_data['Group'] = plot_data[group_col].map(group_labels) if group_labels else plot_data[group_col]
            sns.scatterplot(
                data=plot_data, x=x, y=y, hue='Group',
                palette={k: UNIFORM_BLUE for k in plot_data['Group'].unique()},
                s=60
            )
        else:
            sns.scatterplot(data=df, x=x, y=y, color=UNIFORM_BLUE, s=60)
        ax.set_title(f'{y} vs {x}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        n_rows = (num_plots + n_cols_per_row - 1) // n_cols_per_row
        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(figsize[0], n_rows*figsize[1]))
        axes = axes.flatten()
        for i, (x, y) in enumerate(zip(x_cols, y_cols)):
            ax = axes[i]
            if group_col:
                plot_data = df.copy()
                plot_data['Group'] = plot_data[group_col].map(group_labels) if group_labels else plot_data[group_col]
                sns.scatterplot(
                    data=plot_data, x=x, y=y, hue='Group',
                    palette={k: UNIFORM_BLUE for k in plot_data['Group'].unique()},
                    s=40, ax=ax
                )
            else:
                sns.scatterplot(data=df, x=x, y=y, color=UNIFORM_BLUE, s=40, ax=ax)
            ax.set_title(f'{y} vs {x}')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.grid(True)
        for j in range(len(x_cols), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

#---Function: plot_scatter_plot---
def plot_scatter_plot(df, target_col, numeric_cols=None):
    """
    Plots scatter plots of numeric columns against a target column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    target_col : str
        Column to plot on the y-axis.
    numeric_cols : list, optional
        List of numeric columns to plot on the x-axis. 
        If None, all numeric columns except target_col are used.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]

    n_cols = 2
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].scatter(df[col], df[target_col], alpha=0.6, color=UNIFORM_BLUE)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f"{col} vs {target_col}")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

#---Function: plot_swarm_grid---
def plot_swarm_grid(df, value_cols, group_col='Economic_status', hue_col=None,
                    n_rows=2, n_cols=2, color=UNIFORM_BLUE, hue_palette=None,
                    dodge=True, figsize_single=(10,6), figsize_grid=(6,5)):
    """
    Plots swarm plots for one or multiple grouping variables, optionally with hue.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    value_cols : list
        List of numeric columns to plot (only the first is used for y-axis).
    group_col : str or list
        Column(s) to group by on the x-axis. Default is 'Economic_status'.
    hue_col : str, optional
        Column for color grouping. Default is None.
    n_rows : int, optional
        Number of rows per figure grid. Default is 2.
    n_cols : int, optional
        Number of columns per figure grid. Default is 2.
    color : str, optional
        Color for points if no hue. Default is UNIFORM_BLUE.
    hue_palette : dict or list, optional
        Palette for hue groups. Default is None.
    dodge : bool, optional
        Whether to separate points by hue. Default is True.
    figsize_single : tuple, optional
        Figure size for a single plot. Default is (10,6).
    figsize_grid : tuple, optional
        Base figure size for multiple plots. Default is (6,5).
    """
    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = group_col

    y_col = value_cols[0]
    plots_per_fig = n_rows * n_cols

    for i in range(0, len(group_cols), plots_per_fig):
        batch = group_cols[i:i + plots_per_fig]

        #Figure size
        fig_size = figsize_single if len(batch) == 1 else (figsize_grid[0]*n_cols, figsize_grid[1]*n_rows)
        fig, axes = plt.subplots(
            nrows=1 if len(batch) == 1 else n_rows,
            ncols=1 if len(batch) == 1 else n_cols,
            figsize=fig_size,
            sharey=True
        )

        #Ensure axes iterable
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, grp in zip(axes, batch):
            sns.swarmplot(
                data=df,
                x=grp,
                y=y_col,
                hue=hue_col,
                dodge=dodge if hue_col else False,
                palette=hue_palette if hue_col else None,
                color=color if hue_col is None else None,
                ax=ax
            )
            ax.set_title(f'{y_col} by {grp}')
            ax.set_xlabel(grp)
            ax.set_ylabel(y_col)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        #Hide unused axes
        for j in range(len(batch), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

#---Function: plot_violin_grid---
def plot_violin_grid(
    df,
    value_cols,
    group_col='Economic_status',
    n_rows=2,
    n_cols=2,
    palette="#1f77b4"
):
    """
    Plots violin plots for one or multiple grouping variables.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    value_cols : list
        List of numeric columns to plot (only the first is used for y-axis).
    group_col : str or list
        Categorical column(s) used for grouping on the x-axis.
    n_rows : int
        Number of rows per figure grid.
    n_cols : int
        Number of columns per figure grid.
    palette : str, list, or dict
        Color or palette for violins.
    """

    # --- Input normalization ---
    y_col = value_cols[0]

    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = group_col

    #Normalize palette
    if isinstance(palette, str):
        palette = [palette]

    plots_per_fig = n_rows * n_cols

    # --- Plotting ---
    for i in range(0, len(group_cols), plots_per_fig):
        batch = group_cols[i:i + plots_per_fig]

        fig_size = (10, 6) if len(batch) == 1 else (6 * n_cols, 5 * n_rows)
        fig, axes = plt.subplots(
            nrows=1 if len(batch) == 1 else n_rows,
            ncols=1 if len(batch) == 1 else n_cols,
            figsize=fig_size,
            sharey=True
        )

        #Ensure axes is iterable
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, grp in zip(axes, batch):
            sns.violinplot(
                data=df,
                x=grp,
                y=y_col,
                palette=palette,
                inner='quartile',
                ax=ax
            )
            ax.set_title(f'{y_col} by {grp}')
            ax.set_xlabel(grp)
            ax.set_ylabel(y_col)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        #Hide unused axes
        for j in range(len(batch), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()
