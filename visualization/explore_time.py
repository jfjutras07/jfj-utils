import matplotlib.pyplot as plt
import pandas as pd
from .style import UNIFORM_BLUE, PALE_PINK

#--- Function : plot_line_grid_over_time ---
def plot_line_grid_over_time(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    facet_cols: list,
    group_col: str = None,
    group_labels: dict = None,
    agg_func='mean',
    n_cols: int = 2,
    figsize=(14,10),
    xlabel=None,
    ylabel=None,
    title=None,
    colors=None
):
    """
    Generic grid of line plots for a numeric value over time,
    faceted by multiple categorical or ordinal variables.
    
    Parameters:
    - df: pandas DataFrame
    - time_col: column representing time (e.g., G1, G2, G3)
    - value_col: numeric column to aggregate and plot
    - facet_cols: list of variables to facet into subplots
    - group_col: optional column for line grouping (e.g., subject)
    - group_labels: optional dict to map group values to readable labels
    - agg_func: aggregation function ('mean', 'median', etc.)
    - n_cols: number of columns in the subplot grid
    - figsize: figure size
    - xlabel, ylabel, title: labels and title
    - colors: optional list of colors for groups
    """
    
    xlabel = xlabel or time_col
    ylabel = ylabel or value_col
    
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]
    
    n_rows = int(np.ceil(len(facet_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    axes = axes.flatten()
    
    for ax, facet in zip(axes, facet_cols):
        
        if group_col:
            grouped = (
                df
                .groupby([facet, time_col, group_col])[value_col]
                .agg(agg_func)
                .reset_index()
            )
        else:
            grouped = (
                df
                .groupby([facet, time_col])[value_col]
                .agg(agg_func)
                .reset_index()
            )
        
        for i, level in enumerate(sorted(grouped[facet].dropna().unique())):
            subset = grouped[grouped[facet] == level]
            
            if group_col:
                for j, grp in enumerate(subset[group_col].unique()):
                    grp_label = group_labels.get(grp, grp) if group_labels else grp
                    data = subset[subset[group_col] == grp]
                    
                    ax.plot(
                        data[time_col],
                        data[value_col],
                        marker='o',
                        label=f'{grp_label}',
                        color=colors[j % len(colors)]
                    )
            else:
                ax.plot(
                    subset[time_col],
                    subset[value_col],
                    marker='o',
                    label=str(level),
                    color=colors[i % len(colors)]
                )
        
        ax.set_title(f'{facet} vs {value_col}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    
    for ax in axes[len(facet_cols):]:
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    if group_col:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()

#--- Function : plot_line_over_time ---
def plot_line_over_time(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    group_col: str = None,
    group_labels: dict = None,
    agg_func='mean',
    figsize=(10,6),
    xlabel=None,
    ylabel=None,
    title=None,
    colors=None
):
    """
    Generic line plot for a numeric value over any 'time' column.
    
    Parameters:
    - df: pandas DataFrame
    - time_col: column representing time (year, month, etc.)
    - value_col: numeric column to plot
    - group_col: optional column to group by
    - group_labels: optional dict to map group values to readable labels
    - agg_func: aggregation function ('mean', 'median', etc.)
    - figsize: figure size
    - xlabel, ylabel, title: labels and title
    - colors: optional list of colors for each group
    """
    
    xlabel = xlabel or time_col
    ylabel = ylabel or value_col
    title = title or f'{value_col} over {time_col}'
    
    # default colors: uniform blue and pale pink
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]
    
    if group_col:
        grouped = df.groupby([time_col, group_col])[value_col].agg(agg_func).reset_index()
        
        if group_labels:
            grouped['Group'] = grouped[group_col].map(group_labels)
        else:
            grouped['Group'] = grouped[group_col]
        
        plt.figure(figsize=figsize)
        unique_groups = grouped['Group'].unique()
        for i, grp in enumerate(unique_groups):
            subset = grouped[grouped['Group'] == grp]
            plt.plot(subset[time_col], subset[value_col], marker='o',
                     label=grp, color=colors[i % len(colors)])
    else:
        grouped = df.groupby(time_col)[value_col].agg(agg_func).reset_index()
        plt.figure(figsize=figsize)
        plt.plot(grouped[time_col], grouped[value_col], marker='o', color=UNIFORM_BLUE)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if group_col:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
