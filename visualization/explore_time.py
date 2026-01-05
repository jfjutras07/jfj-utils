import matplotlib.pyplot as plt
import pandas as pd
from .style import UNIFORM_BLUE, PALE_PINK

#--- Function : plot_line_grid_over_time ---
def plot_line_grid_over_time(
    df: pd.DataFrame,
    value_cols: list,
    group_col: str = None,
    facet_col: str = None,
    time_col: str = 'Time',
    value_col_name: str = 'MeanValue',
    agg_func='mean',
    figsize=(13, 4),
    xlabel=None,
    ylabel=None,
    title=None,
    colors=None
):
    """
    Generic line plots in a grid for multiple columns over time, optionally grouped.

    Parameters:
    - df: pandas DataFrame
    - value_cols: list of numeric columns to track over time
    - group_col: column to group by (optional)
    - facet_col: column defining subplots (optional)
    - time_col: name of the time column to create from value_cols
    - value_col_name: name of the numeric value column created after melt
    - agg_func: aggregation function ('mean', 'median', etc.)
    - figsize: figure size
    - xlabel, ylabel, title: labels and title
    - colors: optional list of colors for each group
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Melt automatically if multiple columns
    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name=time_col,
        value_name=value_col_name
    )

    xlabel = xlabel or time_col
    ylabel = ylabel or value_col_name
    title = title or f'{value_col_name} over {time_col}'

    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK, 'green', 'orange', 'purple', 'brown']

    if facet_col:
        facets = df_long[facet_col].unique()
        n_facets = len(facets)

        fig, axes = plt.subplots(1, n_facets, figsize=figsize, sharey=True)
        if n_facets == 1:
            axes = [axes]

        for ax, facet in zip(axes, facets):
            facet_data = df_long[df_long[facet_col] == facet]
            if group_col:
                grouped = facet_data.groupby([time_col, group_col])[value_col_name].agg(agg_func).reset_index()
                for i, level in enumerate(sorted(grouped[group_col].dropna().unique())):
                    plot_data = grouped[grouped[group_col] == level]
                    ax.plot(
                        plot_data[time_col],
                        plot_data[value_col_name],
                        marker='o', linewidth=2,
                        label=str(level), color=colors[i % len(colors)]
                    )
            else:
                grouped = facet_data.groupby(time_col)[value_col_name].agg(agg_func).reset_index()
                ax.plot(grouped[time_col], grouped[value_col_name], marker='o', linewidth=2, color=UNIFORM_BLUE)
            ax.set_title(facet)
            ax.set_xlabel(xlabel)
            ax.grid(True)

        axes[0].set_ylabel(ylabel)
        if group_col:
            axes[-1].legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.suptitle(title, fontsize=13)
    else:
        plt.figure(figsize=figsize)
        if group_col:
            grouped = df_long.groupby([time_col, group_col])[value_col_name].agg(agg_func).reset_index()
            for i, level in enumerate(sorted(grouped[group_col].dropna().unique())):
                plot_data = grouped[grouped[group_col] == level]
                plt.plot(
                    plot_data[time_col],
                    plot_data[value_col_name],
                    marker='o', linewidth=2,
                    label=str(level), color=colors[i % len(colors)]
                )
            plt.legend(title=group_col)
        else:
            grouped = df_long.groupby(time_col)[value_col_name].agg(agg_func).reset_index()
            plt.plot(grouped[time_col], grouped[value_col_name], marker='o', linewidth=2, color=UNIFORM_BLUE)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)

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
