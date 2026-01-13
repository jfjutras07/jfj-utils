import matplotlib.pyplot as plt
import pandas as pd
from .style import UNIFORM_BLUE, PALE_PINK
import math
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#--- Function : plot_line_grid_over_time ---
def plot_line_grid_over_time(df, value_cols, group_cols=None, facet_col=None,
                             time_col='Time', value_col_name='MeanValue',
                             agg_func='mean', figsize=(13,4), xlabel=None, ylabel=None, title=None,
                             colors=None):
    """
    Generic line plots in a grid for multiple numeric columns over time, optionally grouped.
    Automatically loops over multiple categorical variables if provided.

    Parameters:
    - df: pandas DataFrame
    - value_cols: list of numeric columns to track over time
    - group_cols: single column or list of categorical columns to group by
    - facet_col: column defining subplots (optional)
    - time_col: name of the time column created from value_cols
    - value_col_name: name of the numeric value column after melt
    - agg_func: aggregation function ('mean', 'median', etc.)
    - figsize: figure size
    - xlabel, ylabel, title: labels and title
    - colors: optional list of colors for groups
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    #Ensure group_cols is a list
    if group_cols is not None and not isinstance(group_cols, list):
        group_cols = [group_cols]

    #Melt numeric columns once
    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name=time_col,
        value_name=value_col_name
    )

    xlabel = xlabel or time_col
    ylabel = ylabel or value_col_name

    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK, 'green', 'orange', 'purple', 'brown']

    #Loop over all categorical columns
    if group_cols:
        for var in group_cols:
            t_title = title or f'{value_col_name} over {time_col} by {var}'
            if facet_col:
                facets = df_long[facet_col].unique()
                n_facets = len(facets)

                fig, axes = plt.subplots(1, n_facets, figsize=figsize, sharey=True)
                if n_facets == 1:
                    axes = [axes]

                for ax, facet in zip(axes, facets):
                    facet_data = df_long[df_long[facet_col] == facet]
                    grouped = facet_data.groupby([time_col, var])[value_col_name].agg(agg_func).reset_index()
                    for i, level in enumerate(sorted(grouped[var].dropna().unique())):
                        plot_data = grouped[grouped[var] == level]
                        ax.plot(plot_data[time_col], plot_data[value_col_name], marker='o', linewidth=2,
                                label=str(level), color=colors[i % len(colors)])
                    ax.set_title(facet)
                    ax.set_xlabel(xlabel)
                    ax.grid(True)

                axes[0].set_ylabel(ylabel)
                axes[-1].legend(title=var, bbox_to_anchor=(1.05,1), loc='upper left')
                fig.suptitle(t_title, fontsize=13)
                plt.tight_layout()
                plt.show()
            else:
                grouped = df_long.groupby([time_col, var])[value_col_name].agg(agg_func).reset_index()
                plt.figure(figsize=figsize)
                for i, level in enumerate(sorted(grouped[var].dropna().unique())):
                    plot_data = grouped[grouped[var] == level]
                    plt.plot(plot_data[time_col], plot_data[value_col_name], marker='o', linewidth=2,
                             label=str(level), color=colors[i % len(colors)])
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(t_title)
                plt.legend(title=var)
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    else:
        #No categorical variables, simple plot
        if facet_col:
            facets = df_long[facet_col].unique()
            n_facets = len(facets)

            fig, axes = plt.subplots(1, n_facets, figsize=figsize, sharey=True)
            if n_facets == 1:
                axes = [axes]

            for ax, facet in zip(axes, facets):
                facet_data = df_long[df_long[facet_col] == facet]
                grouped = facet_data.groupby(time_col)[value_col_name].agg(agg_func).reset_index()
                ax.plot(grouped[time_col], grouped[value_col_name], marker='o', linewidth=2, color=UNIFORM_BLUE)
                ax.set_title(facet)
                ax.set_xlabel(xlabel)
                ax.grid(True)

            axes[0].set_ylabel(ylabel)
            plt.suptitle(title or f'{value_col_name} over {time_col}', fontsize=13)
            plt.tight_layout()
            plt.show()
        else:
            grouped = df_long.groupby(time_col)[value_col_name].agg(agg_func).reset_index()
            plt.figure(figsize=figsize)
            plt.plot(grouped[time_col], grouped[value_col_name], marker='o', linewidth=2, color=UNIFORM_BLUE)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title or f'{value_col_name} over {time_col}')
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

#--- Function : plot_temporal_data ---
def plot_temporal_data(df, value_cols, time_col='Time', group_cols=None,
                       facet_col=None, agg_func='mean', rolling_window=None, 
                       show_std=False, title=None, colors=None, figsize=(6,4)):
    """
    Generic temporal exploration for multiple numeric columns, with optional grouping and faceting.
    Enhanced with rolling average and standard deviation for robust signal analysis.

    Parameters:
    - df: pandas DataFrame
    - value_cols: list of numeric columns to explore over time
    - time_col: column representing time (week, month, day, etc.)
    - group_cols: list of categorical columns to group by (optional)
    - facet_col: column defining subplots (optional)
    - agg_func: aggregation function ('mean', 'median', etc.)
    - rolling_window: int, size of the window for smoothing (optional)
    - show_std: bool, if True, shows the standard deviation as a shaded area
    - title: main title for all plots
    - colors: list of colors to cycle through groups
    - figsize: size for each subplot
    """

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore pandas FutureWarning

    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK, 'green', 'orange', 'purple', 'brown']

    if group_cols is not None and not isinstance(group_cols, list):
        group_cols = [group_cols]

    # Melt numeric columns for long format
    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name='Variable',
        value_name='Value'
    )

    # Determine facets
    facets = [None]
    if facet_col:
        facets = df_long[facet_col].dropna().unique()

    # Layout: 2 facets per row
    n_facets = len(facets)
    n_rows = math.ceil(n_facets / 2)
    n_cols = min(2, n_facets)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows), squeeze=False)

    for idx, facet in enumerate(facets):
        r = idx // 2
        c = idx % 2
        ax = axes[r, c]

        facet_data = df_long if facet is None else df_long[df_long[facet_col] == facet]

        if group_cols:
            for i, var in enumerate(group_cols):
                grouped = facet_data.groupby([time_col, var], observed=False)['Value'].agg([agg_func, 'std']).reset_index()
                
                for j, level in enumerate(sorted(grouped[var].dropna().unique())):
                    plot_data = grouped[grouped[var] == level].sort_values(time_col)
                    
                    y_values = plot_data[agg_func]
                    if rolling_window:
                        y_values = y_values.rolling(window=rolling_window, center=True).mean()
                    
                    color = colors[j % len(colors)]
                    line, = ax.plot(plot_data[time_col], y_values, marker='o',
                                    label=f'{var}: {level}', color=color, linewidth=2)
                    
                    if show_std:
                        ax.fill_between(plot_data[time_col], 
                                        plot_data[agg_func] - plot_data['std'], 
                                        plot_data[agg_func] + plot_data['std'], 
                                        color=color, alpha=0.15)
        else:
            grouped = facet_data.groupby(time_col)['Value'].agg([agg_func, 'std']).reset_index()
            grouped = grouped.sort_values(time_col)
            
            y_values = grouped[agg_func]
            if rolling_window:
                y_values = y_values.rolling(window=rolling_window, center=True).mean()
                
            ax.plot(grouped[time_col], y_values, marker='o', color=UNIFORM_BLUE, linewidth=2)
            
            if show_std:
                ax.fill_between(grouped[time_col], 
                                grouped[agg_func] - grouped['std'], 
                                grouped[agg_func] + grouped['std'], 
                                color=UNIFORM_BLUE, alpha=0.15)

        ax.set_title(facet if facet else 'All data')
        ax.set_xlabel(time_col)
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=8)

    # Hide any empty subplots
    for idx in range(n_facets, n_rows*n_cols):
        r = idx // 2
        c = idx % 2
        fig.delaxes(axes[r, c])

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
