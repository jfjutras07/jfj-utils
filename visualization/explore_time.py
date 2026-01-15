import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import warnings
from .style import UNIFORM_BLUE, PALE_PINK

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#--- Function : plot_line_grid_over_time ---
def plot_line_grid_over_time(df, value_cols, time_col='Time', group_col=None, 
                             facet_col=None, agg_func='mean', figsize=(13, 4), 
                             colors=None, title=None):
    """
    Grid of line plots for numeric columns over time. 
    Supports optional grouping and faceting.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK, 'green', 'orange', 'purple', 'brown']

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name='Variable',
        value_name='Value'
    )

    facets = [None] if facet_col is None else df_long[facet_col].dropna().unique()
    n_facets = len(facets)
    fig, axes = plt.subplots(1, n_facets, figsize=figsize, sharey=True, squeeze=False)
    axes = axes.flatten()

    for i, facet in enumerate(facets):
        ax = axes[i]
        facet_data = df_long if facet is None else df_long[df_long[facet_col] == facet]
        
        group_vars = [time_col]
        if group_col:
            group_vars.append(group_col)
            
        grouped = facet_data.groupby(group_vars)['Value'].agg(agg_func).reset_index()
        
        if group_col:
            for j, level in enumerate(sorted(grouped[group_col].unique())):
                plot_data = grouped[grouped[group_col] == level]
                ax.plot(plot_data[time_col], plot_data['Value'], marker='o', 
                        linewidth=2, label=str(level), color=colors[j % len(colors)])
        else:
            ax.plot(grouped[time_col], grouped['Value'], marker='o', 
                    linewidth=2, color=UNIFORM_BLUE)

        ax.set_title(str(facet) if facet else value_cols[0])
        ax.set_xlabel(time_col)
        if i == 0:
            ax.set_ylabel(agg_func.capitalize())

    if group_col:
        axes[-1].legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    main_title = title or f"{', '.join(value_cols)} over {time_col}"
    fig.suptitle(main_title, fontsize=13)
    plt.tight_layout()
    plt.show()

#--- Function : plot_temporal_data ---
def plot_temporal_data(df, value_cols, time_col='Time', group_col=None,
                       facet_col=None, agg_func='mean', rolling_window=None, 
                       show_std=False, title=None, colors=None, figsize=(7, 5)):
    """
    Advanced temporal exploration with rolling averages and standard deviation shading.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK, 'green', 'orange', 'purple', 'brown']

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name='Variable',
        value_name='Value'
    )

    facets = [None] if facet_col is None else df_long[facet_col].dropna().unique()
    n_facets = len(facets)
    n_rows = math.ceil(n_facets / 2)
    n_cols = min(2, n_facets)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows), squeeze=False)

    for idx, facet in enumerate(facets):
        ax = axes[idx // 2, idx % 2]
        facet_data = df_long if facet is None else df_long[df_long[facet_col] == facet]

        group_vars = [time_col]
        if group_col: group_vars.append(group_col)
        
        stats = facet_data.groupby(group_vars)['Value'].agg([agg_func, 'std']).reset_index()

        if group_col:
            for j, level in enumerate(sorted(stats[group_col].unique())):
                plot_data = stats[stats[group_col] == level].sort_values(time_col)
                y_vals = plot_data[agg_func]
                
                if rolling_window:
                    y_vals = y_vals.rolling(window=rolling_window, center=True).mean()
                
                color = colors[j % len(colors)]
                ax.plot(plot_data[time_col], y_vals, marker='o', linewidth=2, 
                        label=f"{level}", color=color)
                
                if show_std:
                    ax.fill_between(plot_data[time_col], 
                                    plot_data[agg_func] - plot_data['std'],
                                    plot_data[agg_func] + plot_data['std'], 
                                    color=color, alpha=0.15)
        else:
            stats = stats.sort_values(time_col)
            y_vals = stats[agg_func]
            if rolling_window:
                y_vals = y_vals.rolling(window=rolling_window, center=True).mean()
                
            ax.plot(stats[time_col], y_vals, marker='o', color=UNIFORM_BLUE, linewidth=2)
            if show_std:
                ax.fill_between(stats[time_col], 
                                stats[agg_func] - stats['std'],
                                stats[agg_func] + stats['std'], 
                                color=UNIFORM_BLUE, alpha=0.15)

        ax.set_title(str(facet) if facet else 'Signal Analysis')
        ax.set_xlabel(time_col)
        ax.set_ylabel('Value')
        if group_col:
            ax.legend(fontsize=8)

    for i in range(n_facets, n_rows * n_cols):
        fig.delaxes(axes[i // 2, i % 2])

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
