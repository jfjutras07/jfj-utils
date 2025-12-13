import matplotlib.pyplot as plt
import pandas as pd

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
    title=None
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
    """
    
    xlabel = xlabel or time_col
    ylabel = ylabel or value_col
    title = title or f'{value_col} over {time_col}'
    
    if group_col:
        grouped = df.groupby([time_col, group_col])[value_col].agg(agg_func).reset_index()
        
        if group_labels:
            grouped['Group'] = grouped[group_col].map(group_labels)
        else:
            grouped['Group'] = grouped[group_col]
        
        plt.figure(figsize=figsize)
        for grp in grouped['Group'].unique():
            subset = grouped[grouped['Group'] == grp]
            plt.plot(subset[time_col], subset[value_col], marker='o', label=grp)
    else:
        grouped = df.groupby(time_col)[value_col].agg(agg_func).reset_index()
        plt.figure(figsize=figsize)
        plt.plot(grouped[time_col], grouped[value_col], marker='o')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if group_col:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
