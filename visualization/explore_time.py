import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#--- Function : plot_line_over_time ---
def plot_line_over_time(df: pd.DataFrame, value_col: str, group_col: str = None, agg_func='mean'):
    """
    Simple generic line plot for a numeric value over time.
    
    Parameters:
        df: DataFrame containing the data
        value_col: column to plot (numeric)
        group_col: optional column to group by (e.g., Developed/Developing)
        agg_func: aggregation function ('mean', 'median', etc.)
    """
    
    plt.figure(figsize=(10,6))
    
    if group_col:
        grouped = df.groupby(['Year', group_col])[value_col].agg(agg_func).reset_index()
        palette = sns.color_palette("tab10", n_colors=grouped[group_col].nunique())
        sns.lineplot(data=grouped, x='Year', y=value_col, hue=group_col, marker='o', palette=palette)
    else:
        grouped = df.groupby('Year')[value_col].agg(agg_func).reset_index()
        sns.lineplot(data=grouped, x='Year', y=value_col, marker='o')
    
    plt.title(f'{value_col} over Time')
    plt.xlabel('Year')
    plt.ylabel(value_col)
    plt.grid(True)
    plt.show()
