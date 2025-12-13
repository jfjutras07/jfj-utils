import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import math
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#--- Function : plot_numeric_distribution ---
def plot_numeric_distribution(df, numeric_cols, bins=40):
    """
    Plot histograms + KDE and boxplots for selected numeric columns.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to visualize (must be explicitly provided)
    - bins: number of bins for histogram
    """
    plt.style.use('seaborn-v0_8')
    
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        fig, axes = plt.subplots(1, 2, figsize = (14, 4))
        
        # Histogram + KDE
        sns.histplot(df[col].dropna(), kde=True, bins=bins, ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")
        
        # Boxplot
        sns.boxplot(x=df[col].dropna(), ax=axes[1])
        axes[1].set_title(f"Outliers in {col}")
        axes[1].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : qq_plot_numeric ---
def qq_plot_numeric(df, numeric_cols=None):
    """
    Generate Q-Q plots for numeric columns in a DataFrame to check normality.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to visualize (default: all numeric)

    Returns:
    - None (displays plots)
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    plt.style.use('seaborn-v0_8')
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        fig, ax = plt.subplots(figsize = (6, 6))
        stats.probplot(col_data, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot of {col}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.tight_layout()
        plt.show()
        plt.close()

#--- Function : plot_correlation_heatmap ---
def plot_correlation_heatmap(df, numeric_cols = None, method = 'spearman', figsize = (12,8), cmap = 'coolwarm', annot = True, fmt = ".2f"):
    """
    Plot a correlation heatmap for numeric columns in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to include (default: all numeric columns)
    - method: correlation method ('pearson', 'spearman', 'kendall')
    - figsize: figure size
    - cmap: colormap for heatmap
    - annot: whether to annotate the correlation coefficients
    - fmt: string formatting for annotations
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    corr_matrix = df[numeric_cols].corr(method=method)

    plt.figure(figsize = figsize)
    sns.heatmap(corr_matrix, annot = annot, fmt = fmt, cmap = cmap)
    plt.title(f"{method.capitalize()} Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    plt.close()

#--- Function plot_pairplot ---
def plot_pairplot(df, features, hue=None, diag_kind='kde', corner=True, alpha=0.5, figsize=(10, 10)):
    """
    Create a Seaborn pairplot for selected features of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - features: list of numerical feature names to include in the pairplot
    - hue: optional categorical column name for color grouping
    - diag_kind: type of plot for the diagonal ('hist' or 'kde')
    - corner: if True, plots only the lower triangle
    - alpha: transparency of points
    - figsize: size of the figure
    """
    sns.set(style="ticks")
    
    pairplot = sns.pairplot(
        df[features],
        hue=hue,
        diag_kind=diag_kind,
        corner=corner,
        plot_kws={'alpha': alpha}
    )
    pairplot.fig.set_size_inches(figsize)
    pairplot.fig.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()

#--- Function : plot_violin_grid ---
def plot_violin_grid(
    df,
    value_cols,
    group_col='Economic_status',
    n_rows=2,
    n_cols=2,
    palette='pastel'
):
    """
    Generic function to plot violin plots in a grid layout.
    
    Parameters:
        df: pd.DataFrame containing the data
        value_cols: list of numeric columns to plot
        group_col: column representing the grouping variable (e.g., 'Developed' vs 'Developing')
        n_rows, n_cols: number of rows and columns per figure
        palette: color palette for violin plots
    """
    #Loop through variables in batches that fit the grid
    for i in range(0, len(value_cols), n_rows * n_cols):
        batch = value_cols[i:i + n_rows * n_cols]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), sharey=False)
        axes = axes.flatten()  # Flatten in case of multiple rows/columns

        for ax, col in zip(axes, batch):
            # Violin plot: chosen because it shows the full distribution of the data,
            # including density, median, and quartiles, allowing comparison between groups
            sns.violinplot(
                data=df,
                x=group_col,
                y=col,
                palette=palette,
                inner='quartile',
                ax=ax
            )
            #Set plot title and axis labels
            ax.set_title(f'Distribution of {col} by {group_col}')
            ax.set_xlabel(group_col)
            ax.set_ylabel(col)
            # Add grid lines for easier reading
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        #Remove any empty subplots
        for j in range(len(batch), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

#--- Function : plot_scatter_grid ---
def plot_scatter_grid(df: pd.DataFrame, x_cols: list, y_cols: list, group_col: str = None,
                      group_labels: dict = None, n_cols_per_row: int = 2, figsize=(14,6)):
    """
    Generic function to plot scatterplots in a grid layout.
    
    Parameters:
    - df: DataFrame containing the data
    - x_cols: list of columns for x-axis
    - y_cols: list of columns for y-axis
    - group_col: optional column to color points by (categorical)
    - group_labels: optional dict to map group values to readable labels
    - n_cols_per_row: number of plots per row
    - figsize: figure size per row
    """
    
    #Determine number of plots
    num_plots = len(x_cols)
    n_rows = (num_plots + n_cols_per_row - 1) // n_cols_per_row
    
    plt.figure(figsize=(figsize[0], n_rows * figsize[1]))
    
    for i, (x, y) in enumerate(zip(x_cols, y_cols)):
        ax = plt.subplot(n_rows, n_cols_per_row, i+1)
        
        if group_col:
            plot_data = df.copy()
            if group_labels:
                plot_data['Group'] = plot_data[group_col].map(group_labels)
            else:
                plot_data['Group'] = plot_data[group_col]
            sns.scatterplot(data=plot_data, x=x, y=y, hue='Group', palette='tab10')
        else:
            sns.scatterplot(data=df, x=x, y=y, color='blue')
        
        #Titles and labels
        ax.set_title(f'{y} vs {x}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
