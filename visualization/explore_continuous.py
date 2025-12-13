import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    plt.figure(figsize=figsize)
    
    pairplot = sns.pairplot(
        df[features],
        hue=hue,
        diag_kind=diag_kind,
        corner=corner,
        plot_kws={'alpha': alpha}
    )
    pairplot.fig.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()
