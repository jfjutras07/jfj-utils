import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import math
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#---Function:plot_box_grid---
def plot_box_grid(
    df,
    value_cols,
    group_col='Economic_status',
    n_rows=2,
    n_cols=2,
    palette='pastel'
):
    """
    Generic function to plot boxplots in a grid layout.
    
    Parameters:
        df:pd.DataFrame containing the data
        value_cols:list with ONE numeric column (dependent variable)
        group_col:list of grouping variables (independent variables)
        n_rows,n_cols:number of rows and columns per figure
        palette:color palette for boxplots
    """

    #Force list behavior
    if isinstance(group_col,str):
        group_col=[group_col]

    y_col=value_cols[0]
    plots_per_fig=n_rows*n_cols

    #Loop through grouping variables in fixed-size grids
    for i in range(0,len(group_col),plots_per_fig):
        batch=group_col[i:i+plots_per_fig]

        if len(batch)==1:
            #Single plot: bigger figure
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(
                data=df,
                x=batch[0],
                y=y_col,
                palette=palette,
                ax=ax
            )
            ax.set_title(f'{y_col} by {batch[0]}')
            ax.set_xlabel(batch[0])
            ax.set_ylabel(y_col)
            ax.grid(axis='y',linestyle='--',alpha=0.5)
            #Rotate x labels for readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            #Multiple plots: use grid
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6*n_cols,5*n_rows),
                sharey=True
            )
            axes = axes.flatten()

            for ax, grp in zip(axes, batch):
                sns.boxplot(
                    data=df,
                    x=grp,
                    y=y_col,
                    palette=palette,
                    ax=ax
                )
                ax.set_title(f'{y_col} by {grp}')
                ax.set_xlabel(grp)
                ax.set_ylabel(y_col)
                ax.grid(axis='y',linestyle='--',alpha=0.5)
                #Rotate x labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            #Remove unused axes
            for j in range(len(batch), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

#---Function: plot_box_plot---
def plot_box_plot(
    df,
    value_cols,
    category_col,
    hue_col,
    palette='pastel',
    figsize=(16, 6)
):
    """
    Generic function to plot a single boxplot figure with one categorical axis
    and a hue-based split (e.g., Gender).

    Parameters:
        df : pd.DataFrame
            Data containing the variables
        value_cols : list
            List with ONE numeric column (dependent variable)
        category_col : str
            Main categorical variable displayed on the x-axis (e.g., JobTitle)
        hue_col : str
            Secondary grouping variable used for color split
        palette : str or dict
            Color palette for hue categories
        figsize : tuple
            Size of the figure
    """

    #Force list behavior for value_cols
    y_col = value_cols[0]

    plt.figure(figsize=figsize)

    sns.boxplot(
        data=df,
        x=category_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        dodge=True
    )

    plt.title(f'{y_col} by {category_col} and {hue_col}')
    plt.xlabel(category_col)
    plt.ylabel(y_col)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    #Improve readability of categorical axis
    plt.xticks(rotation=30, ha='right')

    plt.legend(title=hue_col)
    plt.tight_layout()
    plt.show()

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

#---Function:plot_heatmap_grid---
def plot_heatmap_grid(
    df,
    value_col,
    index_col,
    columns_col=None,
    aggfunc='median',
    cmap='Blues',
    fmt=".0f",
    figsize=(10,6)
):
    """
    Generic function to plot a heatmap of aggregated numeric values.

    Parameters:
        df: pd.DataFrame containing the data
        value_col: numeric column to visualize (dependent variable)
        index_col: column to use as rows
        columns_col: column to use as columns (optional, for 2D heatmap)
        aggfunc: aggregation function for pivot_table ('median', 'mean', etc.)
        cmap: color map for heatmap
        fmt: string format for annotations
        figsize: figure size
    """

    #Create pivot table
    if columns_col is not None:
        pivot = df.pivot_table(
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc=aggfunc
        )
    else:
        # Only index, single column heatmap
        pivot = df.groupby(index_col)[value_col].agg(aggfunc).to_frame()

    #Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5
    )
    plt.title(f"{aggfunc.capitalize()} {value_col} by {index_col}" + (f" and {columns_col}" if columns_col else ""))
    plt.ylabel(index_col)
    if columns_col:
        plt.xlabel(columns_col)
    else:
        plt.xlabel(value_col)
    plt.show()

# --- Function : plot_numeric_bivariate ---
def plot_numeric_bivariate(df, numeric_cols, hue='Gender', bins=40):
    """
    For each numeric column in numeric_cols:
    - Plot histogram + KDE for the first group in hue
    - Plot histogram + KDE for the second group in hue
    - Plot combined boxplot for the column by hue
    
    Each row: 3 plots (hist1, hist2, boxplot)
    
    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to visualize
    - hue: categorical column to split distributions
    - bins: number of bins for histograms
    """
    import seaborn as sns
    plt.style.use('seaborn-v0_8')

    groups = df[hue].unique()
    if len(groups) != 2:
        print("Error: This function requires exactly 2 groups in the hue column.")
        return
    
    palette = {groups[0]: "#ADD8E6", groups[1]: "#90EE90"}  # light blue / light green

    for col in numeric_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # Histogram + KDE for first group
        sns.histplot(
            df[df[hue]==groups[0]],
            x=col,
            kde=True,
            bins=bins,
            color=palette[groups[0]],
            alpha=0.6,
            ax=axes[0]
        )
        axes[0].set_title(f"{col} distribution - {groups[0]}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")

        # Histogram + KDE for second group
        sns.histplot(
            df[df[hue]==groups[1]],
            x=col,
            kde=True,
            bins=bins,
            color=palette[groups[1]],
            alpha=0.6,
            ax=axes[1]
        )
        axes[1].set_title(f"{col} distribution - {groups[1]}")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Count")

        # Combined boxplot
        sns.boxplot(
            x=hue,
            y=col,
            data=df,
            palette=palette,
            ax=axes[2]
        )
        axes[2].set_title(f"{col} by {hue}")
        axes[2].set_xlabel(hue)
        axes[2].set_ylabel(col)

        plt.tight_layout()
        plt.show()
        plt.close()

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

#---Function:plot_scatter_grid---
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

    num_plots = len(x_cols)

    #Single plot: bigger figure
    if num_plots == 1:
        fig, ax = plt.subplots(figsize=(10,6))
        x, y = x_cols[0], y_cols[0]
        if group_col:
            plot_data = df.copy()
            if group_labels:
                plot_data['Group'] = plot_data[group_col].map(group_labels)
            else:
                plot_data['Group'] = plot_data[group_col]
            sns.scatterplot(data=plot_data, x=x, y=y, hue='Group', palette='tab10', s=60)
        else:
            sns.scatterplot(data=df, x=x, y=y, color='blue', s=60)

        ax.set_title(f'{y} vs {x}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    
    else:
        n_rows = (num_plots + n_cols_per_row - 1) // n_cols_per_row
        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(figsize[0], n_rows * figsize[1]))
        axes = axes.flatten()

        for i, (x, y) in enumerate(zip(x_cols, y_cols)):
            ax = axes[i]
            if group_col:
                plot_data = df.copy()
                if group_labels:
                    plot_data['Group'] = plot_data[group_col].map(group_labels)
                else:
                    plot_data['Group'] = plot_data[group_col]
                sns.scatterplot(data=plot_data, x=x, y=y, hue='Group', palette='tab10', s=40, ax=ax)
            else:
                sns.scatterplot(data=df, x=x, y=y, color='blue', s=40, ax=ax)

            ax.set_title(f'{y} vs {x}')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.grid(True)

        #Hide unused subplots
        for j in range(len(x_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

#---Function:plot_swarm_grid---
def plot_swarm_grid(
    df,
    value_cols,
    group_col='Economic_status',
    hue_col=None,
    n_rows=2,
    n_cols=2,
    palette='pastel',
    dodge=True,
    figsize_single=(10,6),
    figsize_grid=(6,5)
):
    """
    Generic function to plot swarm plots in a grid layout.

    Parameters:
        df: pd.DataFrame containing the data
        value_cols: list with ONE numeric column (dependent variable)
        group_col: list of grouping variables (independent variables)
        hue_col: column name for hue (optional)
        n_rows, n_cols: number of rows and columns per figure
        palette: color palette for swarm plots
        dodge: whether to separate hue levels (True by default)
        figsize_single: figsize for a single plot
        figsize_grid: base figsize per subplot when multiple plots
    """
    #Force list behavior
    if isinstance(group_col, str):
        group_col = [group_col]

    y_col = value_cols[0]
    plots_per_fig = n_rows * n_cols

    #Loop through grouping variables in fixed-size grids
    for i in range(0, len(group_col), plots_per_fig):
        batch = group_col[i:i + plots_per_fig]

        if len(batch) == 1:
            # Single plot: bigger figure
            fig, ax = plt.subplots(figsize=figsize_single)
            sns.swarmplot(
                data=df,
                x=batch[0],
                y=y_col,
                hue=hue_col,
                dodge=dodge if hue_col is not None else False,
                palette=palette,
                ax=ax
            )
            ax.set_title(f'{y_col} by {batch[0]}')
            ax.set_xlabel(batch[0])
            ax.set_ylabel(y_col)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            #Multiple plots: use grid
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize_grid[0]*n_cols, figsize_grid[1]*n_rows),
                sharey=True
            )
            axes = axes.flatten()

            for ax, grp in zip(axes, batch):
                sns.swarmplot(
                    data=df,
                    x=grp,
                    y=y_col,
                    hue=hue_col,
                    dodge=dodge if hue_col is not None else False,
                    palette=palette,
                    ax=ax
                )
                ax.set_title(f'{y_col} by {grp}')
                ax.set_xlabel(grp)
                ax.set_ylabel(y_col)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            #Remove unused axes
            for j in range(len(batch), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

#---Function:plot_violin_grid---
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
        df:pd.DataFrame containing the data
        value_cols:list with ONE numeric column (dependent variable)
        group_col:list of grouping variables (independent variables)
        n_rows,n_cols:number of rows and columns per figure
        palette:color palette for violin plots
    """

    #Force list behavior
    if isinstance(group_col,str):
        group_col=[group_col]

    y_col=value_cols[0]
    plots_per_fig=n_rows*n_cols

    #Loop through grouping variables in fixed-size grids
    for i in range(0,len(group_col),plots_per_fig):
        batch=group_col[i:i+plots_per_fig]

        if len(batch)==1:
            #Single plot: bigger figure
            fig, ax = plt.subplots(figsize=(10,6))
            sns.violinplot(
                data=df,
                x=batch[0],
                y=y_col,
                palette=palette,
                inner='quartile',
                ax=ax
            )
            ax.set_title(f'{y_col} by {batch[0]}')
            ax.set_xlabel(batch[0])
            ax.set_ylabel(y_col)
            ax.grid(axis='y',linestyle='--',alpha=0.5)
            #Rotate x labels for readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            #Multiple plots: use grid
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6*n_cols,5*n_rows),
                sharey=True
            )
            axes=axes.flatten()

            for ax,grp in zip(axes,batch):
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
                ax.grid(axis='y',linestyle='--',alpha=0.5)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            #Remove unused axes
            for j in range(len(batch), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

#--- Function : qq_plot_numeric ---
def qq_plot_numeric(df, numeric_cols=None):
    """
    Generate Q-Q plots for numeric columns in a DataFrame to check normality,
    displaying 2 plots per row.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to visualize (default: all numeric)

    Returns:
    - None (displays plots)
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    plt.style.use('seaborn-v0_8')
    
    n_cols = 2
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        col_data = df[col].dropna()
        stats.probplot(col_data, dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot of {col}")
        axes[i].set_xlabel("Theoretical Quantiles")
        axes[i].set_ylabel("Sample Quantiles")
    
    #Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

#--- Function residuals_fitted ---
def residuals_fitted(model, df=None, predictors=None, n_cols=2, resid_attr='resid', fitted_attr='fittedvalues'):
    """
    Generic residuals vs fitted values plots for any model with residuals and fitted values.

    Parameters
    ----------
    model : object
        Fitted model object containing residuals and fitted values.
    df : pandas.DataFrame, optional
        Dataset used to fit the model, required if predictors are provided.
    predictors : list of str, optional
        List of variable names in df to plot residuals against.
        If None, plots residuals vs fitted values only.
    n_cols : int
        Number of columns in subplot grid.
    resid_attr : str
        Attribute name in the model object for residuals (default 'resid').
    fitted_attr : str
        Attribute name in the model object for fitted values (default 'fittedvalues').

    Returns
    -------
    None
    """
    
    #Extract residuals and fitted values
    resid = getattr(model, resid_attr)
    fitted = getattr(model, fitted_attr)
    
    #If no predictors, just plot residuals vs fitted values
    if predictors is None:
        plt.figure(figsize=(8, 6))
        plt.scatter(fitted, resid, alpha=0.7)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        plt.show()
        return
    
    #Otherwise, plot residuals vs each predictor
    n_plots = len(predictors)
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    for i, var in enumerate(predictors):
        axes[i].scatter(df[var], resid, alpha=0.7)
        axes[i].axhline(0, color='red', linestyle='--')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Residuals')
        axes[i].set_title(f'Residuals vs {var}')
    
    #Hide unused subplots if any
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

#--- Function : scatter_numeric ---
def scatter_numeric(df, target_col, numeric_cols=None):
    """
    Generate scatter plots of numeric columns vs the target variable to explore
    potential relationships and variance patterns.

    Parameters:
    - df: pandas DataFrame
    - target_col: str, column name of the dependent variable
    - numeric_cols: list of numeric columns to visualize (default: all numeric except target_col)

    Returns:
    - None (displays plots)
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]

    plt.style.use('seaborn-v0_8')
    n_cols = 2
    n_rows = math.ceil(len(numeric_cols)/n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        axes[i].scatter(df[col], df[target_col], alpha=0.6)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f"{col} vs {target_col}")
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
