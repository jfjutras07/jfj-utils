import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from .style import UNIFORM_BLUE, PALE_PINK
from visualization.style import SEQUENTIAL_CMAP

#---Function:plot_box_grid---
def plot_box_grid(df, value_cols, group_col='Economic_status', n_rows=2, n_cols=2, palette=UNIFORM_BLUE):
    if isinstance(group_col,str):
        group_col=[group_col]
    y_col=value_cols[0]
    plots_per_fig=n_rows*n_cols
    for i in range(0,len(group_col),plots_per_fig):
        batch=group_col[i:i+plots_per_fig]
        if len(batch)==1:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(data=df, x=batch[0], y=y_col, palette=palette, ax=ax)
            ax.set_title(f'{y_col} by {batch[0]}'); ax.set_xlabel(batch[0]); ax.set_ylabel(y_col)
            ax.grid(axis='y',linestyle='--',alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout(); plt.show()
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols,5*n_rows), sharey=True)
            axes=axes.flatten()
            for ax, grp in zip(axes,batch):
                sns.boxplot(data=df, x=grp, y=y_col, palette=palette, ax=ax)
                ax.set_title(f'{y_col} by {grp}'); ax.set_xlabel(grp); ax.set_ylabel(y_col)
                ax.grid(axis='y',linestyle='--',alpha=0.5); ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            for j in range(len(batch), len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout(); plt.show()

#---Function: plot_box_plot---
def plot_box_plot(df, value_cols, category_col, hue_col, palette={0:UNIFORM_BLUE,1:PALE_PINK}, figsize=(16, 6)):
    y_col=value_cols[0]
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=category_col, y=y_col, hue=hue_col, palette=palette, dodge=True)
    plt.title(f'{y_col} by {category_col} and {hue_col}'); plt.xlabel(category_col); plt.ylabel(y_col)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30, ha='right'); plt.legend(title=hue_col); plt.tight_layout(); plt.show()

#---Function: plot_correlation_heatmap---
def plot_correlation_heatmap(df, numeric_cols=None, method='spearman', figsize=(12,8),
                             cmap=SEQUENTIAL_CMAP, annot=True, fmt=".2f"):
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
    if columns_col is not None:
        pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_col, aggfunc=aggfunc)
    else:
        pivot = df.groupby(index_col)[value_col].agg(aggfunc).to_frame()

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5)
    plt.title(f"{aggfunc.capitalize()} {value_col} by {index_col}" + (f" and {columns_col}" if columns_col else ""))
    plt.ylabel(index_col)
    plt.xlabel(columns_col if columns_col else value_col)
    plt.show()
                          
#---Function: plot_numeric_bivariate---
def plot_numeric_bivariate(df, numeric_cols, hue='Gender', bins=40):
    groups=df[hue].unique()
    if len(groups)!=2: print("Error: exactly 2 groups required."); return
    palette={groups[0]:UNIFORM_BLUE, groups[1]:PALE_PINK}
    for col in numeric_cols:
        fig, axes=plt.subplots(1,3,figsize=(18,4))
        sns.histplot(df[df[hue]==groups[0]], x=col, kde=True, bins=bins, color=palette[groups[0]], alpha=0.6, ax=axes[0])
        axes[0].set_title(f"{col} distribution - {groups[0]}"); axes[0].set_xlabel(col); axes[0].set_ylabel("Count")
        sns.histplot(df[df[hue]==groups[1]], x=col, kde=True, bins=bins, color=palette[groups[1]], alpha=0.6, ax=axes[1])
        axes[1].set_title(f"{col} distribution - {groups[1]}"); axes[1].set_xlabel(col); axes[1].set_ylabel("Count")
        sns.boxplot(x=hue, y=col, data=df, palette=palette, ax=axes[2])
        axes[2].set_title(f"{col} by {hue}"); axes[2].set_xlabel(hue); axes[2].set_ylabel(col)
        plt.tight_layout(); plt.show(); plt.close()

#---Function: plot_numeric_distribution---
def plot_numeric_distribution(df, numeric_cols, bins=40):
    for col in numeric_cols:
        if col not in df.columns: print(f"Warning: {col} not found."); continue
        fig, axes=plt.subplots(1,2,figsize=(14,4))
        sns.histplot(df[col].dropna(), kde=True, bins=bins, color=UNIFORM_BLUE, ax=axes[0])
        axes[0].set_title(f"Distribution of {col}"); axes[0].set_xlabel(col); axes[0].set_ylabel("Count")
        sns.boxplot(x=df[col].dropna(), color=UNIFORM_BLUE, ax=axes[1])
        axes[1].set_title(f"Outliers in {col}"); axes[1].set_xlabel(col)
        plt.tight_layout(); plt.show(); plt.close()

#---Function: plot_pairplot---
def plot_pairplot(df, features, hue=None, diag_kind='kde', corner=True, alpha=0.5, figsize=(10,10)):
    sns.set(style="ticks")
    pairplot=sns.pairplot(df[features], hue=hue, diag_kind=diag_kind, corner=corner, plot_kws={'alpha': alpha})
    pairplot.fig.set_size_inches(figsize); pairplot.fig.suptitle("Pairplot of Selected Features", y=1.02); plt.show()

#---Function: plot_scatter_grid---
def plot_scatter_grid(df: pd.DataFrame, x_cols: list, y_cols: list, group_col: str = None,
                      group_labels: dict = None, n_cols_per_row: int = 2, figsize=(14,6)):
    num_plots=len(x_cols)
    if num_plots==1:
        fig, ax=plt.subplots(figsize=(10,6)); x, y=x_cols[0], y_cols[0]
        if group_col:
            plot_data=df.copy()
            plot_data['Group']=plot_data[group_col].map(group_labels) if group_labels else plot_data[group_col]
            sns.scatterplot(data=plot_data, x=x, y=y, hue='Group', palette={k:UNIFORM_BLUE for k in plot_data['Group'].unique()}, s=60)
        else: sns.scatterplot(data=df, x=x, y=y, color=UNIFORM_BLUE, s=60)
        ax.set_title(f'{y} vs {x}'); ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(True); plt.tight_layout(); plt.show()
    else:
        n_rows=(num_plots+n_cols_per_row-1)//n_cols_per_row
        fig, axes=plt.subplots(n_rows, n_cols_per_row, figsize=(figsize[0], n_rows*figsize[1])); axes=axes.flatten()
        for i,(x,y) in enumerate(zip(x_cols,y_cols)):
            ax=axes[i]
            if group_col:
                plot_data=df.copy(); plot_data['Group']=plot_data[group_col].map(group_labels) if group_labels else plot_data[group_col]
                sns.scatterplot(data=plot_data, x=x, y=y, hue='Group', palette={k:UNIFORM_BLUE for k in plot_data['Group'].unique()}, s=40, ax=ax)
            else: sns.scatterplot(data=df, x=x, y=y, color=UNIFORM_BLUE, s=40, ax=ax)
            ax.set_title(f'{y} vs {x}'); ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(True)
        for j in range(len(x_cols), len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); plt.show()

#---Function: plot_swarm_grid---
def plot_swarm_grid(df, value_cols, group_col='Economic_status', hue_col=None, n_rows=2, n_cols=2, palette=UNIFORM_BLUE, dodge=True, figsize_single=(10,6), figsize_grid=(6,5)):
    if isinstance(group_col,str): group_col=[group_col]
    y_col=value_cols[0]; plots_per_fig=n_rows*n_cols
    for i in range(0,len(group_col),plots_per_fig):
        batch=group_col[i:i+plots_per_fig]
        if len(batch)==1:
            fig, ax=plt.subplots(figsize=figsize_single)
            sns.swarmplot(data=df, x=batch[0], y=y_col, hue=hue_col, dodge=dodge if hue_col else False, palette={0:UNIFORM_BLUE,1:PALE_PINK} if hue_col else UNIFORM_BLUE, ax=ax)
            ax.set_title(f'{y_col} by {batch[0]}'); ax.set_xlabel(batch[0]); ax.set_ylabel(y_col); ax.grid(axis='y',linestyle='--',alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right'); plt.tight_layout(); plt.show()
        else:
            fig, axes=plt.subplots(n_rows,n_cols,figsize=(figsize_grid[0]*n_cols,figsize_grid[1]*n_rows), sharey=True)
            axes=axes.flatten()
            for ax, grp in zip(axes,batch):
                sns.swarmplot(data=df, x=grp, y=y_col, hue=hue_col, dodge=dodge if hue_col else False, palette={0:UNIFORM_BLUE,1:PALE_PINK} if hue_col else UNIFORM_BLUE, ax=ax)
                ax.set_title(f'{y_col} by {grp}'); ax.set_xlabel(grp); ax.set_ylabel(y_col); ax.grid(axis='y',linestyle='--',alpha=0.5)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            for j in range(len(batch), len(axes)): axes[j].set_visible(False)
            plt.tight_layout(); plt.show()

#---Function: plot_violin_grid---
def plot_violin_grid(df, value_cols, group_col='Economic_status', n_rows=2, n_cols=2, palette=UNIFORM_BLUE):
    if isinstance(group_col,str): group_col=[group_col]
    y_col=value_cols[0]; plots_per_fig=n_rows*n_cols
    for i in range(0,len(group_col),plots_per_fig):
        batch=group_col[i:i+plots_per_fig]
        if len(batch)==1:
            fig, ax=plt.subplots(figsize=(10,6))
            sns.violinplot(data=df, x=batch[0], y=y_col, palette=palette, inner='quartile', ax=ax)
            ax.set_title(f'{y_col} by {batch[0]}'); ax.set_xlabel(batch[0]); ax.set_ylabel(y_col)
            ax.grid(axis='y',linestyle='--',alpha=0.5); ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right'); plt.tight_layout(); plt.show()
        else:
            fig, axes=plt.subplots(n_rows,n_cols,figsize=(6*n_cols,5*n_rows), sharey=True)
            axes=axes.flatten()
            for ax,grp in zip(axes,batch):
                sns.violinplot(data=df, x=grp, y=y_col, palette=palette, inner='quartile', ax=ax)
                ax.set_title(f'{y_col} by {grp}'); ax.set_xlabel(grp); ax.set_ylabel(y_col)
                ax.grid(axis='y',linestyle='--',alpha=0.5); ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            for j in range(len(batch), len(axes)): axes[j].set_visible(False)
            plt.tight_layout(); plt.show()

#---Function: scatter_numeric---
def scatter_numeric(df, target_col, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols=df.select_dtypes(include='number').columns.tolist()
        numeric_cols=[c for c in numeric_cols if c!=target_col]
    n_cols=2; n_rows=math.ceil(len(numeric_cols)/n_cols)
    fig, axes=plt.subplots(n_rows,n_cols,figsize=(12,5*n_rows)); axes=axes.flatten()
    for i,col in enumerate(numeric_cols):
        axes[i].scatter(df[col], df[target_col], alpha=0.6, color=UNIFORM_BLUE)
        axes[i].set_xlabel(col); axes[i].set_ylabel(target_col); axes[i].set_title(f"{col} vs {target_col}")
    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout(); plt.show()
