import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def plot_discrete_distribution(df, discrete_cols, top_n=10):
    """
    Plot count histograms for discrete / categorical columns.

    Parameters:
    - df: pandas DataFrame
    - discrete_cols: list of discrete/categorical columns to visualize
    - top_n: number of most frequent categories to display (default: 10)
    """
    for col in discrete_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue
        
        counts = df[col].value_counts().nlargest(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(counts.index.astype(str), counts.values, color='lightgreen', edgecolor='black')
        ax.set_title(f"Counts of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()
