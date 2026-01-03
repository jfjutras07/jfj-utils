import matplotlib.pyplot as plt
import warnings
from .style import UNIFORM_BLUE, PALE_PINK

warnings.filterwarnings("ignore")

#---Function: plot_binary_distribution---
def plot_binary_distribution(df, binary_cols, figsize=(6,3)):
    """
    Plot two pie charts side by side for each binary column:
    - Left: Proportion
    - Right: Counts
    """
    for col in binary_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_index()
        labels = [str(i) for i in counts.index]
        sizes = counts.values
        colors = [UNIFORM_BLUE, PALE_PINK]  # UNIFORM_BLUE and pale pink for binary

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].pie(
            sizes,
            labels=labels,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        axes[0].set_title(f"{col} - Proportion")
        axes[0].set_aspect('equal')

        total = sizes.sum()
        axes[1].pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{int(round(p/100*total))}",
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        axes[1].set_title(f"{col} - Counts")
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.show()
        plt.close()
