import matplotlib.pyplot as plt
import warnings
import math
from .style import UNIFORM_BLUE, PALE_PINK

warnings.filterwarnings("ignore")

#---Function: plot_binary_distribution---
def plot_binary_distribution(df, binary_cols, figsize=(12,4)):
    """
    Plot binary distributions as pie charts.
    Each binary variable gets two plots (proportion, counts).
    Two variables are displayed per row (4 plots total).
    If an odd number of variables is provided, the last row is half-filled.
    """

    #Keep only existing columns
    binary_cols = [col for col in binary_cols if col in df.columns]
    n_cols = len(binary_cols)
    if n_cols == 0:
        print("No valid binary columns provided.")
        return

    n_rows = math.ceil(n_cols / 2)
    fig, axes = plt.subplots(n_rows, 4, figsize=(figsize[0], figsize[1] * n_rows))

    #Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    colors = [UNIFORM_BLUE, PALE_PINK]

    for i, col in enumerate(binary_cols):
        row = i // 2
        base_col = (i % 2) * 2  # 0 for left pair, 2 for right pair

        series = df[col].dropna()
        counts = series.value_counts().sort_index()
        labels = [str(i) for i in counts.index]
        sizes = counts.values
        total = sizes.sum()

        #Proportion plot
        axes[row, base_col].pie(
            sizes,
            labels=labels,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        axes[row, base_col].set_title(f"{col} - Proportion")
        axes[row, base_col].set_aspect('equal')

        #Count plot
        axes[row, base_col + 1].pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{int(round(p/100*total))}",
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        axes[row, base_col + 1].set_title(f"{col} - Counts")
        axes[row, base_col + 1].set_aspect('equal')

    #Turn off unused axes (odd number of variables)
    for j in range(n_cols * 2, n_rows * 4):
        axes.flatten()[j].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
