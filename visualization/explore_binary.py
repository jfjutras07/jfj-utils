import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_binary_distribution(df, binary_cols):
    """
    Plot side-by-side pie charts for binary columns (0/1).

    Left: Proportion
    Right: Counts
    """
    for col in binary_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_index()  # 0 then 1
        labels = [str(i) for i in counts.index]
        sizes = counts.values
        colors = ["#ADD8E6", "#87CEFA"]

        fig = plt.figure(figsize=(10, 4))

        # Left pie chart: Proportion
        ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])  # [left, bottom, width, height]
        ax1.pie(
            sizes,
            labels=labels,
            autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        ax1.set_title(f"{col} - Proportion")
        ax1.set_aspect('equal')

        # Right pie chart: Counts
        ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])
        total = sizes.sum()
        ax2.pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{int(round(p/100*total))}",
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        ax2.set_title(f"{col} - Counts")
        ax2.set_aspect('equal')

        plt.show()
        plt.close()
