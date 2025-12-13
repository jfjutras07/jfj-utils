import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_binary_distribution(df, binary_cols):
    """
    Plot side-by-side pie charts for binary columns (0/1) in the same row.

    For each binary column:
    - Left: Proportion
    - Right: Counts
    """
    for col in binary_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_index()  # 0 then 1
        labels = [str(i) for i in counts.index]
        sizes = counts.values
        colors = ["#ADD8E6", "#87CEFA"]  # Bleu pâle et légèrement plus foncé

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 ligne, 2 colonnes

        # --- Left: Proportion ---
        axes[0].pie(
            sizes,
            labels=labels,
            autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        axes[0].set_title(f"{col} - Proportion")

        # --- Right: Counts ---
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

        plt.tight_layout()
        plt.show()
        plt.close()
