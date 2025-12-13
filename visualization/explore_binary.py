import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#--- Function : plot_binary_distribution ---
def plot_binary_distribution(df, binary_cols):
    """
    Plot pie charts for binary columns.
    
    Parameters:
    - df: pandas DataFrame
    - binary_cols: list of binary columns (0/1) to visualize
    """

    for col in binary_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue

        series = df[col].dropna()
        counts = series.value_counts().sort_index()  # Ensure 0 then 1

        labels = [str(i) for i in counts.index]
        sizes = counts.values
        colors = ["#ADD8E6", "#87CEFA"]  # Bleu pâle et légèrement plus foncé

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
        plt.close()
