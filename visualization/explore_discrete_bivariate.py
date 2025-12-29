# --- Bivariate dot plot ---
def plot_discrete_dot_bivariate(df, col, hue_col='Gender', normalize=True, figsize=(8,4)):
    """
    Dot plot for discrete variables bivariately.
    Two dots per category (Male/Female), side by side visually.
    Labels next to dots, small offset, colors by hue_col.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    colors = {'Male': '#ADD8E6', 'Female': '#90EE90'}
    categories = df[col].dropna().unique()
    offsets = {'Male': -0.15, 'Female': 0.15}  # vertical offset

    fig, ax = plt.subplots(figsize=figsize)

    for gender in df[hue_col].unique():
        subset = df[df[hue_col] == gender]
        counts = subset[col].value_counts()
        if normalize:
            counts = counts / counts.sum()

        for i, cat in enumerate(counts.index):
            x = counts[cat]
            y = i + offsets[gender]
            ax.plot(x, y, 'o', color=colors[gender], label=gender if i==0 else "")
            ax.text(x, y, f" {x:.2f}" if normalize else f" {int(x)}",
                    va='center', fontsize=9, color='black')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([str(c) for c in counts.index])
    ax.set_title(f'Dot plot of {col} by {hue_col}')
    ax.set_xlabel("Proportion" if normalize else "Count")
    ax.set_ylabel(col)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.close()


# --- Bivariate lollipop plot ---
def plot_discrete_lollipop_bivariate(df, col, hue_col='Gender', normalize=True, figsize=(8,4)):
    """
    Lollipop plot for discrete variables bivariately.
    Two lollipop lines per category (Male/Female), slightly offset vertically.
    Labels next to lollipops, colors by hue_col.
    """
    if col not in df.columns or hue_col not in df.columns:
        print(f"Warning: {col} or {hue_col} not found. Skipping.")
        return

    colors = {'Male': '#ADD8E6', 'Female': '#90EE90'}
    categories = df[col].dropna().unique()
    offsets = {'Male': -0.15, 'Female': 0.15}  # vertical offset

    fig, ax = plt.subplots(figsize=figsize)

    for gender in df[hue_col].unique():
        subset = df[df[hue_col] == gender]
        counts = subset[col].value_counts()
        if normalize:
            counts = counts / counts.sum()

        for i, cat in enumerate(counts.index):
            val = counts[cat]
            y = i + offsets[gender]
            ax.hlines(y=y, xmin=0, xmax=val, color=colors[gender], linewidth=4)
            ax.plot(val, y, 'o', color=colors[gender])
            ax.text(val, y, f" {val:.2f}" if normalize else f" {int(val)}",
                    va='center', fontsize=9, color='black')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([str(c) for c in counts.index])
    ax.set_title(f'Lollipop plot of {col} by {hue_col}')
    ax.set_xlabel("Proportion" if normalize else "Count")
    ax.set_ylabel(col)
    ax.legend(df[hue_col].unique(), loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.close()
