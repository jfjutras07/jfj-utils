import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .style import UNIFORM_BLUE, PALE_PINK

#--- Function : plot_predicted_actual ---
def plot_predicted_actual(y_true, y_pred, feature_names=None, model_name="Model", colors=None):
    """
    Comparison between predicted and actual values for regression models.
    Supports both single-target and multi-target regression.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]

    # Convert to numpy arrays for consistency
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Handle dimensions for single or multi-target
    if y_true_np.ndim == 1:
        y_true_np = y_true_np.reshape(-1, 1)
        y_pred_np = y_pred_np.reshape(-1, 1)

    n_targets = y_true_np.shape[1]

    # Extract feature names from DataFrame if available
    if feature_names is None:
        if isinstance(y_true, pd.DataFrame):
            feature_names = y_true.columns.tolist()
        elif isinstance(y_true, pd.Series):
            feature_names = [y_true.name]
        else:
            feature_names = [f"Target {i+1}" for i in range(n_targets)]

    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), squeeze=False)
    axes = axes.flatten()

    for i in range(n_targets):
        ax = axes[i]
        
        # Scatter actual vs predicted
        ax.scatter(y_true_np[:, i], y_pred_np[:, i], alpha=0.6, color=colors[0], edgecolor='white', s=50)

        # Perfect prediction diagonal line
        min_val = min(y_true_np[:, i].min(), y_pred_np[:, i].min())
        max_val = max(y_true_np[:, i].max(), y_pred_np[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=2, color=colors[1])

        ax.set_xlabel(f"Actual {feature_names[i]}")
        ax.set_ylabel(f"Predicted {feature_names[i]}")
        ax.set_title(f"{model_name}: {feature_names[i]}")

    plt.tight_layout()
    plt.show()
