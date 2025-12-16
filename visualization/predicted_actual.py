import numpy as np
import matplotlib.pyplot as plt

#--- Function : predicted_actual ---
def predicted_actual(y_true, y_pred, feature_names=None, model_name="Model"):
    """
    Generic visualization for regression models comparing predicted vs actual values.
    Works for single-target and multi-target regression.

    Parameters
    ----------
    y_true : array-like or pandas DataFrame
        True target values. Can be 1D or 2D.
    y_pred : array-like
        Predicted target values. Must have the same shape as y_true.
    feature_names : list of str, optional
        Names of the target variables. If None and y_true is a DataFrame,
        column names will be used. Otherwise generic names are assigned.
    model_name : str, optional
        Name of the model to display in plot titles.

    Returns
    -------
    None
        Displays the plot.
    """

    #Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #Determine number of targets
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]

    #Default target names if none provided
    if feature_names is None:
        feature_names = [f"Target {i+1}" for i in range(n_targets)]

    #Plot
    plt.figure(figsize=(6 * n_targets, 5))

    for i in range(n_targets):
        plt.subplot(1, n_targets, i + 1)

        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())

        #Diagonal perfect prediction line
        plt.plot([min_val, max_val], [min_val, max_val],
                 linestyle='--', linewidth=2)

        plt.xlabel(f"Actual {feature_names[i]}")
        plt.ylabel(f"Predicted {feature_names[i]}")
        plt.title(f"{model_name}: {feature_names[i]}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
