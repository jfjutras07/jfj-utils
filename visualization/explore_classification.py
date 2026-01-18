import os
os.environ["PYTHONWARNINGS"] = "ignore:The max_iter was reached:sklearn.exceptions.ConvergenceWarning"

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import parallel_backend
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning

# Additional global filter for the current process
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Importing centralized style constants
# from .style import UNIFORM_BLUE, PALE_PINK, SEQUENTIAL_CMAP, WHITE, GREY_DARK, DEFAULT_FIGSIZE

# --- Function : plot_classification_diagnostics ---
def plot_classification_diagnostics(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    cv=5,
    colors=None,
    figsize=(20, 6)
):
    """
    Displays a classification dashboard. 
    Specific ConvergenceWarnings are suppressed via environment variables and local filters.
    """

    # --- Forced Model Adjustments ---
    if hasattr(model, "set_params"):
        # We relax the tolerance (tol) and set a high max_iter to reduce convergence friction
        model.set_params(max_iter=20000, tol=1e-3)

    if colors is None:
        colors = ['#1f77b4', '#ff7f0e'] # Standard defaults

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Learning Curves ---
    # Using catch_warnings here as a double safety layer for multi-processing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1 # Warnings usually leak from here
        )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    axes[0].plot(train_sizes, train_mean, label="Training F1", color=colors[0], lw=2)
    axes[0].plot(train_sizes, test_mean, label="Validation F1 (CV)", color=colors[1], linestyle="--", lw=2)
    axes[0].set_title("Learning Curves: Model Capacity")
    axes[0].set_xlabel("Training Set Size")
    axes[0].set_ylabel("F1 Weighted Score")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    # --- Confusion Matrix ---
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )

    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    # --- ROC Curve ---
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    axes[2].plot(fpr, tpr, color=colors[1], lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    axes[2].plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    axes[2].set_title("ROC Curve")
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Summary ---
    print(f"Diagnostics completed. Final ROC-AUC: {roc_auc:.4f}")
