import os
import warnings

# 1. Set environment variable BEFORE importing sklearn to silence sub-processes
os.environ["PYTHONWARNINGS"] = "ignore::sklearn.exceptions.ConvergenceWarning"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import parallel_backend
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning

# 2. Global filter for the main process
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
    Displays a classification dashboard with Learning Curves, Confusion Matrix, and ROC Curve.
    Silences ConvergenceWarnings by adjusting model parameters and using warning filters.
    """

    # --- Robust Model Configuration ---
    if hasattr(model, "set_params"):
        params = model.get_params()
        settings_to_update = {}
        
        # Increasing iterations and slightly relaxing tolerance to ensure "soft" convergence
        if "max_iter" in params:
            settings_to_update["max_iter"] = 10000
        if "tol" in params:
            settings_to_update["tol"] = 1e-3  # Slightly higher tolerance prevents endless oscillation
            
        if settings_to_update:
            model.set_params(**settings_to_update)

    if colors is None:
        # Fallback to defaults if style constants are missing
        try:
            from .style import UNIFORM_BLUE, PALE_PINK
            colors = [UNIFORM_BLUE, PALE_PINK]
        except (ImportError, ValueError):
            colors = ["#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Section 1: Learning Curves ---
    # Wrap in a context manager to ensure silence during multi-processing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
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

    # --- Section 2: Confusion Matrix ---
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues", # Simplified for robustness
        ax=axes[1],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )

    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticklabels(["Stay", "Leave"])
    axes[1].set_yticklabels(["Stay", "Leave"])

    # --- Section 3: ROC Curve ---
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    axes[2].plot(fpr, tpr, color=colors[1], lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    axes[2].plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate (Recall)")
    axes[2].set_title("Receiver Operating Characteristic (ROC)")
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Final Summary Logs ---
    gap = train_mean[-1] - test_mean[-1]
    print("--- Classification Diagnostics Summary ---")
    print(f"Generalization Gap (Train-CV F1) : {gap:.4f}")
    print(f"Final Test ROC-AUC : {roc_auc:.4f}")
    print("-" * 42)
