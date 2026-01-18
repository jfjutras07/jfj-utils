import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import parallel_backend
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning

# Importing centralized style constants
from .style import UNIFORM_BLUE, PALE_PINK, SEQUENTIAL_CMAP, WHITE, GREY_DARK, DEFAULT_FIGSIZE

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
    Displays a classification dashboard with three key visualizations:
    Learning Curves, Confusion Matrix, and ROC Curve.
    Fully suppresses ConvergenceWarnings and ensures model convergence.
    """

    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.simplefilter("ignore", UserWarning)

    # --- Force convergence for SAG/SAGA/LogisticRegression/Linear models ---
    if hasattr(model, "set_params"):
        params = model.get_params()
        if "max_iter" in params:
            model.set_params(max_iter=50000)
        if "tol" in params:
            model.set_params(tol=1e-4)

    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Learning Curves ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with parallel_backend("loky", inner_max_num_threads=1):
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

    # --- Confusion Matrix ---
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=SEQUENTIAL_CMAP,
        ax=axes[1],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )

    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticklabels(["Stay", "Leave"])
    axes[1].set_yticklabels(["Stay", "Leave"])

    # --- ROC Curve ---
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    axes[2].plot(fpr, tpr, color=PALE_PINK, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    axes[2].plot([0, 1], [0, 1], color=GREY_DARK, lw=2, linestyle="--")
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate (Recall)")
    axes[2].set_title("Receiver Operating Characteristic (ROC)")
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Diagnostics logs ---
    gap = train_mean[-1] - test_mean[-1]

    print("--- Classification Diagnostics Summary ---")
    print(f"Generalization Gap (Train-CV F1) : {gap:.4f}")
    print(f"Final Test ROC-AUC : {roc_auc:.4f}")
    print("-" * 42)
