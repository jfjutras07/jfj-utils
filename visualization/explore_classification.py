import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import parallel_backend
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning

# Global filter for warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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
    Handles both direct estimators and Scikit-Learn Pipelines for parameter updates.
    Checks parameter existence before injection to avoid XGBoost/Tree errors.
    """

    # --- Section: Parameter Injection (Avoiding ValueError & Warnings) ---
    if hasattr(model, "set_params"):
        params = model.get_params()
        settings = {}
        
        # Define target parameters for linear models
        potential_updates = {"max_iter": 20000, "tol": 1e-3}

        if 'model' in params:
            # Logic for Pipelines: check if the inner model has these attributes
            inner_params = params['model'].get_params()
            for key, value in potential_updates.items():
                if key in inner_params:
                    settings[f"model__{key}"] = value
        else:
            # Logic for direct estimators
            for key, value in potential_updates.items():
                if key in params:
                    settings[key] = value
        
        # Apply parameters only if they are relevant to this specific model
        if settings:
            try:
                model.set_params(**settings)
            except Exception:
                pass

    if colors is None:
        colors = ['#1f77b4', '#ff7f0e']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Section: Learning Curves ---
    with warnings.catch_warnings():
        # Capture warnings from parallel workers
        warnings.simplefilter("ignore")
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
    axes[0].set_title("Learning Curves")
    axes[0].set_xlabel("Training Set Size")
    axes[0].set_ylabel("F1 Score")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    # --- Section: Confusion Matrix ---
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
    axes[1].set_xticklabels(["Stay", "Leave"])
    axes[1].set_yticklabels(["Stay", "Leave"])

    # --- Section: ROC Curve ---
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    axes[2].plot(fpr, tpr, color=colors[1], lw=2, label=f"AUC = {roc_auc:.2f}")
    axes[2].plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    axes[2].set_title("ROC Curve")
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Diagnostics Summary: ROC-AUC = {roc_auc:.4f}")

# --- Function : plot_feature_importance ---
def plot_feature_importance(model, feature_names, figsize=(8, 5)):
    """
    Standard feature importance plot for Logistic Regression with your colors.
    Positive coefficients in PALE_PINK, negative coefficients in UNIFORM_BLUE.
    Horizontal bars aligned on zero, sorted for readability.
    """
    # --- Extract coefficients ---
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        coefs = model.named_steps['model'].coef_[0]
    elif hasattr(model, 'coef_'):
        coefs = model.coef_[0]
    else:
        raise ValueError("Model does not have coefficients")

    # --- Prepare DataFrame ---
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    }).sort_values(by='Coefficient', ascending=True)  # standard: smallest on top, largest on bottom

    # --- Plot ---
    plt.figure(figsize=figsize)
    colors = [PALE_PINK if c > 0 else UNIFORM_BLUE for c in importance_df['Coefficient']]
    plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, edgecolor="black")
    plt.axvline(0, color=GREY_DARK, linewidth=1)
    plt.title("Feature Importance (Logistic Regression Coefficients)", fontweight='bold')
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
