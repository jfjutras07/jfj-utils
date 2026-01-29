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
from .style import UNIFORM_BLUE, PALE_PINK, GREY_DARK

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

# --- Function : plot_classification_impact ---
def plot_classification_impact(
    model,
    X_test,
    y_test,
    colors=None,
    figsize=(18, 6)
) :
    """
    Displays Lift and Cumulative Gains charts.
    Used to evaluate the business efficiency of a targeted strategy 
    compared to a random approach.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must support predict_proba.")

    # Data preparation for Ranking
    df_results = pd.DataFrame({'actual': y_test, 'proba': y_proba})
    df_results = df_results.sort_values(by='proba', ascending=False).reset_index(drop=True)
    
    # Calculate Cumulative Gains
    df_results['cumulative_positives'] = df_results['actual'].cumsum()
    total_positives = df_results['actual'].sum()
    df_results['gain'] = df_results['cumulative_positives'] / total_positives
    
    # Calculate Lift
    # (Proportion of positives in sample) / (Proportion of positives in total population)
    df_results['population_percentage'] = (df_results.index + 1) / len(df_results)
    df_results['lift'] = df_results['gain'] / df_results['population_percentage']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Cumulative Gains Chart ---
    axes[0].plot(df_results['population_percentage'], df_results['gain'], color=colors[0], lw=3, label="Model")
    axes[0].plot([0, 1], [0, 1], color=GREY_DARK, linestyle="--", label="Random Selection")
    axes[0].fill_between(df_results['population_percentage'], df_results['gain'], df_results['population_percentage'], alpha=0.1, color=colors[0])
    axes[0].set_title("Cumulative Gains Chart", fontweight='bold')
    axes[0].set_xlabel("% of Population Contacted (Sorted by Proba)")
    axes[0].set_ylabel("% of Total Targets Captured")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # --- Plot 2: Lift Chart ---
    axes[1].plot(df_results['population_percentage'], df_results['lift'], color=colors[1], lw=3, label="Lift Curve")
    axes[1].axhline(1, color=GREY_DARK, linestyle="--", label="Baseline (Lift=1)")
    axes[1].set_title("Lift Chart", fontweight='bold')
    axes[1].set_xlabel("% of Population Contacted")
    axes[1].set_ylabel("Lift (Multiplier)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

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

# --- Function : plot_precision_recall ---
def plot_precision_recall(
    model,
    X_test,
    y_test,
    colors=None,
    figsize=(18, 6)
):
    """
    Displays Precision-Recall Curve and Threshold Analysis.
    Optimizes the decision threshold based on the F1-Score.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must support predict_proba.")

    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    # --- F1-Score Optimization ---
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_f1 = f1_scores[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: PR Curve ---
    axes[0].plot(recall, precision, color=colors[0], lw=3, label=f"AP = {avg_precision:.2f}")
    axes[0].scatter(recall[best_idx], precision[best_idx], color='red', s=50, zorder=5, label=f"Best F1: {best_f1:.2f}")
    axes[0].fill_between(recall, precision, alpha=0.15, color=colors[0])
    axes[0].set_title("Precision-Recall Curve", fontweight='bold')
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.3)

    # --- Plot 2: Threshold Analysis ---
    axes[1].plot(thresholds, precision[:-1], label="Precision", color=colors[0], lw=2)
    axes[1].plot(thresholds, recall[:-1], label="Recall", color=colors[1], lw=2, linestyle="--")
    axes[1].plot(thresholds, f1_scores[:-1], label="F1-Score", color=GREY_DARK, lw=2, alpha=0.6)
    axes[1].axvline(best_threshold, color='red', linestyle="--", label=f"Best Threshold: {best_threshold:.2f}")
    
    axes[1].set_title("Metrics vs. Decision Threshold", fontweight='bold')
    axes[1].set_xlabel("Probability Threshold")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
