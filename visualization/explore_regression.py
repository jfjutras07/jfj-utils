import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['PYTHONWARNINGS'] = 'ignore'

from sklearn.model_selection import learning_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Function : plot_regression_diagnostics ---
def plot_regression_diagnostics(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    critical_feature,
    cv=5,
    colors=None,
    figsize=(16, 6)
):
    """
    Evaluates model capacity via learning curves and segment bias via error analysis.
    Suppresses ConvergenceWarnings during parallel processing.
    """

    if colors is None:
        colors = ["#1f77b4", "#f4a3a8"]

    if not isinstance(model, Pipeline):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    if hasattr(model, "set_params"):
        params = model.get_params()
        if "model__max_iter" in params:
            model.set_params(model__max_iter=10000)
        elif "max_iter" in params:
            model.set_params(max_iter=10000)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    axes[0].plot(train_sizes, train_mean, label="Training Score", color=colors[0], lw=2)
    axes[0].plot(train_sizes, test_mean, label="Validation Score (CV)", color=colors[1], linestyle="--", lw=2)
    axes[0].set_title("Learning Curves: Model Capacity", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Training Set Size")
    axes[0].set_ylabel("R2 Score")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    abs_error = np.abs(y_test - y_pred)
    feature_vals = X_test[critical_feature]

    if pd.api.types.is_numeric_dtype(feature_vals) and feature_vals.nunique() > 15:
        sns.scatterplot(x=feature_vals, y=abs_error, alpha=0.5, color=colors[0], ax=axes[1])
        sns.regplot(x=feature_vals, y=abs_error, scatter=False, color=colors[1], ax=axes[1])
    else:
        sns.boxplot(x=feature_vals, y=abs_error, color=colors[0], ax=axes[1],
                    medianprops={"color": "white", "linewidth": 2})
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    axes[1].set_title(f"Segment Bias: Error by {critical_feature}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(critical_feature)
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    gap = train_mean[-1] - test_mean[-1]
    print("--- Robustness Diagnostics Summary ---")
    print(f"Generalization Gap (Train-CV) : {gap:.4f}")
    
    if pd.api.types.is_numeric_dtype(feature_vals) and feature_vals.nunique() > 15:
        correlation = pd.Series(feature_vals).corr(pd.Series(abs_error))
        print(f"Error Correlation with {critical_feature} : {correlation:.4f}")
    print("-" * 40)
