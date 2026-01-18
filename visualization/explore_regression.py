import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

from visualization.style import UNIFORM_BLUE, PALE_PINK

def plot_regression_diagnostics(model, X_train, y_train, X_test, y_test, critical_feature, cv=5, 
                                 colors=None, figsize=(16, 6)):
    """
    Evaluates model capacity via learning curves and segment bias via error analysis.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK]
        
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Learning Curves (Capacity) ---
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='r2', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    axes[0].plot(train_sizes, train_mean, label='Training Score', color=colors[0], lw=2)
    axes[0].plot(train_sizes, test_mean, label='Validation Score (CV)', color=colors[1], linestyle='--', lw=2)
    axes[0].set_title('Learning Curves: Model Capacity')
    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('R2 Score')
    axes[0].legend(loc='best')

    # --- Slice Analysis (Segment Bias) ---
    if hasattr(model, "feature_names_in_"):
        X_pred = X_test[model.feature_names_in_]
    else:
        X_pred = X_test.select_dtypes(include=np.number)

    y_pred = model.predict(X_pred)
    abs_error = np.abs(y_test - y_pred)
    feature_vals = X_test[critical_feature]
    
    if feature_vals.nunique() > 15:
        sns.scatterplot(x=feature_vals, y=abs_error, alpha=0.5, color=colors[0], ax=axes[1])
        sns.regplot(x=feature_vals, y=abs_error, scatter=False, color=colors[1], ax=axes[1])
    else:
        sns.boxplot(x=feature_vals, y=abs_error, color=colors[0], ax=axes[1], 
                    medianprops={"color": "white", "linewidth": 2})
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right')

    axes[1].set_title(f'Segment Bias: Error by {critical_feature}')
    axes[1].set_xlabel(critical_feature)
    axes[1].set_ylabel('Absolute Error')

    plt.tight_layout()
    plt.show()

    # Numerical Logs
    gap = train_mean[-1] - test_mean[-1]
    correlation = 0
    if pd.api.types.is_numeric_dtype(feature_vals) and feature_vals.nunique() > 15:
        correlation = np.corrcoef(feature_vals.astype(float), abs_error)[0, 1]
    
    print(f"--- Robustness Diagnostics Summary ---")
    print(f"Generalization Gap (Train-CV) : {gap:.4f}")
    if feature_vals.nunique() > 15:
        print(f"Error Correlation with {critical_feature} : {correlation:.4f}")
    print("-" * 40)
