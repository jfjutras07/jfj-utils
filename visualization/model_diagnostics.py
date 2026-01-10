import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

#--- Function : plot_regression_diagnostics ---
def plot_regression_diagnostics(model, X_train, y_train, X_test, y_test, critical_feature, cv=5, 
                                    colors=[UNIFORM_BLUE, PALE_PINK], figsize=(16, 6)):
    """
    Parameters:
    -----------
    model : estimator or Pipeline
        The champion model to evaluate.
    X_train, y_train : Training data for learning curves.
    X_test, y_test : Test data for slice analysis.
    critical_feature : str
        Feature name to analyze for segment bias.
    cv : int
        Folds for learning curves.
    colors : list
        [Main color for training/points, Secondary color for validation/bias].
    """
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
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best')

    # --- Slice Analysis (Segment Bias) ---
    y_pred = model.predict(X_test)
    abs_error = np.abs(y_test - y_pred)
    
    feature_vals = X_test[critical_feature]
    
    # Check if feature is categorical or continuous
    if feature_vals.nunique() > 15:
        sns.scatterplot(x=feature_vals, y=abs_error, alpha=0.5, color=colors[0], ax=axes[1])
        sns.regplot(x=feature_vals, y=abs_error, scatter=False, color=colors[1], ax=axes[1])
    else:
        sns.boxplot(x=feature_vals, y=abs_error, color=colors[0], ax=axes[1])
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right')

    axes[1].set_title(f'Segment Bias: Error by {critical_feature}')
    axes[1].set_xlabel(critical_feature)
    axes[1].set_ylabel('Absolute Error')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Numerical Logs
    gap = train_mean[-1] - test_mean[-1]
    correlation = np.corrcoef(feature_vals.astype(float), abs_error)[0, 1] if feature_vals.nunique() > 15 else 0
    
    print(f"--- Robustness Diagnostics Summary ---")
    print(f"Generalization Gap (Train-CV) : {gap:.4f}")
    print(f"Error Correlation with {critical_feature} : {correlation:.4f}")
    print("-" * 40)
