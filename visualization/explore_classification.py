import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc

# Importing centralized style constants
from .style import UNIFORM_BLUE, PALE_PINK, SEQUENTIAL_CMAP, WHITE, GREY_DARK, DEFAULT_FIGSIZE

# --- Function : plot_classification_diagnostics ---
def plot_classification_diagnostics(model, X_train, y_train, X_test, y_test, cv=5, 
                                   colors=None, figsize=(20, 6)):
    """
    Displays a classification dashboard with three key visualizations:
    Learning Curves, Confusion Matrix, and ROC Curve.
    """
    if colors is None:
        colors = ['#4A90E2', '#FF6B6B'] # Uniform Blue and Pale Pink
        
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Learning Curves (Generalization Analysis) ---
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='f1_weighted', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    axes[0].plot(train_sizes, train_mean, label='Training F1', color=colors[0], lw=2)
    axes[0].plot(train_sizes, test_mean, label='Validation F1 (CV)', color=colors[1], linestyle='--', lw=2)
    axes[0].set_title('Learning Curves: Model Capacity')
    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('F1 Weighted Score')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.3)

    # --- Confusion Matrix (Classification Bias) ---
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_xticklabels(['Stay', 'Leave'])
    axes[1].set_yticklabels(['Stay', 'Leave'])

    # --- ROC Curve (Performance Balance) ---
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate (Recall)')
    axes[2].set_title('Receiver Operating Characteristic (ROC)')
    axes[2].legend(loc="lower right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Numerical Logs
    gap = train_mean[-1] - test_mean[-1]
                                     
    print(f"--- Classification Diagnostics Summary ---")
    print(f"Generalization Gap (Train-CV F1) : {gap:.4f}")
    print(f"Final Test ROC-AUC : {roc_auc:.4f}")
    print("-" * 42)

#---Function: plot_feature_importance---
def plot_feature_importance(model, predictors, title=None, figsize=None):
    """
    Generates a directional bar chart of model coefficients.
    Supports both standalone models and Scikit-Learn Pipelines.
    Filters out features with zero coefficients (Lasso).
    """
    
    # Extraction of coefficients (handling Pipeline or raw model)
    if hasattr(model, 'named_steps'):
        coefs = model.named_steps['model'].coef_[0]
    else:
        coefs = model.coef_[0]
        
    # Prepare and sort data
    importance_df = pd.DataFrame({'Feature': predictors, 'Coef': coefs})
    # Keeping only non-zero coefficients
    importance_df = importance_df[importance_df['Coef'] != 0].sort_values(by='Coef', ascending=True)

    if importance_df.empty:
        print("No non-zero coefficients to plot.")
        return

    # Setup plot
    plot_figsize = figsize or (10, len(importance_df) * 0.4)
    fig, ax = plt.subplots(figsize=plot_figsize)

    # Apply colors: PALE_PINK for positive risk, UNIFORM_BLUE for negative retention
    bar_colors = [PALE_PINK if x > 0 else UNIFORM_BLUE for x in importance_df['Coef']]

    # Plotting
    bars = ax.barh(importance_df['Feature'], importance_df['Coef'], color=bar_colors)

    # Styling
    title_text = title or 'Drivers of Attrition: Risk (+) vs. Retention (-)'
    ax.set_title(title_text, fontsize=14, fontweight='bold', color=GREY_DARK, pad=20)
    ax.axvline(x=0, color=GREY_DARK, linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Coefficient Magnitude (Log-Odds)', color=GREY_DARK)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add grid for readability
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return importance_df
