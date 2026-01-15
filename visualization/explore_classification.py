import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Importing centralized style constants
from .style import UNIFORM_BLUE, PALE_PINK, SEQUENTIAL_CMAP, WHITE, GREY_DARK, DEFAULT_FIGSIZE

#--- Function : plot_classification_diagnostics ---
def plot_classification_diagnostics(model, X_train, y_train, X_test, y_test, cv=5, 
                                    figsize=(16, 6)):
    """
    Displays a 1x2 classification dashboard.
    Left: Learning curves to detect Overfitting/Underfitting using F1 Macro.
    Right: Confusion matrix to identify specific class misclassifications.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Learning Curves (Generalization Analysis) ---
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='f1_macro', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Training score with solid line
    axes[0].plot(train_sizes, train_mean, label='Training Score (F1)', color=UNIFORM_BLUE, lw=2.5)
    # Validation score with dashed line for distinction
    axes[0].plot(train_sizes, test_mean, label='Validation Score (CV)', color=PALE_PINK, linestyle='--', lw=2.5)
    
    axes[0].set_title('Learning Curves: Model Capacity', fontweight='bold', pad=15)
    axes[0].set_xlabel('Training Set Size', color=GREY_DARK)
    axes[0].set_ylabel('F1 Macro Score', color=GREY_DARK)
    axes[0].legend(loc='lower right', frameon=True)

    # --- Confusion Matrix (Error Analysis) ---
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=SEQUENTIAL_CMAP, 
        ax=axes[1], 
        cbar=False,
        linewidths=1,
        linecolor=WHITE,
        annot_kws={"weight": "bold"}
    )
    
    axes[1].set_title('Confusion Matrix: Error Distribution', fontweight='bold', pad=15)
    axes[1].set_xlabel('Predicted Label', color=GREY_DARK)
    axes[1].set_ylabel('True Label', color=GREY_DARK)

    plt.tight_layout()
    plt.show()

    # Numerical Diagnostics Logs
    gap = train_mean[-1] - test_mean[-1]
    final_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n" + "="*45)
    print(f"   CLASSIFICATION DIAGNOSTICS SUMMARY")
    print(f"="*45)
    print(f"Generalization Gap (Train-CV) : {gap:.4f}")
    print(f"Final F1 Macro (Test)         : {final_f1:.4f}")
    print(f"Detailed Report:\n")
    print(classification_report(y_test, y_pred))
    print("="*45 + "\n")

    return {"gap": gap, "f1_test": final_f1}
