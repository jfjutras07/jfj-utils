import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report, f1_score

#--- Function : plot_classification_diagnostics ---
def plot_classification_diagnostics(model, X_train, y_train, X_test, y_test, cv=5, 
                                   colors=[UNIFORM_BLUE, PALE_PINK], figsize=(16, 6)):
    """
    Displays a 1x2 classification dashboard.
    Left: Learning curves to detect Overfitting/Underfitting.
    Right: Confusion matrix to identify specific class misclassifications.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Learning Curves (Generalization) ---
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='f1_macro', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    axes[0].plot(train_sizes, train_mean, label='Training Score (F1)', color=colors[0], lw=2)
    axes[0].plot(train_sizes, test_mean, label='Validation Score (CV)', color=colors[1], linestyle='--', lw=2)
    axes[0].set_title('Learning Curves: Model Capacity')
    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('F1 Macro Score')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best')

    # --- Confusion Matrix (Error Analysis) ---
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title('Confusion Matrix: Error Distribution')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

    # Numerical Logs
    gap = train_mean[-1] - test_mean[-1]
    final_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"--- Classification Diagnostics Summary ---")
    print(f"Generalization Gap (Train-CV) : {gap:.4f}")
    print(f"Final F1 Macro (Test)        : {final_f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 40)

    return {"gap": gap, "f1_test": final_f1}
