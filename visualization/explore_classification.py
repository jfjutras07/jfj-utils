import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from .style import UNIFORM_BLUE

#---Function: plot_classification_diagnostics---
def plot_classification_diagnostics(model, X_test, y_test, feature_names=None):
    """
    Visualize a logistic regression classifier:
    - Confusion matrix
    - Top coefficients by magnitude
    
    Parameters:
    -----------
    model : trained LogisticRegression
    X_test : test features
    y_test : test target
    feature_names : list of feature names, optional
    """
    #Predictions
    y_pred = model.predict(X_test)
    
    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=UNIFORM_BLUE, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.tight_layout()
    plt.show()
    
    #Coefficients
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(model.coef_.shape[1])]
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    
    print("Top 20 features by coefficient value:\n")
    print(coef_df.head(20))
