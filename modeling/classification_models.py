from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

#---Function:logistic_regression---
def logistic_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Logistic Regression for binary or multiclass classification.
    Standardizes features and optimizes regularization (C) to handle linear decision boundaries.
    """
    # Define the base pipeline
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced'))
    ])
    
    if for_stacking:
        return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    # Testing different regularization strengths (C) and penalty types
    param_grid = {
        'model__C': [0.1, 1.0, 10.0],
        'model__penalty': ['l1', 'l2']
    }
    
    # Optimization using weighted F1-score to balance Precision and Recall
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate probabilities for the AUC-ROC metric if binary classification
    y_prob = None
    if len(np.unique(y_train)) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print(f"--- Logistic Regression Optimized ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score (Weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    if y_prob is not None:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
        
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 35)
    
    return best_model
