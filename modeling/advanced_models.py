import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from IPython.display import display

# Models
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Metrics
from sklearn.metrics import accuracy_score, f1_score, r2_score

#---Function:bayesian_classification---
def bayesian_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Logistic Regression with L2 penalty as a Bayesian point estimate.
    """
    # Merging default params with user provided params
    params = {'solver': 'saga', 'max_iter': 1000, 'class_weight': 'balanced'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(**params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__C': [0.1, 1.0, 10.0], 'model__penalty': ['l2']}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Bayesian Classification Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model

#---Function:bayesian_regression---
def bayesian_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Bayesian Ridge Regression with automated tuning.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', BayesianRidge(**model_params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__alpha_1': [1e-6, 1e-5], 'model__lambda_1': [1e-6, 1e-5]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Bayesian Regression Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)
    return best_model

#---Function:gaussian_process_classification---
def gaussian_process_classification(train_df, test_df, outcome, predictors, cv=3, for_stacking=False, **model_params):
    """
    Gaussian Process Classifier (GPC) with RBF kernel.
    """
    kernel = 1.0 * RBF(1.0)
    params = {'kernel': kernel, 'random_state': 42}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GaussianProcessClassifier(**params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__max_iter_predict': [100, 200]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Gaussian Process Classification Summary ---")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model

#---Function:gaussian_process_regression---
def gaussian_process_regression(train_df, test_df, outcome, predictors, cv=3, for_stacking=False, **model_params):
    """
    Gaussian Process Regressor (GPR) with noise estimation.
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1)
    params = {'kernel': kernel, 'n_restarts_optimizer': 10, 'random_state': 42}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GaussianProcessRegressor(**params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__alpha': [1e-10, 1e-2]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Gaussian Process Regression Summary ---")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)
    return best_model

#---Function:knn_classification---
def knn_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    K-Nearest Neighbors Classifier with weight='distance' to help with imbalance.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(**model_params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_neighbors': [3, 5, 11], 'model__weights': ['distance']}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- KNN Classification Summary ---")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    
    perm_imp = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42)
    display(pd.DataFrame({'Feature': predictors, 'Importance': perm_imp.importances_mean}).sort_values('Importance', ascending=False).head(5))
    print("-" * 35)
    return best_model

#---Function:knn_regression---
def knn_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    K-Nearest Neighbors Regressor with permutation importance.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor(**model_params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_neighbors': [3, 5, 11], 'model__weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- KNN Regression Summary ---")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)
    return best_model

#---Function:mlp_classification---
def mlp_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Multi-layer Perceptron Classifier.
    """
    params = {'max_iter': 1000, 'random_state': 42, 'early_stopping': True}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(**params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__hidden_layer_sizes': [(50,), (100, 50)], 'model__activation': ['relu', 'tanh']}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- MLP Classification Summary ---")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model

#---Function:mlp_regression---
def mlp_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Multi-layer Perceptron Regressor.
    """
    params = {'max_iter': 1000, 'random_state': 42, 'early_stopping': True}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(**params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__hidden_layer_sizes': [(50,), (100, 50)], 'model__activation': ['relu', 'tanh']}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- MLP Regression Summary ---")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)
    return best_model

#---Function:svm_classification---
def svm_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Support Vector Classification (SVC) with class_weight='balanced'.
    """
    params = {'probability': True, 'class_weight': 'balanced'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVC(**params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__kernel': ['rbf', 'poly'], 'model__C': [0.1, 1, 10]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- SVM Classification Summary ---")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    
    perm_imp = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42)
    display(pd.DataFrame({'Feature': predictors, 'Importance': perm_imp.importances_mean}).sort_values('Importance', ascending=False).head(5))
    print("-" * 35)
    return best_model

#---Function:svm_regression---
def svm_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Support Vector Regression (SVR) with permutation importance.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVR(**model_params))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__kernel': ['rbf', 'poly'], 'model__C': [0.1, 1, 10]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- SVM Regression Summary ---")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    
    perm_imp = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42)
    display(pd.DataFrame({'Feature': predictors, 'Importance': perm_imp.importances_mean}).sort_values('Importance', ascending=False).head(5))
    print("-" * 35)
    return best_model
