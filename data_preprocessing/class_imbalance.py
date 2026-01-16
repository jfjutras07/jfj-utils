import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

#--- Function: class_imbalance_correction ---
def class_imbalance_correction(df, target_col, classifier, test_size=0.2, random_state=42):
    """
    Analyze class distribution, apply the most suitable resampling strategy 
    to the training set to prevent leakage, and evaluate the model.

    Example:
    --------
    results = class_imbalance_correction(df, target_col='Attrition', classifier=RandomForestClassifier())

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing features and the target column.
    target_col : str
        Name of the target variable.
    classifier : estimator object
        A scikit-learn compatible classifier.
    test_size : float, default 0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default 42
        Seed for reproducibility.

    Returns:
    --------
    result_dict : dict
        Dictionary containing:
        - 'pipeline': The trained imblearn Pipeline
        - 'strategy_used': The name of the resampling method applied
        - 'balanced_accuracy': Balanced accuracy score on test set
        - 'classification_report': Detailed metrics report
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    counts = y_train.value_counts(normalize=True)
    minority_ratio = counts.min()
    n_samples = len(y_train)
    
    strategy_name = "None (Balanced enough)"
    resampler = None

    if n_samples > 500000:
        strategy_name = "Random Under-Sampling (Big Data)"
        resampler = RandomUnderSampler(random_state=random_state)
    elif minority_ratio < 0.05:
        strategy_name = "SMOTE + Tomek Links (Severe Imbalance)"
        resampler = SMOTETomek(random_state=random_state)
    elif minority_ratio < 0.25:
        strategy_name = "Standard SMOTE (Moderate Imbalance)"
        resampler = SMOTE(random_state=random_state)

    steps = []
    if resampler:
        steps.append(('resampler', resampler))
    steps.append(('classifier', classifier))
    
    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"--- Class Imbalance Report ---")
    print(f"Minority Ratio in Train Set: {minority_ratio:.2%}")
    print(f"Resampling Strategy: {strategy_name}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\nClassification Report:")
    print(report)

    return {
        'pipeline': pipeline,
        'strategy_used': strategy_name,
        'balanced_accuracy': bal_acc,
        'classification_report': report
    }
