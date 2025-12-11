from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#--- Function : evaluate_binary_classifier ---
def evaluate_binary_classifier(model, X_test, y_test):
    """
    Evaluate a binary classifier and return metrics with definitions.
    
    Parameters:
    -----------
    model : trained classifier (must implement predict and predict_proba)
    X_test : test features
    y_test : test target
    
    Returns:
    --------
    results : dict of metrics with values and definitions
    y_pred : predicted labels
    y_prob : predicted probabilities for the positive class
    """
    #Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    #Metrics dictionary
    results = {
        "accuracy": {
            "value": accuracy_score(y_test, y_pred),
            "definition": "Accuracy = proportion of all predictions that are correct (true positives + true negatives) / total samples."
        },
        "precision": {
            "value": precision_score(y_test, y_pred),
            "definition": "Precision = of all predicted positives, the proportion that were actually positive. High precision means few false positives."
        },
        "recall": {
            "value": recall_score(y_test, y_pred),
            "definition": "Recall = of all actual positives, the proportion detected correctly. High recall means few false negatives."
        },
        "f1_score": {
            "value": f1_score(y_test, y_pred),
            "definition": "F1-score = harmonic mean of precision and recall. Balances false positives and false negatives."
        },
        "roc_auc": {
            "value": roc_auc_score(y_test, y_prob),
            "definition": "ROC-AUC = probability that the classifier ranks a random positive higher than a random negative. Measures ranking quality independent of threshold."
        }
    }
    
    return results, y_pred, y_prob
