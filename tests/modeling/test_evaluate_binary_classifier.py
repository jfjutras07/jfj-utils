import numpy as np
from eda.classification import evaluate_binary_classifier

# --- Dummy model to control predictions ---
class DummyBinaryClassifier:
    def __init__(self, y_pred, y_prob):
        self._y_pred = np.array(y_pred)
        self._y_prob = np.array(y_prob)

    def predict(self, X):
        return self._y_pred

    def predict_proba(self, X):
        return np.column_stack([1 - self._y_prob, self._y_prob])

#--- Function : test_evaluate_binary_classifier_basic ---
def test_evaluate_binary_classifier_basic():
    X_test = np.zeros((4, 2))
    y_test = np.array([0, 1, 0, 1])

    model = DummyBinaryClassifier(
        y_pred=[0, 1, 0, 1],
        y_prob=[0.1, 0.9, 0.2, 0.8],
    )

    results, y_pred, y_prob = evaluate_binary_classifier(model, X_test, y_test)

    assert isinstance(results, dict)
    assert np.array_equal(y_pred, y_test)
    assert np.allclose(y_prob, [0.1, 0.9, 0.2, 0.8])

#--- Function : test_evaluate_binary_classifier_perfect_scores ---
def test_evaluate_binary_classifier_perfect_scores():
    X_test = np.zeros((6, 2))
    y_test = np.array([0, 1, 0, 1, 0, 1])

    model = DummyBinaryClassifier(
        y_pred=[0, 1, 0, 1, 0, 1],
        y_prob=[0.01, 0.99, 0.02, 0.98, 0.05, 0.95],
    )

    results, _, _ = evaluate_binary_classifier(model, X_test, y_test)

    assert results["accuracy"]["value"] == 1.0
    assert results["precision"]["value"] == 1.0
    assert results["recall"]["value"] == 1.0
    assert results["f1_score"]["value"] == 1.0
    assert results["roc_auc"]["value"] == 1.0

#--- Function : test_evaluate_binary_classifier_all_wrong ---
def test_evaluate_binary_classifier_all_wrong():
    X_test = np.zeros((4, 2))
    y_test = np.array([0, 1, 0, 1])

    model = DummyBinaryClassifier(
        y_pred=[1, 0, 1, 0],
        y_prob=[0.9, 0.1, 0.8, 0.2],
    )

    results, _, _ = evaluate_binary_classifier(model, X_test, y_test)

    assert results["accuracy"]["value"] == 0.0
    assert results["recall"]["value"] == 0.0
    assert results["roc_auc"]["value"] == 0.0

#--- Function : test_evaluate_binary_classifier_random_auc ---
def test_evaluate_binary_classifier_random_auc():
    X_test = np.zeros((6, 2))
    y_test = np.array([0, 1, 0, 1, 0, 1])

    model = DummyBinaryClassifier(
        y_pred=[0, 0, 1, 1, 0, 1],
        y_prob=[0.4, 0.6, 0.6, 0.4, 0.5, 0.5],
    )

    results, _, _ = evaluate_binary_classifier(model, X_test, y_test)

    assert abs(results["roc_auc"]["value"] - 0.5) < 0.1

#--- Function : test_evaluate_binary_classifier_output_structure ---
def test_evaluate_binary_classifier_output_structure():
    X_test = np.zeros((2, 2))
    y_test = np.array([0, 1])

    model = DummyBinaryClassifier(
        y_pred=[0, 1],
        y_prob=[0.3, 0.7],
    )

    results, _, _ = evaluate_binary_classifier(model, X_test, y_test)

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }

    assert set(results.keys()) == expected_metrics

    for metric in expected_metrics:
        assert "value" in results[metric]
        assert "definition" in results[metric]
        assert isinstance(results[metric]["value"], float)
        assert isinstance(results[metric]["definition"], str)
