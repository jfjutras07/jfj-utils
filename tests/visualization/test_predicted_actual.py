import numpy as np
from visualization.explore_continuous import plot_predicted_actual

#--- Function : test_plot_predicted_actual_single_target ---
def test_plot_predicted_actual_single_target():
    y_true = np.random.rand(50)
    y_pred = y_true + np.random.normal(0, 0.1, size=50)

    plot_predicted_actual(
        y_true=y_true,
        y_pred=y_pred,
        model_name="Linear Regression"
    )

    assert True

#--- Function : test_plot_predicted_actual_single_target_with_feature_name ---
def test_plot_predicted_actual_single_target_with_feature_name():
    y_true = np.random.rand(40)
    y_pred = y_true + np.random.normal(0, 0.05, size=40)

    plot_predicted_actual(
        y_true=y_true,
        y_pred=y_pred,
        feature_names=["Income"],
        model_name="Ridge"
    )

    assert True

#--- Function : test_plot_predicted_actual_multi_target ---
def test_plot_predicted_actual_multi_target():
    y_true = np.random.rand(60, 2)
    y_pred = y_true + np.random.normal(0, 0.1, size=(60, 2))

    plot_predicted_actual(
        y_true=y_true,
        y_pred=y_pred,
        model_name="MultiOutputModel"
    )

    assert True

#--- Function : test_plot_predicted_actual_multi_target_with_feature_names ---
def test_plot_predicted_actual_multi_target_with_feature_names():
    y_true = np.random.rand(30, 3)
    y_pred = y_true + np.random.normal(0, 0.05, size=(30, 3))

    plot_predicted_actual(
        y_true=y_true,
        y_pred=y_pred,
        feature_names=["Target A", "Target B", "Target C"],
        model_name="Neural Net"
    )

    assert True

#--- Function : test_plot_predicted_actual_custom_colors ---
def test_plot_predicted_actual_custom_colors():
    y_true = np.random.rand(50)
    y_pred = y_true + np.random.normal(0, 0.1, size=50)

    plot_predicted_actual(
        y_true=y_true,
        y_pred=y_pred,
        colors=["#1f77b4", "#ff7f0e"],
        model_name="XGBoost"
    )

    assert True
