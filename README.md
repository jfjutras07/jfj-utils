## jfj-utils

A lightweight and evolving collection of Python utilities designed to support analytics workflows. This library will gradually expand to include tools for data ingestion, cleaning, preprocessing, exploratory data analysis (EDA), visualization, feature engineering, modeling helpers, and general-purpose utilities.

## Features and Goals

Reusable helper functions for analytics and data science projects

A clean, modular project structure that can scale over time

Easy integration into notebooks and production code

Clear separation by functional area

## Project Structure
```
jfj_utils/
  ingestion/
    readers.py
        check_data
        read_folder
        read_table
  preprocessing/
    missing.py
        missing_stats
        missing_summary
    name_cleaning.py
        clean_names
        clean_names_multiple
    outliers.py
        detect_outliers_iqr
        detect_outliers_zscore
        winsorize_columns
    pca.py
        perform_pca
    text_cleaning.py
        clean_text
  eda/
    best_transformation.py
        best_transformation
        best_transformation_for_df
    check_normality.py
        normality_check
        numeric_skew_kurt
    date_formats.py
        detect_date_patterns
    describe_structure.py
        describe_structure
    multicollinearity.py
        check_multicollinearity
    stats_non_param.py
        mann_whitney_cliff
        robust_ancova
    topic_sentiment_analysis.py
        topic_sentiment_analysis
  visualization/
    confusion_matrix.py
        plot_logreg_results
    explore_binary.py
        plot_binary_distribution
    explore_continuous
        plot_correlation_heatmap
        plot_numeric_distribution
        plot_pairplot
        plot_scatter_grid
        plot_violin_grid
        qq_plot_numeric
        residuals_fitted
        scatter_numeric
    explore_discrete.py
        plot_discrete_distribution
    explore_time.py
        plot_line_over_time
    predicted_actual.py
        predicted_actual
    scree_biplot.py
        scree_biplot
    text_exploration.py
        text_exploration_basic
  modeling/
    binary_classifier_evaluation.py
        evaluate_binary_classifier
    regularization.py
        fit_regularized_models
  tests/
  utils/
    utils.py
        log_message
        timeit
  __init__.py
  pyproject.toml
  setup.py
  README.md
```


Each submodule is intended to contain small, focused utilities related to its theme (e.g., ingestion helpers, visualization wrappers, feature creation functions, etc.).

## Status

Active development — modules will be progressively added and refined.
