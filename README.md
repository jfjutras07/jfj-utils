## jfj-utils

A lightweight and evolving collection of Python utilities designed to support analytics workflows. This library will gradually expand to include tools for data ingestion, cleaning, preprocessing, exploratory data analysis (EDA), visualization, feature engineering, modeling helpers, optimization, and general-purpose utilities.

## Features and Goals

Reusable helper functions for analytics and data science projects

A clean, modular project structure that can scale over time

Easy integration into notebooks and production code

Clear separation by functional area

## Project Structure
```
jfj_utils/
  ingestion/
  preprocessing/
  eda/
  visualization/
  modeling/
  optimization/
  tests/
  utils/
  __init__.py
  pyproject.toml
  setup.py
  README.md
```


Each submodule is intended to contain small, focused utilities related to its theme (e.g., ingestion helpers, visualization wrappers, feature creation functions, etc.).

## Status

Active development — modules will be progressively added and refined.

## API Overview

### Ingestion: Utilities for loading, validating, and structuring raw data from files and folders.

*readers.py*
- `check_data`
- `read_folder`
- `read_table`

### Preprocessing: Functions for cleaning, transforming, and preparing datasets for analysis and modeling.

*missing.py*
- `missing_stats`
- `missing_summary`

*name_cleaning.py*
- `clean_names`
- `clean_names_multiple`

*outliers.py*
- `detect_outliers_iqr`
- `detect_outliers_zscore`
- `winsorize_columns`

*pca.py*
- `perform_pca`

*text_cleaning.py*
- `clean_text`

### EDA: Exploratory data analysis tools for understanding distributions, relationships, and data structure.

*best_transformation.py*
- `best_transformation`
- `best_transformation_for_df`

*check_normality.py*
- `normality_check`
- `numeric_skew_kurt`
- `test_homogeneity`
  
*date_formats.py*
- `detect_date_patterns`

*describe_structure.py*
- `describe_structure`

*multicollinearity.py*
- `check_multicollinearity`

*stats_non_param.py*
- `dunn_friedman_posthoc`
- `dunn_posthoc`
- `friedman_test`
- `games_howell_posthoc`
- `kruskal_wallis_test`
- `mann_whitney_cliff`
- `one_sample_wilcoxon`
- `paired_wilcoxon`
- `permanova_test`
- `quade_test_placeholder`
- `art_test_placeholder`

*stats_param.py*
- `ancova_test`
- `anova_test`
- `f_test_variance`
- `mancova_test`
- `manova_test`
- `one_sample_ttest`
- `paired_ttest`
- `repeated_anova`
- `tukey_posthoc`
- `two_sample_ttest`
- `welch_anova_test`
  
*topic_sentiment_analysis.py*
- `topic_sentiment_analysis`

### Visualization: High-level plotting utilities for exploring data and communicating analytical results.

*choropleth_map.py*

*confusion_matrix.py*
- `plot_logreg_results`

*explore_binary.py*
- `plot_binary_distribution`

*explore_continuous.py*
- `plot_correlation_heatmap`
- `plot_numeric_bivariate`
- `plot_numeric_distribution`
- `plot_pairplot`
- `plot_scatter_grid`
- `plot_violin_grid`
- `qq_plot_numeric`
- `residuals_fitted`
- `scatter_numeric`

*explore_discrete.py*
- `plot_discrete_distribution`
- `plot_discrete_distribution_grid`
- `plot_discrete_dot_distribution`
- `plot_discrete_lollipop_distribution`

*explore_discrete_bivariate.py*
- `plot_discrete_bivariate`
- `plot_discrete_bivariate_grid`
- `plot_discrete_dot_bivariate`
- `plot_discrete_lollipop_bivariate`

*explore_time.py*
- `plot_line_over_time`

*predicted_actual.py*
- `predicted_actual`

*scree_biplot.py*
- `scree_biplot`

*text_exploration.py*
- `text_exploration_basic`

### Modeling: Helpers for fitting, evaluating, and comparing statistical and machine learning models.

*binary_classifier_evaluation.py*
- `evaluate_binary_classifier`

*classification.py*
- `logistic_regression`

*regression.py*
- `cox_regression`
- `gamma_regression`
- `linear_regression`
- `poisson_regression`
- `polynomial_regression`
- `quantile_regression`
- `robust_regression`

*regularization.py*
- `fit_regularized_models`

### Optimization : Prescriptive analytics tools for simulation, decision support, and optimization.

### Utils: Shared low-level utilities used across the library.

*utils.py*
- `log_message`
- `timeit`



