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
  data_preprocessing/
  eda/
  ingestion/
  modeling/
  optimization/
  tests/
  visualization
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

### Data_preprocessing: Functions for cleaning, transforming, and preparing datasets for analysis and modeling.

*encoding.py*
- `binary_encode_columns`
- `label_encode_columns`
- `one_hot_encode_columns`
- `ordinal_encode_columns`
  
*feature_engineering.py*
- `mi_classification`
- `mi_regression`
  
*missing.py*
- `missing_stats`
- `missing_summary`

*name_cleaning.py*
- `clean_names`
- `clean_names_multiple`

*normalizing.py*
- `normalize_columns`
  
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

*calculate_vif.py*
- `calculate_vif`

*check_normality.py*
- `homogeneity_check`
- `normality_check`
- `skew_kurt_check`

*chi_square_test.py*
- `chi_square_test`

*date_formats.py*
- `detect_date_patterns`

*describe_structure.py*
- `describe_structure`

*multicollinearity.py*
- `correlation_check`
- `VIF_check`

*premodeling_check.py*
- `premodeling_regresion_check`

*stats_diagnostics.py*
- `stats_diagnostics`

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
- `multi_factor_anova`
- `one_sample_ttest`
- `paired_ttest`
- `repeated_anova`
- `robust_anova`
- `tukey_posthoc`
- `two_sample_ttest`
- `welch_anova_test`
  
*topic_sentiment_analysis.py*
- `topic_sentiment_analysis`

### Ingestion: Utilities for loading, validating, and structuring raw data from files and folders.

*readers.py*
- `check_data`
- `read_folder`
- `read_table`

### Modeling: Helpers for fitting, evaluating, and comparing statistical and machine learning models.

*classification.py*
- `logistic_regression`

*regression.py*
- `cox_regression`
- `gamma_regression`
- `linear_mixed_model`
- `linear_regression`
- `poisson_regression`
- `polynomial_regression`
- `quantile_regression`
- `robust_regression`

*regularization.py*
- `compare_regularized_models`
- `elasticnet_regression`
- `lasso_regression`
- `ridge_regression`
 
### Optimization : Prescriptive analytics tools for simulation, decision support, and optimization.

### Visualization: High-level plotting utilities for exploring data and communicating analytical results.

*chi_square_heatmap.py*
- `chi_square_heatmap`
  
*choropleth_map.py*
- `choropleth_map`

*confusion_matrix.py*
- `plot_logreg_results`

*explore_binary.py*
- `plot_binary_distribution`

*explore_continuous.py*
- `plot_box_grid`
- `plot_box_plot`
- `plot_correlation_heatmap`
- `plot_heatmap_grid`
- `plot_numeric_bivariate`
- `plot_numeric_distribution`
- `plot_pairplot`
- `plot_scatter_grid`
- `plot_scatter_plot`
- `plot_swarm_grid`
- `plot_violin_grid`

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
- `plot_line_grid_over_time`

*predicted_actual.py*
- `plot_predicted_actual`

*scree_biplot.py*
- `plot_scree_biplot`

*text_exploration.py*
- `text_exploration_basic`

### Utils: Shared low-level utilities used across the library.

*utils.py*
- `log_message`
- `timeit`



