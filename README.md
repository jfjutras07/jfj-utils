## jfj-utils

An evolving collection of Python utilities designed to support analytics workflows. This library will gradually expand to include tools for data ingestion, cleaning, preprocessing, exploratory data analysis (EDA), visualization, feature engineering, modeling helpers, optimization, and general-purpose utilities.

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

*class_imbalance.py*
- `class_imbalance_correction`

*cleaning.py*
- Class : `column_dropper`
- `clean_names`
- `clean_names_multiple`
- `clean_text`

*dimensionality_reduction.py*
- `perform_famd`
- `perform_mca`
- `perform_pca`
  
*encoding.py*
- Class : `categorical_encoder`-
- `binary_encode_columns`
- `label_encode_columns`
- `one_hot_encode_columns`
- `ordinal_encode_columns`
  
*feature_engineering.py*
- Class : `ratio_generator`-
- `mi_classification`
- `mi_regression`
  
*missing.py*
- Class : `group_imputer`-
- Class: `logical_imputer`
- `missing_stats`
- `missing_summary`

*normalizing.py*
- Class : `skewness_corrector`-
- `normalize_columns`
  
*outliers.py*
- `detect_outliers_iqr`
- `detect_outliers_zscore`
- `winsorize_columns`

*scaling.py*
- Class : `feature_scaler`-
- `minmax_scaler`
- `robust_scaler`
- `standard_scaler`

### EDA: Exploratory data analysis tools for understanding distributions, relationships, and data structure.

*best_transformation.py*
- `best_transformation`
- `best_transformation_for_df`

*check_normality.py*
- `homogeneity_check`
- `normality_check`
- `skew_kurt_check`

*chi_square_test.py*
- `chi_square_test`

*classification.py*
- `logistic_regression`

*date_formats.py*
- `detect_date_patterns`

*describe_structure.py*
- `describe_structure`

*explainability.py*
- `feature_importance`
- `interaction_effects`
- `lime_analysis`
- `pdp_analysis`
- `permutation_importance_calc`
- `shap_analysis`

*multicollinearity.py*
- `correlation_check`
- `VIF_check`

*premodeling_check.py*
- `premodeling_classification_check`
- `premodeling_clustering_check`
- `premodeling_regresion_check`

*regression.py*
- `cox_regression`
- `gamma_regression`
- `linear_mixed_model`
- `linear_regression`
- `negative_binomial_regression`
- `poisson_regression`
- `polynomial_regression`
- `quantile_regression`
- `robust_regression`

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

*advanced_models.py*
- `bayesian_classification`
- `bayesian_regression`
- `gaussian_process_classification`
- `gaussian_process_regression`
- `knn_classification`
- `knn_regression`
- `mlp_regression`
- `mlp_classification`
- `svm_classification`
- `svm_regression`
  
*classification_models.py*
- `logistic_regression`

*classification_trees.py*
- `catboost_classification`
- `compare_classification_tree_models`
- `decision_tree_classification`
- `lightgbm_classification`
- `random_forest_classification`
- `xgboost_classification`
  
*clustering_models.py*
- `agglomerative_clustering`
- `birch_clustering`
- `compare_clustering_models`
- `dbscan_clustering`
- `gaussian_mixture_clustering`
- `kmeans_clustering`
- `kmedoids_clustering`

*model_stability.py*
- `check_classification_model_stability`
- `check_clustering_model_stability`
- `check_regression_model_stability`
  
*regression_models.py*
- `linear_regression`
- `polynomial_regression`
- `quantile_regression`
- `robust_regression`

*regression_trees.py*
- `catboost_regression`
- `compare_regression_tree_models`
- `decision_tree_regression`
- `lightgbm_regression`
- `random_forest_regression`
- `xgboost_regression`

*regularization.py*
- `compare_regularized_models`
- `elasticnet_regression`
- `lasso_regression`
- `ridge_regression`

*stacking.py*
- `stacking_classification_ensemble`
- `stacking_regression_ensemble`
  
### Optimization : Prescriptive analytics tools for simulation, decision support, and optimization.

### Visualization: High-level plotting utilities for exploring data and communicating analytical results.

*chi_square_heatmap.py*
- `chi_square_heatmap`
  
*choropleth_map.py*
- `choropleth_map`

*explore_binary.py*
- `plot_binary_distribution`

*explore_classification.py*
- `plot_classification_diagnostics`

*explore_clusters.py*
- `plot_cluster_diagnostics`
- `plot_cluster_projections`
- `plot_cluster_radar_charts`

*explore_continuous.py*
- `plot_box_grid`
- `plot_correlation_heatmap`
- `plot_heatmap_grid`
- `plot_mi_vs_correlation`
- `plot_numeric_bivariate`
- `plot_numeric_distribution`
- `plot_pairplot`
- `plot_scatter_grid`
- `plot_swarm_grid`
- `plot_violin_grid`

*explore_discrete.py*
- `plot_discrete_distribution`
- `plot_discrete_distribution_grid`
- `plot_discrete_dot_distribution`
- `plot_discrete_lollipop_distribution`

*explore_discrete_multivariate.py*
- `plot_discrete_bivariate_grid`
- `plot_discrete_dot_bivariate`
- `plot_discrete_lollipop_bivariate`

*explore_regression.py*
- `plot_regression_diagnostics`

*explore_time.py*
- `plot_line_grid_over_time`
- `plot_temporal_data`

*predicted_actual.py*
- `plot_predicted_actual`

*scree_biplot.py*
- `plot_scree_biplot`

*text_exploration.py*
- `text_exploration_basic`

### Utils: Shared low-level utilities used across the library.

*utils.py*
- Class: `generic_transformer`
- `log_message`
- `timeit`



