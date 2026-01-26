# --- Function: monte_carlo_portfolio ---
def monte_carlo_portfolio(df_portfolio: pd.DataFrame, n_simulations: int = 10000, budget_limit: float = 1000000, random_seed: int = 42) -> pd.DataFrame:
    """
    Execute a Monte Carlo simulation to assess portfolio-level risk.
    Simulates systemic shocks (inflation, HR stress) and project-level propagation 
    (delays and failures) to estimate budget overruns and delivery success.
    """
    import numpy as np

    np.random.seed(random_seed)
    results = []

    for sim in range(n_simulations):
        # Systemic shocks shared by all projects in the simulation run
        macro_cost_inflation = np.random.normal(loc=1.0, scale=0.10)  # General cost fluctuation
        hr_stress_factor     = np.random.beta(a=2, b=5)              # Labor market friction [0,1]
        execution_climate    = np.random.normal(loc=1.0, scale=0.15)  # Overall delivery efficiency

        total_cost = 0
        delivered_count = 0
        failed_count = 0

        # Project-level propagation based on individual risks and systemic shocks
        for _, proj in df_portfolio.iterrows():
            base_cost = proj['MatBudgetLikely'] + proj['HumBudgetLikely']
            
            # Correlated cost impact
            project_cost = base_cost * macro_cost_inflation

            # Dynamic delay probability (base + stress + specific dependency risk)
            delay_prob = min(0.05 + 0.10 * hr_stress_factor + 0.05 * proj['DependencyRisk'], 0.9)
            delayed = np.random.rand() < delay_prob

            # Failure probability conditional on delays and global execution climate
            failure_prob = (
                0.02 + 
                0.05 * proj['ExecutionRisk'] + 
                (0.10 if delayed else 0) + 
                (0.10 if execution_climate < 0.85 else 0)
            )
            failure_prob = min(failure_prob, 0.95)
            failed = np.random.rand() < failure_prob

            if failed:
                failed_count += 1
                total_cost += project_cost * 0.30  # Sunk costs incurred despite failure
            else:
                delivered_count += 1
                total_cost += project_cost * (1.10 if delayed else 1.0) # Penalty for delays

        # Definition of portfolio-level failure conditions
        portfolio_failed = (
            (total_cost > budget_limit) or 
            (failed_count / len(df_portfolio) > 0.25)
        )

        results.append({
            "TotalCost": total_cost,
            "DeliveredProjects": delivered_count,
            "FailedProjects": failed_count,
            "PortfolioFailed": portfolio_failed
        })

  # --- Function: monte_carlo_portfolio_tracking ---
def monte_carlo_portfolio_tracking(df_portfolio: pd.DataFrame, n_sim: int = 10000, budget_vol: float = 0.15, capacity_vol: float = 0.10, macro_fail_prob: float = 0.05, seed: int = 42) -> tuple:
    """
    Perform a granular Monte Carlo simulation to track failure and delivery rates 
    at both the individual project level and the aggregate portfolio level.
    """
    import numpy as np

    np.random.seed(seed)
    project_ids = df_portfolio['ProjectID'].tolist()
    n_projects = len(project_ids)

    # Pre-calculating project-specific risk components for performance optimization
    exec_risks = (df_portfolio['ExecutionRisk'].values / 5) * 0.25
    dep_risks = (df_portfolio['DependencyRisk'].values / 5) * 0.25
    base_probs = exec_risks + dep_risks

    # Initialize results containers
    failure_matrix = np.zeros((n_sim, n_projects))
    delivered_projects = []

    for sim in range(n_sim):
        # Portfolio-level shocks (correlated variables for the entire run)
        macro_shock = np.random.rand() < macro_fail_prob
        budget_shock = np.random.normal(1, budget_vol)
        capacity_shock = np.random.normal(1, capacity_vol)

        # Build dynamic failure probability for this specific iteration
        # Base (Execution + Dependency) + Environmental shocks
        fail_prob_vector = base_probs.copy()
        if macro_shock:
            fail_prob_vector += 0.20
        if budget_shock > 1.2:
            fail_prob_vector += 0.15
        if capacity_shock > 1.2:
            fail_prob_vector += 0.15

        # Monte Carlo draw for each project in the portfolio
        random_draws = np.random.rand(n_projects)
        failures = (random_draws < fail_prob_vector).astype(int)
        
        failure_matrix[sim, :] = failures
        delivered_projects.append(n_projects - np.sum(failures))

    # Aggregating individual project performance
    failure_df = pd.DataFrame(failure_matrix, columns=project_ids)
    project_failure_rate = failure_df.mean().rename("FailureRate")
    project_delivery_rate = (1 - project_failure_rate).rename("DeliveryRate")

    summary = pd.concat([project_failure_rate, project_delivery_rate], axis=1)
    summary = summary.sort_values("FailureRate", ascending=False)

    # Global portfolio KPIs
    portfolio_stats = {
        "Portfolio Failure Probability (%)": np.mean(np.array(delivered_projects) < n_projects) * 100,
        "Average Projects Delivered": np.mean(delivered_projects),
        "P10 Delivered Projects": np.percentile(delivered_projects, 10),
        "P90 Delivered Projects": np.percentile(delivered_projects, 90),
    }

    return summary, portfolio_stats

    return pd.DataFrame(results)
