import pandas as pd
import numpy as np

# --- Function: monte_carlo_portfolio ---
def monte_carlo_portfolio(df_portfolio: pd.DataFrame, n_simulations: int = 10000, budget_limit: float = 1000000, random_seed: int = 42) -> None:
    """
    Execute a Monte Carlo simulation at the portfolio level and print results.
    """
    import numpy as np

    np.random.seed(random_seed)
    results = []

    for sim in range(n_simulations):
        macro_cost_inflation = np.random.normal(loc=1.0, scale=0.10)
        hr_stress_factor     = np.random.beta(a=2, b=5)
        execution_climate    = np.random.normal(loc=1.0, scale=0.15)

        total_cost = 0
        delivered_count = 0
        failed_count = 0

        for _, proj in df_portfolio.iterrows():
            base_cost = proj['MatBudgetLikely'] + proj['HumBudgetLikely']
            project_cost = base_cost * macro_cost_inflation
            delay_prob = min(0.05 + 0.10 * hr_stress_factor + 0.05 * proj['DependencyRisk'], 0.9)
            delayed = np.random.rand() < delay_prob

            failure_prob = (0.02 + 0.05 * proj['ExecutionRisk'] + 
                            (0.10 if delayed else 0) + 
                            (0.10 if execution_climate < 0.85 else 0))
            
            failed = np.random.rand() < min(failure_prob, 0.95)

            if failed:
                failed_count += 1
                total_cost += project_cost * 0.30
            else:
                delivered_count += 1
                total_cost += project_cost * (1.10 if delayed else 1.0)

        portfolio_failed = (total_cost > budget_limit) or (failed_count / len(df_portfolio) > 0.25)
        results.append({
            "TotalCost": total_cost,
            "DeliveredProjects": delivered_count,
            "PortfolioFailed": portfolio_failed
        })

    df_res = pd.DataFrame(results)

    print("=========== MONTE CARLO PORTFOLIO RESULTS ===========")
    print(f"Probability of Portfolio Failure : {df_res['PortfolioFailed'].mean():.1%}")
    print(f"Average Projects Delivered       : {df_res['DeliveredProjects'].mean():.1f}")
    print(f"P10 Delivered Projects           : {df_res['DeliveredProjects'].quantile(0.10):.0f}")
    print(f"P90 Delivered Projects           : {df_res['DeliveredProjects'].quantile(0.90):.0f}")

# --- Function: monte_carlo_portfolio_tracking ---
def monte_carlo_portfolio_tracking(df_portfolio: pd.DataFrame, n_sim: int = 10000, budget_vol: float = 0.15, capacity_vol: float = 0.10, macro_fail_prob: float = 0.05, seed: int = 42) -> None:
    """
    Track individual project risks and portfolio KPIs via Monte Carlo simulation.
    """
    import numpy as np

    np.random.seed(seed)
    project_ids = df_portfolio['ProjectID'].tolist()
    n_projects = len(project_ids)

    exec_risks = (df_portfolio['ExecutionRisk'].values / 5) * 0.25
    dep_risks = (df_portfolio['DependencyRisk'].values / 5) * 0.25
    base_probs = exec_risks + dep_risks

    failure_matrix = np.zeros((n_sim, n_projects))
    delivered_projects = []

    for sim in range(n_sim):
        macro_shock = np.random.rand() < macro_fail_prob
        budget_shock = np.random.normal(1, budget_vol)
        capacity_shock = np.random.normal(1, capacity_vol)

        fail_prob_vector = base_probs.copy()
        if macro_shock: fail_prob_vector += 0.20
        if budget_shock > 1.2: fail_prob_vector += 0.15
        if capacity_shock > 1.2: fail_prob_vector += 0.15

        failures = (np.random.rand(n_projects) < fail_prob_vector).astype(int)
        failure_matrix[sim, :] = failures
        delivered_projects.append(n_projects - np.sum(failures))

    failure_df = pd.DataFrame(failure_matrix, columns=project_ids)
    summary = pd.DataFrame({
        "FailureRate": failure_df.mean(),
        "DeliveryRate": 1 - failure_df.mean()
    }).sort_values("FailureRate", ascending=False)

    print(summary.head(10))
    print("\n=========== MONTE CARLO PORTFOLIO RESULTS ===========")
    print(f"Portfolio Failure Probability (%)  : {np.mean(np.array(delivered_projects) < n_projects) * 100:.2f}")
    print(f"Average Projects Delivered         : {np.mean(delivered_projects):.2f}")
    print(f"P10 Delivered Projects             : {np.percentile(delivered_projects, 10):.2f}")
    print(f"P90 Delivered Projects             : {np.percentile(delivered_projects, 90):.2f}")
