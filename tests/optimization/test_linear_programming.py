from optimization.linear_programming import linear_programming

#--- Function : test_linear_programming_basic ---
def test_linear_programming_basic():

    solution = linear_programming(
        objective=[3, 5],
        constraints_matrix=[
            [1, 2],
            [3, 2]
        ],
        constraints_values=[
            8,
            12
        ],
        maximize=True
    )

    assert isinstance(solution, dict)
    assert solution["success"] is True
    assert "optimal_values" in solution
    assert "objective_value" in solution


#--- Function : test_linear_programming_minimization ---
def test_linear_programming_minimization():

    solution = linear_programming(
        objective=[2, 3],
        constraints_matrix=[
            [1, 1]
        ],
        constraints_values=[
            10
        ],
        maximize=False
    )

    assert solution["success"] is True
    assert "objective_value" in solution
