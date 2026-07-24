from optimization.integer_programming import integer_programming

#--- Function : test_integer_programming_basic ---
def test_integer_programming_basic():

    solution = integer_programming(
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


#--- Function : test_integer_programming_returns_integer_solution ---
def test_integer_programming_returns_integer_solution():

    solution = integer_programming(
        objective=[10, 15],
        constraints_matrix=[
            [2, 3]
        ],
        constraints_values=[
            10
        ],
        maximize=True
    )

    values = solution["optimal_values"]

    assert all(float(x).is_integer() for x in values)
