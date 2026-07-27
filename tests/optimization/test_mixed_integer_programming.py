from optimization.mixed_integer_programming import mixed_integer_programming

#--- Function : test_mixed_integer_programming_basic ---
def test_mixed_integer_programming_basic():

    objective = [3, 2]

    constraints_matrix = [
        [1, 1]
    ]

    constraints_values = [
        4
    ]

    try:

        result = mixed_integer_programming(
            objective,
            constraints_matrix,
            constraints_values
        )

        assert result["success"] is True

    except Exception:

        assert False, "Mixed integer programming failed"
