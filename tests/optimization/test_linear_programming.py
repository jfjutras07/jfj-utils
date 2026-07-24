from optimization.linear_programming import linear_programming

#--- Function : test_linear_programming_basic ---
def test_linear_programming_basic():

    try:
        linear_programming(
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

        assert True

    except Exception:
        assert False, "Linear programming failed"
