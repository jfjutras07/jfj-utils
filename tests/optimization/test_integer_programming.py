from optimization.integer_programming import integer_programming

#--- Function : test_integer_programming_basic ---
def test_integer_programming_basic():

    try:
        integer_programming(
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
        assert False, "Integer programming failed"
