import pandas as pd

from decision.mcda import ahp_weights

#--- Function : test_ahp_weights_basic ---
def test_ahp_weights_basic():

    matrix = pd.DataFrame(
        [
            [1, 3, 5],
            [1/3, 1, 3],
            [1/5, 1/3, 1]
        ],
        columns=["Impact", "Feasibility", "Cost"],
        index=["Impact", "Feasibility", "Cost"]
    )

    try:
        ahp_weights(matrix)
        assert True

    except Exception:
        assert False, "AHP weights calculation failed"

#--- Function : test_ahp_weights_non_square_raises ---
def test_ahp_weights_non_square_raises():

    matrix = [
        [1, 2, 3],
        [2, 1, 4]
    ]

    try:
        ahp_weights(matrix)
        assert False, "ValueError should have been raised"

    except ValueError:
        assert True
