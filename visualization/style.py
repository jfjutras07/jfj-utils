"""
Centralized visualization style definitions.
Graphical roles only – no semantic assumptions.

"""

# ======================
# Core colors
# ======================

PRIMARY_COLOR = "#1f77b4"      # professional balanced blue
SECONDARY_COLOR = "#f4a3c4"    # soft contrast (only for bivariate categorical)

# ======================
# Usage rules
# ======================

UNIVARIATE_COLOR = PRIMARY_COLOR
MULTIVARIATE_COLOR = PRIMARY_COLOR
BIVARIATE_PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR]

# ======================
# Colormaps
# ======================

SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"

# ======================
# Defaults
# ======================

DEFAULT_ALPHA = 0.8
DEFAULT_EDGE_COLOR = "white"
