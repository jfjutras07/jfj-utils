"""
Centralized visualization style definitions.
Graphical roles only – no semantic assumptions.
Compatible with all EDA and visualization functions.
"""

#======================
# Core colors
#======================

PRIMARY_COLOR = "#1f77b4"      # professional balanced blue
SECONDARY_COLOR = "#f4a3c4"    # soft contrast (for bivariate categorical)

#======================
# Usage roles
#======================

UNIVARIATE_COLOR = PRIMARY_COLOR           # single variable plots
MULTIVARIATE_COLOR = PRIMARY_COLOR         # multivariate plots
BIVARIATE_PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR]  # bivariate plots
UNIFORM_BLUE = PRIMARY_COLOR               # legacy name used in some functions

#======================
# Colormaps
#======================

SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"

#======================
# Defaults
#======================

DEFAULT_ALPHA = 0.8
DEFAULT_EDGE_COLOR = "white"
