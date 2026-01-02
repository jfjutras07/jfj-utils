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
PALE_PINK = "#f4c2c2"          # used in some continuous plots
PALE_GREEN = "#90EE90"         # light green
PALE_BLUE = "#ADD8E6"          # light blue
UNIFORM_BLUE = PRIMARY_COLOR   # legacy name used in some functions

#======================
# Usage roles
#======================

UNIVARIATE_COLOR = PRIMARY_COLOR           # single variable plots
MULTIVARIATE_COLOR = PRIMARY_COLOR         # multivariate plots
BIVARIATE_PALETTE = [PRIMARY_COLOR, SECONDARY_COLOR]  # bivariate plots
DOT_PLOT_PALETTE = [PALE_BLUE, PALE_GREEN]             # dot/lollipop plots

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
