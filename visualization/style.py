"""
Centralized visualization style definitions.
Graphical roles only – no semantic assumptions.
"""

# ======================
# Core colors (categorical plots)
# ======================
UNIFORM_BLUE = "#1f77b4"       # professional balanced blue
PALE_PINK = "#f4a3c4"          # soft contrast for bivariate/categorical
BLACK = "#000000"              # text/edge labels

# ======================
# Categorical palettes
# ======================
BIVARIATE_PALETTE = [UNIFORM_BLUE, PALE_PINK]
UNIVARIATE_COLOR = UNIFORM_BLUE
MULTIVARIATE_COLOR = UNIFORM_BLUE

# ======================
# Colormaps (continuous plots)
# ======================
SEQUENTIAL_CMAP = "Blues"      # for numeric data
DIVERGING_CMAP = "RdBu_r"      # for correlation/diverging data

# ======================
# Defaults
# ======================
DEFAULT_ALPHA = 0.8
DEFAULT_EDGE_COLOR = "white"
DEFAULT_FIGSIZE = (10,6)
