import matplotlib.pyplot as plt
import seaborn as sns

"""
Centralized visualization style definitions.
Professional BI standards for high data-ink ratio.
"""

# ======================
# Core colors (HEX)
# ======================
# A professional, corporate blue
UNIFORM_BLUE = "#1f77b4"  
# A soft, non-aggressive coral for contrast
PALE_PINK = "#FF6F61"     
# Dark grey for text instead of pure black (reduces eye strain)
GREY_DARK = "#2C3E50"     
# Light grey for grids and secondary elements
GREY_LIGHT = "#BDC3C7"    
WHITE = "#FFFFFF"

# ======================
# Palettes & Colormaps
# ======================
# Qualitative: for distinct categories
BIVARIATE_PALETTE = [UNIFORM_BLUE, PALE_PINK]
# Sequential: for magnitudes (heatmaps, density)
SEQUENTIAL_CMAP = sns.light_palette(UNIFORM_BLUE, as_cmap=True)
# Diverging: for correlations or growth/decline
DIVERGING_CMAP = "RdBu_r" 
# Choropleth Map
SEQUENTIAL_PLOTLY_SCALE = [[0, "#e3f2fd"], [1, "#1f77b4"]] # Light to Uniform Blue

# ======================
# BI Standards & Defaults
# ======================
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_ALPHA = 0.85

# Configuration dictionary for Matplotlib/Seaborn
# This ensures consistency even if Seaborn defaults change
BI_RC_PARAMS = {
    "axes.facecolor": WHITE,
    "axes.edgecolor": GREY_LIGHT,
    "axes.labelcolor": GREY_DARK,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": "#F0F0F0",
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.color": GREY_DARK,
    "ytick.color": GREY_DARK,
    "figure.titlesize": 16,
    "figure.dpi": 100
}

def apply_bi_style():
    """
    Apply the BI expert style to the global matplotlib environment.
    Run this once at the start of your notebook or script.
    """
    plt.rcParams.update(BI_RC_PARAMS)
    sns.set_palette(BIVARIATE_PALETTE)
    print("BI visualization style applied successfully.")
