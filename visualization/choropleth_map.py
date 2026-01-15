import plotly.express as px
from .style import SEQUENTIAL_PLOTLY_SCALE, GREY_DARK, WHITE

#---Function: plot_choropleth---
def plot_choropleth(
    df, 
    country_col='Country', 
    value_col='PC1_health_index', 
    title='Choropleth Map', 
    color_scale=SEQUENTIAL_PLOTLY_SCALE,
    hover_cols=None,
    show_legend=True
):
    """
    Generic function to create a choropleth map for country-level data.
    Aligned with BI style standards for clean cartographic display.
    """
    
    # Prepare hover data configuration with fixed precision
    hover_data = {value_col: ":.2f"}
    if hover_cols:
        for col in hover_cols:
            hover_data[col] = True

    # Build the figure using Plotly Express
    fig = px.choropleth(
        df,
        locations=country_col,
        locationmode='country names',
        color=value_col,
        hover_name=country_col,
        hover_data=hover_data,
        color_continuous_scale=color_scale,
        title=f"<b>{title}</b>"
    )

    # Apply BI Style Refinements for layout and aesthetics
    fig.update_layout(
        title_font=dict(size=18, color=GREY_DARK),
        geo=dict(
            showframe=False, 
            showcoastlines=True, 
            projection_type='equirectangular',
            coastlinecolor=WHITE,
            landcolor="#f9f9f9"
        ),
        coloraxis_colorbar=dict(
            title=dict(text=value_col, side="right"),
            thicknessmode="pixels", thickness=15,
            lenmode="fraction", len=0.6
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        showlegend=show_legend
    )

    return fig
