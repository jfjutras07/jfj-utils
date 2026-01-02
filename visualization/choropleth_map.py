import plotly.express as px

from .style import SEQUENTIAL_PLOTLY_SCALE

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
    """
    
    hover_data = {value_col: True}
    if hover_cols:
        for col in hover_cols:
            hover_data[col] = True

    fig = px.choropleth(
        df,
        locations=country_col,
        locationmode='country names',
        color=value_col,
        hover_name=country_col,
        hover_data=hover_data,
        color_continuous_scale=color_scale,
        title=title
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        coloraxis_colorbar=dict(title=value_col),
        showlegend=show_legend
    )

    return fig
