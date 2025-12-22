import plotly.express as px

#--- Function : plot_choropleth ---
def plot_choropleth(
    df, 
    country_col='Country', 
    value_col='PC1_health_index', 
    title='Choropleth Map', 
    color_scale='RdYlGn_r',
    hover_cols=None,
    show_legend=True
):
    """
    Generic function to create a choropleth map for country-level data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least country names and a numeric value to map.
    country_col : str
        Name of the column with country names (ISO-3 codes recommended, but names also work).
    value_col : str
        Column to visualize with color.
    title : str
        Title of the map.
    color_scale : str
        Color scale for the map. Default 'RdYlGn_r' (green = good, red = bad).
    hover_cols : list of str or None
        Additional columns to show when hovering over a country.
    show_legend : bool
        Whether to show the color scale legend.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Choropleth figure object.
    """
    
    hover_data = {value_col: True}
    if hover_cols:
        for col in hover_cols:
            hover_data[col] = True

    fig = px.choropleth(
        df,
        locations=country_col,
        locationmode='country names',  # change to 'ISO-3' if using ISO codes
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
