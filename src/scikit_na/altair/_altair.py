__all__ = ['plot_dist', 'plot_stairs', 'plot_heatmap']
from altair import Chart, X, Y, Color
from altair import data_transformers
from pandas import DataFrame
data_transformers.disable_max_rows()


def plot_dist(
        data: DataFrame,
        col: str,
        col_na: str,
        na_label: str = None,
        na_replace: dict = {True: 'NA', False: 'Filled'},
        kind: str = "hist",
        step: bool = False,
        norm: bool = True,
        xlabel: str = None,
        ylabel: str = None,
        chart_kws: dict = {},
        markarea_kws: dict = {},
        markbar_kws: dict = {},
        joinagg_kws: dict = {},
        calc_kws: dict = {},
        density_kws: dict = {},
        x_kws: dict = {},
        y_kws: dict = {},
        color_kws: dict = {}) -> Chart:
    """Plot distribution of values in a column `col` grouped by
    NA/non-NA values in column `col_na`.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col : str
        Column to display distribution of values.
    col_na : str
        Column to group values by.
    na_label : str, optional
        Legend title.
    na_replace : dict, optional
        Dictionary to replace values returned by
        :py:meth:`pandas.Series.isna()` method.
    kind : str, optional
        Plot kind: "hist" or "kde".
    step : bool, optional
        Draw step plot.
    norm : bool, optional
        Normalize values in groups.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()`.
    markarea_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_area()`.
    markbar_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_bar()`.
    joinagg_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`altair.Chart.transform_joinaggregate()`.
    calc_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`altair.Chart.transform_calculate()`.
    density_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`altair.Chart.transform_density()`.
    x_kws : dict, optional
        Keyword arguments passed to py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to py:meth:`altair.Y()`.
    color_kws : dict, optional
        Keyword arguments passed to py:meth:`altair.Color()`.

    Returns
    -------
    Chart
        Altair Chart object.
    """

    if not ylabel:
        ylabel = "Frequency" if kind == 'hist' else "Density"

    markbar_kws.setdefault('opacity', 0.5)
    markarea_kws.setdefault('opacity', 0.5)
    markarea_kws.setdefault('interpolate', 'step')

    density_kws.update({'density': col})
    density_kws.update({'groupby': [col_na]})
    density_kws.update({'as_': [col, ylabel]})

    joinagg_kws.setdefault('total', 'count()')
    joinagg_kws.update({'groupby': [col_na]})

    x_kws.setdefault('bin', True)
    x_kws.update({'title': xlabel or col})

    y_kws.setdefault('type', 'quantitative')
    y_kws.setdefault('stack', None)
    y_kws.update({'title': ylabel})

    calc_kws.setdefault('y', '1 / datum.total')
    color_kws.update({'title': na_label or col_na})

    data_copy = data.loc[:, [col, col_na]].copy()
    data_copy[col_na] = data_copy.loc[:, col_na].isna().replace(na_replace)

    # Chart creation routine
    chart = Chart(data_copy)

    if step:
        chart = chart.mark_area(**markarea_kws)
    else:
        chart = chart.mark_bar(**markbar_kws)

    if kind == "hist":
        if norm:
            y_shorthand = 'sum(y)'
            chart = chart.transform_joinaggregate(**joinagg_kws)
            chart = chart.transform_calculate(**calc_kws)
        else:
            y_shorthand = 'count()'

    elif kind == 'kde':
        y_shorthand = ylabel
        chart = chart.transform_density(**density_kws)

    chart = chart.encode(
        x=X(col, **x_kws),
        y=Y(y_shorthand, **y_kws),
        color=Color(col_na, **color_kws),
        tooltip=['count()']
    )

    return chart


def plot_stairs():
    pass


def plot_heatmap():
    pass
