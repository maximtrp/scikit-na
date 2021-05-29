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

    if not ylabel:
        ylabel = "Frequency" if kind == 'hist' else "Density"

    markbar_kws.setdefault('opacity', 0.5)
    markarea_kws.setdefault('opacity', 0.5)
    markarea_kws.setdefault('interpolate', 'step')
    density_kws.setdefault('density', col)
    density_kws.setdefault('groupby', [col_na])
    density_kws.setdefault('as_', [col, ylabel])

    joinagg_kws.setdefault('total', 'count()')
    joinagg_kws.setdefault('groupby', [col_na])
    calc_kws.setdefault('y', '1 / datum.total')

    x_kws.setdefault('title', xlabel or col)
    y_kws.setdefault('type', 'quantitative')
    y_kws.setdefault('stack', None)
    y_kws.setdefault('title', ylabel)
    color_kws.setdefault('title', na_label or col_na)

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
        X(col, **x_kws),
        Y(y_shorthand, **y_kws),
        Color(col_na, **color_kws)
    )

    return chart


def plot_stairs():
    pass


def plot_heatmap():
    pass
