__all__ = ['plot_dist', 'plot_scatter', 'plot_stairs', 'plot_heatmap']
from altair import Chart, X, Y, Color, selection_multi, condition, value
from altair import data_transformers
from pandas import DataFrame, Index
from numpy import array, arange, ndarray, argmin, r_
from functools import partial
from .._descr import _get_rows_after_cum_dropna
from typing import Union, Optional, List
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

    chart = chart.mark_area(**markarea_kws)\
        if step\
        else chart.mark_bar(**markbar_kws)

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

    selection = selection_multi(fields=[col_na], bind='legend')
    chart = chart.encode(
        x=X(col, **x_kws),
        y=Y(y_shorthand, **y_kws),
        color=Color(col_na, **color_kws),
        tooltip=['count()']
    ).add_selection(selection)

    return chart


def plot_scatter(
        data: DataFrame,
        x: str,
        y: str,
        col_na: str,
        na_label: str = None,
        na_replace: dict = {True: 'NA', False: 'Filled'},
        xlabel: str = None,
        ylabel: str = None,
        circle_kws: dict = {},
        color_kws: dict = {},
        x_kws: dict = {},
        y_kws: dict = {}):
    data_copy = data.loc[:, [x, y, col_na]].copy()
    data_copy[col_na] = data_copy[col_na].isna().replace(na_replace)
    base = Chart(data_copy)

    circle_kws.setdefault('opacity', 0.5)
    x_kws.update({'title': xlabel or x})
    y_kws.update({'title': ylabel or y})
    color_kws.update({'title': col_na or na_label})

    selection = selection_multi(fields=[col_na], bind='legend')
    points = base.mark_circle(**circle_kws).encode(
        x=X(x, **x_kws),
        y=Y(y, **y_kws),
        color=Color(col_na, **color_kws),
        opacity=condition(selection, value(circle_kws['opacity']), value(0))
    ).add_selection(selection)

    return points


def plot_stairs(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        xlabel: str = 'Column',
        ylabel: str = 'Instances',
        tooltip_label: str = 'Size difference',
        dataset_label: str = '(Whole dataset)',
        area_kws: dict = {},
        chart_kws: dict = {},
        x_kws: dict = {},
        y_kws: dict = {}):
    """Stairs plot for dataset and specified columns. Displays the changes in
    dataset size (rows/instances number) after applying
    :py:meth:`pandas.DataFrame.dropna()` to each column cumulatively.

    Columns are sorted by maximum influence on dataset size.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Union[List, ndarray, Index]], optional
        Columns to display on plot.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    tooltip_label : str, optional
        Label for differences in dataset size that is displayed on a tooltip.
    dataset_label : str, optional
        Label for the whole dataset (before dropping any NAs).
    area_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_area` method.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart` class.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X` class.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y` class.

    Returns
    -------
    altair.Chart
        Chart object.
    """
    cols = array(columns if columns is not None else data.columns).tolist()
    stairs_values = []
    stairs_labels = []

    while len(cols) > 0:
        get_rows = partial(
            _get_rows_after_cum_dropna, data, stairs_labels)
        rows_after_dropna = list(map(get_rows, cols))
        stairs_values.append(min(rows_after_dropna))
        stairs_labels.append(cols[argmin(rows_after_dropna)])
        cols.remove(cols[argmin(rows_after_dropna)])

    stairs_values = array([data.shape[0]] + stairs_values)
    stairs_labels = [dataset_label] + stairs_labels
    data_sizes = DataFrame({
        xlabel: stairs_labels,
        ylabel: stairs_values
    })
    data_sizes[tooltip_label] = data_sizes[ylabel].diff().fillna(0)

    area_kws.setdefault('interpolate', 'step-after')
    area_kws.setdefault('line', True)
    x_kws.setdefault('sort', '-y')
    x_kws.update({'shorthand': xlabel})
    y_kws.update({'shorthand': ylabel})

    chart = Chart(data_sizes, **chart_kws)\
        .mark_area(**area_kws)\
        .encode(
            x=X(**x_kws),
            y=Y(**y_kws),
            tooltip=[xlabel, ylabel, tooltip_label]
        )
    return chart


def plot_heatmap(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = [],
        tooltip_cols: Optional[Union[List, ndarray, Index]] = [],
        na_replace: dict = {False: 'Filled', True: 'NA'},
        sort: bool = True,
        xlabel: str = 'Columns',
        ylabel: str = 'Rows',
        zlabel: str = 'Values',
        chart_kws: dict = {'height': 300},
        rect_kws: dict = {},
        x_kws: dict = {'sort': None},
        y_kws: dict = {'sort': None},
        color_kws: dict = {}) -> Chart:
    cols = array(columns if columns is not None else data.columns)
    tt_cols = array(tooltip_cols) if tooltip_cols is not None else []

    data_copy = data.loc[:, r_[cols, tt_cols]].copy()
    data_copy.loc[:, cols] = data_copy.loc[:, cols].isna().replace(na_replace)
    if sort:
        data_copy.sort_values(by=cols.tolist(), inplace=True)
    data_copy[ylabel] = arange(data.shape[0])
    data_copy = data_copy.melt(
        id_vars=r_[[ylabel], tt_cols],
        value_vars=cols,
        var_name=xlabel,
        value_name=zlabel)

    x_kws.update({'shorthand': xlabel, 'type': 'nominal'})
    y_kws.update({'shorthand': ylabel, 'type': 'ordinal'})
    color_kws.update({'shorthand': zlabel, 'type': 'nominal'})

    chart = Chart(data_copy, **chart_kws)\
        .mark_rect(**rect_kws)\
        .encode(
            x=X(**x_kws),
            y=Y(**y_kws),
            color=Color(**color_kws),
            tooltip=tt_cols.tolist()
        )

    return chart
