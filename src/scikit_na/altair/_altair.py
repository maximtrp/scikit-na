"""Altair-backed plotting functions."""
__all__ = [
    'plot_hist', 'plot_kde', 'plot_corr',
    'plot_scatter', 'plot_stairs', 'plot_heatmap']
from typing import Union, Optional, List, Iterable
from functools import partial
from numbers import Integral
from ipywidgets import widgets, interact
from numpy import array, arange, ndarray, argmin, r_, nan, fill_diagonal
from pandas import DataFrame, Index
from altair import (
    Chart, Color, condition, data_transformers, selection_multi,
    Scale, Text, value, X, Y)
from .._stats import _get_rows_after_cum_dropna, _select_cols, correlate
# Allow plotting mote than 5000 rows
data_transformers.disable_max_rows()


def plot_hist(
        data: DataFrame,
        col: str,
        col_na: str,
        na_label: str = None,
        na_replace: dict = None,
        heuristic: bool = True,
        thres_uniq: int = 20,
        step: bool = False,
        norm: bool = True,
        font_size: int = 14,
        xlabel: str = None,
        ylabel: str = "Frequency",
        chart_kws: dict = None,
        markarea_kws: dict = None,
        markbar_kws: dict = None,
        joinagg_kws: dict = None,
        calc_kws: dict = None,
        x_kws: dict = None,
        y_kws: dict = None,
        color_kws: dict = None) -> Chart:
    """Histogram plot.

    Plots a histogram of values in a column `col` grouped by NA/non-NA values
    in column `col_na`.

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
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()`.

    Returns
    -------
    Chart
        Altair Chart object.
    """
    if not chart_kws:
        chart_kws = {}
    if not markarea_kws:
        markarea_kws = {'opacity': 0.5, 'interpolate': 'step'}
    if not markbar_kws:
        markbar_kws = {'opacity': 0.5}
    if not joinagg_kws:
        joinagg_kws = {'total': 'count()', 'groupby': [col_na]}
    if not calc_kws:
        calc_kws = {'y': '1 / datum.total'}
    if not x_kws:
        x_kws = {'title': xlabel or col}
    if not y_kws:
        y_kws = {'type': 'quantitative', 'stack': None, 'title': ylabel}
    if not color_kws:
        color_kws = {'title': na_label or col_na}
    if not na_replace:
        na_replace = {True: 'NA', False: 'Filled'}

    # Simple heuristic for choosing histplot parameters
    if heuristic:
        # 1) If dtype is object, do not bin anything and treat as nominal
        if data[col].dtype == object:
            x_kws.update({'bin': False})
            x_kws.update({'type': 'nominal'})

        # 2) Check the number of unique values
        else:
            few_uniques = data[col].dropna().unique().size < thres_uniq
            integers = data[col].dropna()\
                .apply(lambda x: not isinstance(x, Integral)).sum()

            if not integers and few_uniques:
                x_kws.update({'bin': False})
                x_kws.update({'type': 'ordinal'})
            else:
                x_kws.update({'bin': True})
                x_kws.update({'type': 'quantitative'})

    data_copy = data.loc[:, [col, col_na]].copy()
    data_copy[col_na] = data_copy.loc[:, col_na].isna().replace(na_replace)

    # Chart creation
    chart = Chart(data_copy)

    chart = chart.mark_area(**markarea_kws)\
        if step\
        else chart.mark_bar(**markbar_kws)

    # Normed vs non-normed histogram
    if norm:
        y_shorthand = 'sum(y)'
        chart = chart.transform_joinaggregate(**joinagg_kws)
        chart = chart.transform_calculate(**calc_kws)
    else:
        y_shorthand = 'count()'

    selection = selection_multi(fields=[col_na], bind='legend')
    chart = chart.encode(
        x=X(col, **x_kws),
        y=Y(y_shorthand, **y_kws),
        color=Color(col_na, **color_kws),
        tooltip=['count()'],
        opacity=condition(
            selection,
            value(markarea_kws['opacity'] if step else markbar_kws['opacity']),
            value(0))
    ).add_selection(selection)

    return chart\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)


def plot_kde(
        data: DataFrame,
        col: str,
        col_na: str,
        na_label: str = None,
        na_replace: dict = None,
        font_size: int = 14,
        xlabel: str = None,
        ylabel: str = "Density",
        chart_kws: dict = None,
        markarea_kws: dict = None,
        density_kws: dict = None,
        x_kws: dict = None,
        y_kws: dict = None,
        color_kws: dict = None) -> Chart:
    """Density plot.

    Plots distribution of values in a column `col` grouped by
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
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()`.
    markarea_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_area()`.
    density_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`altair.Chart.transform_density()`.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()`.

    Returns
    -------
    Chart
        Altair Chart object.
    """
    if not chart_kws:
        chart_kws = {}
    if not markarea_kws:
        markarea_kws = {'opacity': 0.5}
    if not density_kws:
        density_kws = {
            'density': col, 'groupby': [col_na], 'as_': [col, ylabel]}
    if not x_kws:
        x_kws = {'title': xlabel or col}
    if not y_kws:
        y_kws = {'type': 'quantitative', 'stack': None, 'title': ylabel}
    if not color_kws:
        color_kws = {'title': na_label or col_na}
    if not na_replace:
        na_replace = {True: 'NA', False: 'Filled'}

    y_shorthand = ylabel

    data_copy = data.loc[:, [col, col_na]].copy()
    data_copy[col_na] = data_copy.loc[:, col_na].isna().replace(na_replace)

    # Chart creation
    chart = Chart(data_copy, **chart_kws).mark_area(**markarea_kws)
    chart = chart.transform_density(**density_kws)

    selection = selection_multi(fields=[col_na], bind='legend')
    chart = chart.encode(
        x=X(col, **x_kws),
        y=Y(y_shorthand, **y_kws),
        color=Color(col_na, **color_kws),
        opacity=condition(
            selection,
            value(markarea_kws['opacity']),
            value(0))
    ).add_selection(selection)

    return chart\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)


def plot_scatter(
        data: DataFrame,
        x_col: str,
        y_col: str,
        col_na: str,
        na_label: str = None,
        na_replace: dict = None,
        font_size: int = 14,
        xlabel: str = None,
        ylabel: str = None,
        circle_kws: dict = None,
        color_kws: dict = None,
        x_kws: dict = None,
        y_kws: dict = None):
    """Scatter plot.

    Parameters
    ----------
    data : DataFrame
        Input data.
    x_col : str
        Column name corresponding to X axis.
    y_col : str
        Column name corresponding to Y axis.
    col_na : str
        Column name
    na_label : str, optional
        Label for NA values in legend.
    na_replace : dict, optional
        NA replacement mapping, by default {True: 'NA', False: 'Filled'}.
    font_size : int, optional
        Font size for plotting, by default 14.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    circle_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_circle()`.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()`.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.

    Returns
    -------
    altair.Chart
        Scatter plot.
    """
    if not circle_kws:
        circle_kws = {'opacity': 0.5}
    if not color_kws:
        color_kws = {'title': col_na or na_label}
    if not x_kws:
        x_kws = {'title': xlabel or x_col}
    if not y_kws:
        y_kws = {'title': ylabel or y_col}
    if not na_replace:
        na_replace = {True: 'NA', False: 'Filled'}

    data_copy = data.loc[:, [x_col, y_col, col_na]].copy()
    data_copy[col_na] = data_copy[col_na].isna().replace(na_replace)
    base = Chart(data_copy)

    selection = selection_multi(fields=[col_na], bind='legend')
    points = base.mark_circle(**circle_kws).encode(
        x=X(x_col, **x_kws),
        y=Y(y_col, **y_kws),
        color=Color(col_na, **color_kws),
        opacity=condition(selection, value(circle_kws['opacity']), value(0))
    ).add_selection(selection)

    return points\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)


def plot_stairs(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        xlabel: str = 'Columns',
        ylabel: str = 'Instances',
        tooltip_label: str = 'Size difference',
        dataset_label: str = '(Whole dataset)',
        font_size: int = 14,
        area_kws: dict = None,
        chart_kws: dict = None,
        x_kws: dict = None,
        y_kws: dict = None):
    """Stairs plot.

    Plots changes in dataset size (rows/instances number) after applying
    :py:meth:`pandas.DataFrame.dropna()` to each column cumulatively.

    Columns are sorted by maximum influence on dataset size.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Union[List, ndarray, Index]], optional
        Columns that are to be displayed on a plot.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    tooltip_label : str, optional
        Label for differences in dataset size that is displayed on a tooltip.
    dataset_label : str, optional
        Label for the whole dataset (before dropping any NAs).
    area_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_area()` method.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()` class.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()` class.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()` class.

    Returns
    -------
    altair.Chart
        Chart object.
    """
    if not area_kws:
        area_kws = {'interpolate': 'step-after', 'line': True}
    if not chart_kws:
        chart_kws = {}
    if not x_kws:
        x_kws = {'sort': '-y', 'shorthand': xlabel}
    if not y_kws:
        y_kws = {'shorthand': ylabel}

    cols = _select_cols(data, columns).tolist()
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

    chart = Chart(data_sizes, **chart_kws)\
        .mark_area(**area_kws)\
        .encode(
            x=X(**x_kws),
            y=Y(**y_kws),
            tooltip=[xlabel, ylabel, tooltip_label]
        )
    return chart\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)


def plot_heatmap(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        tooltip_cols: Optional[Iterable] = None,
        names: list = None,
        sort: bool = True,
        droppable: bool = True,
        font_size: int = 14,
        xlabel: str = 'Columns',
        ylabel: str = 'Rows',
        zlabel: str = 'Values',
        chart_kws: dict = None,
        rect_kws: dict = None,
        x_kws: dict = None,
        y_kws: dict = None,
        color_kws: dict = None) -> Chart:
    """Heatmap plot for NA/non-NA values.

    By default, it also indicates values that are to be dropped by
    :py:meth:`pandas.DataFrame.dropna()` method.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Iterable], optional
        Columns that are to be displayed on a plot.
    tooltip_cols : Optional[Iterable], optional
        Columns to be used in tooltips.
    names : list, optional
        Values labels passed as a list.
        The first element corresponds to non-missing values,
        the second one to NA values, and the last one to droppable values, i.e.
        values to be dropped by :py:meth:`pandas.DataFrame.dropna()`.
    sort : bool, optional
        Sort values as NA/non-NA.
    droppable : bool, optional
        Show values to be dropped by :py:meth:`pandas.DataFrame.dropna()`
        method.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    zlabel : str, optional
        Groups label (shown as a legend title).
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()` class.
    rect_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_rect()` method.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()` class.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()` class.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()` class.

    Returns
    -------
    altair.Chart
        Altair Chart object.
    """
    if not chart_kws:
        chart_kws = {'height': 300}
    if not x_kws:
        x_kws = {'sort': None, 'shorthand': xlabel, 'type': 'nominal'}
    if not y_kws:
        y_kws = {'sort': None, 'shorthand': ylabel, 'type': 'ordinal'}
    if not names:
        names = ['Filled', 'NA', 'Droppable']
    if not color_kws:
        color_kws = {
            'shorthand': zlabel,
            'type': 'nominal',
            'scale': Scale(
                domain=names[0:2] if not droppable else names,
                range=["green", "red", "orange"])
        }
    if not rect_kws:
        rect_kws = {}

    cols = _select_cols(data, columns)
    tt_cols = _select_cols(data, tooltip_cols, [])

    data_copy = data.loc[:, r_[cols, tt_cols]].copy()
    data_copy.loc[:, cols] = data_copy.loc[:, cols].isna()
    if sort:
        cols_sorted = data_copy.loc[:, cols]\
            .sum()\
            .sort_values(ascending=False)\
            .index.tolist()
        data_copy.sort_values(by=cols_sorted, inplace=True)
        x_kws.update({'sort': cols_sorted})

    if droppable:
        non_na_mask = ~data_copy.loc[:, cols].values
        na_rows_mask = data_copy.loc[:, cols].any(axis=1).values[:, None]
        droppable_mask = non_na_mask & na_rows_mask
        data_copy.loc[:, cols] = data_copy.loc[:, cols].astype(int)
        data_copy.loc[:, cols] = data_copy.loc[:, cols]\
            .mask(droppable_mask, other=2)
    else:
        data_copy.loc[:, cols] = data_copy.loc[:, cols].astype(int)

    data_copy.loc[:, cols] = data_copy.loc[:, cols].replace(
        dict(zip([0, 1, 2], names)))

    data_copy[ylabel] = arange(data.shape[0])
    data_copy = data_copy.melt(
        id_vars=r_[[ylabel], tt_cols],
        value_vars=cols,
        var_name=xlabel,
        value_name=zlabel)

    chart = Chart(data_copy, **chart_kws)\
        .mark_rect(**rect_kws)\
        .encode(
            x=X(**x_kws),
            y=Y(**y_kws),
            color=Color(**color_kws),
            tooltip=tt_cols.tolist()
        )

    return chart\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)


def plot_corr(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        mask_diag: bool = True,
        annot_color: str = "black",
        round_sgn: int = 2,
        font_size: int = 14,
        opacity: float = 0.5,
        corr_kws: dict = None,
        chart_kws: dict = None,
        x_kws: dict = None,
        y_kws: dict = None,
        color_kws: dict = None,
        text_kws: dict = None) -> Chart:
    """Correlation heatmap.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Iterable]
        Columns names.
    mask_diag : bool = True
        Mask diagonal on heatmap.
    corr_kws : dict, optional
        Keyword arguments passed to :py:meth:`pandas.DataFrame.corr()` method.
    heat_kws : dict, optional
        Keyword arguments passed to :py:meth:`seaborn.heatmap()` method.

    Returns
    -------
    altair.Chart
        Altair Chart object.
    """
    if not corr_kws:
        corr_kws = {'method': 'spearman'}
    if not chart_kws:
        chart_kws = {}
    if not x_kws:
        x_kws = {'shorthand': 'variable', 'title': ''}
    if not y_kws:
        y_kws = {'shorthand': 'index', 'title': ''}
    if not color_kws:
        color_kws = {
            'shorthand': 'value:Q',
            'title': 'Correlation',
            'scale': Scale(scheme="redblue", domain=[-1, 1], reverse=True)}
    if not text_kws:
        text_kws = {'shorthand': 'value:Q', 'format': f'.{round_sgn}f'}

    cols = _select_cols(data, columns)

    data_corr = correlate(data, columns=cols, **corr_kws)

    if mask_diag:
        fill_diagonal(data_corr.values, nan)
    data_corr_melt = data_corr.reset_index(drop=False).melt(id_vars=['index'])

    base = Chart(data_corr_melt, **chart_kws)\
        .encode(x=X(**x_kws), y=Y(**y_kws))

    heatmap = base.mark_rect().encode(color=Color(**color_kws))
    text = base.mark_text(baseline='middle').encode(text=Text(**text_kws))

    # Draw the chart
    return (heatmap + text)\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_text(fontSize=font_size, color=annot_color)\
        .configure_rect(opacity=opacity)


def view_dist(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        **kwargs):
    """Interactive distribution widget.

    Interactively observe distribution of values in a selected column
    grouped by NA/non-NA values in another column.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Union[list, ndarray, Index] = None
        Column names.

    Returns
    -------
    _InteractFactory
        Interactive widget.
    """
    cols = _select_cols(data, columns)
    na_cols = data.isna().sum(axis=0)\
        .rename('na_num')\
        .to_frame()\
        .query('na_num > 0')\
        .index.values

    return interact(
        lambda Column, NA:
            plot_hist(data, col=Column, col_na=NA, **kwargs)
            if Column != NA
            else widgets.HTML(
                '<em style="color: red">Note: select different columns</em>'),
        Column=cols, NA=na_cols)
