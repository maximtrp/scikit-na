__all__ = [
    'plot_hist', 'plot_kde', 'plot_corr',
    'plot_scatter', 'plot_stairs', 'plot_heatmap']
from .._stats import _get_rows_after_cum_dropna, _select_cols, correlate
from altair import (
    Chart, Color, condition, data_transformers, selection_multi,
    Scale, Text, value, X, Y)
from pandas import DataFrame, Index
from numpy import array, arange, ndarray, argmin, r_, nan, fill_diagonal
from functools import partial
from typing import Union, Optional, List, Iterable
from ipywidgets import widgets, interact
from numbers import Integral
# Allow plotting mote than 5000 rows
data_transformers.disable_max_rows()


def plot_hist(
        data: DataFrame,
        col: str,
        col_na: str,
        na_label: str = None,
        na_replace: dict = {
            True: 'NA', False: 'Filled'},
        heuristic: bool = True,
        thres_uniq: int = 20,
        step: bool = False,
        norm: bool = True,
        font_size: int = 14,
        xlabel: str = None,
        ylabel: str = "Frequency",
        chart_kws: dict = {},
        markarea_kws: dict = {},
        markbar_kws: dict = {},
        joinagg_kws: dict = {},
        calc_kws: dict = {},
        x_kws: dict = {},
        y_kws: dict = {},
        color_kws: dict = {}) -> Chart:
    """Plot a histogram of values in a column `col` grouped by NA/non-NA values
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
    markbar_kws.setdefault('opacity', 0.5)
    markarea_kws.setdefault('opacity', 0.5)
    markarea_kws.setdefault('interpolate', 'step')

    joinagg_kws.setdefault('total', 'count()')
    joinagg_kws.update({'groupby': [col_na]})

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

    x_kws.update({'title': xlabel or col})

    y_kws.setdefault('type', 'quantitative')
    y_kws.setdefault('stack', None)
    y_kws.update({'title': ylabel})

    calc_kws.setdefault('y', '1 / datum.total')
    color_kws.update({'title': na_label or col_na})

    data_copy = data.loc[:, [col, col_na]].copy()
    data_copy[col_na] = data_copy.loc[:, col_na].isna().replace(na_replace)

    # Chart creation
    chart = Chart(data_copy)

    chart = chart.mark_area(**markarea_kws)\
        if step\
        else chart.mark_bar(**markbar_kws)

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
        na_replace: dict = {
            True: 'NA', False: 'Filled'},
        font_size: int = 14,
        xlabel: str = None,
        ylabel: str = "Density",
        chart_kws: dict = {},
        markarea_kws: dict = {},
        joinagg_kws: dict = {},
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
    markarea_kws.setdefault('opacity', 0.5)
    # markarea_kws.setdefault('interpolate', 'step')

    density_kws.update({'density': col})
    density_kws.update({'groupby': [col_na]})
    density_kws.update({'as_': [col, ylabel]})

    x_kws.update({'title': xlabel or col})

    y_kws.setdefault('type', 'quantitative')
    y_kws.setdefault('stack', None)
    y_kws.update({'title': ylabel})
    y_shorthand = ylabel

    color_kws.update({'title': na_label or col_na})

    data_copy = data.loc[:, [col, col_na]].copy()
    data_copy[col_na] = data_copy.loc[:, col_na].isna().replace(na_replace)

    # Chart creation
    chart = Chart(data_copy).mark_area(**markarea_kws)
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
        x: str,
        y: str,
        col_na: str,
        na_label: str = None,
        na_replace: dict = {True: 'NA', False: 'Filled'},
        font_size: int = 14,
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
    return chart\
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)\
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)


def plot_heatmap(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        tooltip_cols: Optional[Iterable] = None,
        names: list = ['Filled', 'NA', 'Droppable'],
        sort: bool = True,
        droppable: bool = True,
        font_size: int = 14,
        xlabel: str = 'Columns',
        ylabel: str = 'Rows',
        zlabel: str = 'Values',
        chart_kws: dict = {'height': 300},
        rect_kws: dict = {},
        x_kws: dict = {'sort': None},
        y_kws: dict = {'sort': None},
        color_kws: dict = {}) -> Chart:
    """Heatmap plot for NA/non-NA values. By default, it also displays values
    that are to be dropped by :py:meth:`pandas.DataFrame.dropna()` method.

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

    x_kws.update({'shorthand': xlabel, 'type': 'nominal'})
    y_kws.update({'shorthand': ylabel, 'type': 'ordinal'})
    color_kws.update({'shorthand': zlabel, 'type': 'nominal'})
    color_kws.setdefault('scale', Scale(
        domain=names[0:2] if not droppable else names,
        range=["green", "red", "orange"]
    ))

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
        drop: bool = True,
        mask_diag: bool = True,
        annot: bool = True,
        annot_color: str = "black",
        round_sgn: int = 2,
        font_size: int = 14,
        opacity: float = 0.5,
        cmap: str = "redblue",
        corr_kws: dict = {},
        chart_kws: dict = {},
        x_kws: dict = {},
        y_kws: dict = {},
        color_kws: dict = {},
        text_kws: dict = {}) -> Chart:
    """Plot a correlation heatmap.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Iterable]
        Columns names.
    drop : bool = True
        Drop columns without NAs.
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
    cols = _select_cols(data, columns)

    corr_kws.setdefault('method', 'spearman')
    data_corr = correlate(data, columns=cols, **corr_kws)
    if mask_diag:
        fill_diagonal(data_corr.values, nan)
    data_corr_melt = data_corr.reset_index(drop=False).melt(id_vars=['index'])

    chart_kws.update({'data': data_corr_melt})
    x_kws.setdefault('shorthand', 'variable')
    x_kws.setdefault('title', '')
    y_kws.setdefault('shorthand', 'index')
    y_kws.setdefault('title', '')

    color_kws.setdefault('shorthand', 'value:Q')
    color_kws.setdefault('title', 'Correlation')
    color_kws.setdefault(
        'scale', Scale(scheme='redblue', domain=[-1, 1], reverse=True))

    text_kws.setdefault('shorthand', 'value:Q')
    text_kws.setdefault('format', f'.{round_sgn}f')

    base = Chart(**chart_kws).encode(x=X(**x_kws), y=Y(**y_kws))
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
    """Interactively observe distribution of values in a selected column
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
