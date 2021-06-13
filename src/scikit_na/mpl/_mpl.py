"""Matplotlib-backed visualization"""
__all__ = [
    'plot_corr', 'plot_stats', 'plot_heatmap', 'plot_hist', 'plot_kde']
from typing import Optional, Union, List, Iterable
from pandas import DataFrame
from pandas.core.indexes.base import Index
from numpy import (
    ndarray, fill_diagonal, nan)
from seaborn import heatmap, histplot, kdeplot, barplot, diverging_palette
from matplotlib.axes import SubplotBase
from matplotlib.patches import Patch
from .._stats import correlate, _select_cols


def plot_corr(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        mask_diag: bool = True,
        corr_kws: dict = None,
        heat_kws: dict = None) -> SubplotBase:
    """Plot a correlation heatmap.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Union[List, ndarray, Index]], optional
        Columns names.
    mask_diag : bool = True
        Mask diagonal on heatmap.
    corr_kws : dict, optional
        Keyword arguments passed to :py:meth:`pandas.DataFrame.corr()`.
    heat_kws : dict, optional
        Keyword arguments passed to :py:meth:`pandas.DataFrame.heatmap()`.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Heatmap AxesSubplot object.
    """
    if not heat_kws:
        heat_kws = {
            'vmin': -1, 'vmax': 1, 'annot': True, 'square': True,
            'cmap': diverging_palette(240, 10, as_cmap=True)}
    if not corr_kws:
        corr_kws = {'method': 'spearman'}

    cols = _select_cols(data, columns)

    data_corr = correlate(data, columns=cols, **corr_kws)
    if mask_diag:
        fill_diagonal(data_corr.values, nan)

    return heatmap(data_corr, **heat_kws)


def plot_stats(
        na_info: DataFrame,
        idxstr: str = None,
        idxint: int = None,
        **kwargs) -> SubplotBase:
    """Plot barplot with NA descriptive statistics.

    Parameters
    ----------
    na_info : DataFrame
        Typically, the output of :py:meth:`scikit_na.describe()` method.
    idxstr : str = None, optional
        Index string labels passed to :py:meth:`pandas.DataFrame.loc` method.
    idxint : int = None, optional
        Index integer labels passed to :py:meth:`pandas.DataFrame.iloc` method.
    kwargs : dict, optional
        Keyword arguments passed to :py:meth:`seaborn.barplot()` method.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Barplot AxesSubplot object.

    Raises
    ------
    ValueError
        Raised if neither ``idxstr`` nor ``idxint`` are passed.
    """
    if not (idxstr is not None or idxint is not None):
        raise ValueError("Error: `idxstr` or `idxint` must be specified")

    if idxstr is not None:
        y_data = na_info.loc[idxstr, :]
    elif idxint is not None:
        y_data = na_info.iloc[idxint, :]
    x_data = na_info.columns

    return barplot(x=x_data, y=y_data, **kwargs)


# def plot_raw(
#         data: DataFrame,
#         columns: Optional[Union[List, ndarray, Index]] = None,
#         splt_kwargs: dict = {},
#         plt_kwargs: dict = {}) -> SubplotBase:
#     """NA eventplot. Plots NA values as red lines and normal values
#     as black lines.

#     Parameters
#     ----------
#     data : DataFrame
#         Input data.
#     columns : Optional[Union[List, ndarray, Index]], optional
#         Columns names.
#     splt_kwargs : dict, optional
#         Keyword arguments passed to
#         :py:meth:`matplotlib.pyplot.subplots` method.
#     plt_kwargs : dict, optional
#         Keyword arguments passed to
#         :py:meth:`matplotlib.axes._subplots.AxesSubplot.eventplot` method.

#     Returns
#     -------
#     matplotlib.axes._subplots.AxesSubplot
#         AxesSubplot object.
#     """
#     cols = array(columns) if columns is not None else data.columns.tolist()
#     fig, ax = subplots(**splt_kwargs)
#     non_nas = []
#     nas = []
#     data_na = data.loc[:, cols].isna().sort_values(by=cols)

#     for i, col in enumerate(cols):
#         na_mask = data_na.loc[:, col].values
#         non_na = argwhere(~na_mask).ravel()
#         na = argwhere(na_mask).ravel()
#         non_nas.append(non_na)
#         nas.append(na)

#     ax.eventplot(
#         non_nas, colors=plt_kwargs.get('colors', ['', 'k'])[0], **plt_kwargs)
#     ax.eventplot(
#         nas, colors=plt_kwargs.get('colors', ['', 'r'])[1], **plt_kwargs)

#     if plt_kwargs.get('orientation', None) == 'vertical':
#         ax.set_xticks(arange(len(cols)))
#         ax.set_xticklabels(cols)
#     else:
#         ax.set_yticks(arange(len(cols)))
#         ax.set_yticklabels(cols)

#     return ax


def plot_heatmap(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        droppable: bool = True,
        sort: bool = True,
        cmap: Optional[Union[List, ndarray]] = None,
        names: Optional[Union[List, ndarray]] = None,
        yaxis: bool = False,
        xaxis: bool = True,
        legend_kws: dict = None,
        sb_kws: dict = None) -> SubplotBase:
    """NA heatmap. Plots NA values as red lines and normal values
    as black lines.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Iterable], optional
        Columns names.
    droppable : bool, optional
        Show values to be dropped by :py:meth:`pandas.DataFrame.dropna()`
        method.
    sort : bool, optional
        Sort DataFrame by selected columns.
    cmap : Optional[Union[List, ndarray]], optional
        Heatmap and legend colormap: non-missing values, droppable values,
        NA values, correspondingly. Passed to :py:meth:`seaborn.heatmap()`
        method.
    names : Optional[Union[List, ndarray]], optional
        Legend labels: non-missing values, droppable values,
        NA values, correspondingly.
    yaxis : bool, optional
        Show Y axis.
    xaxis : bool, optional
        Show X axis.
    legend_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`matplotlib.axes._subplots.AxesSubplot()` method.
    sb_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`seaborn.heatmap` method.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        AxesSubplot object.
    """
    if not cmap:
        cmap = ['green', 'orange', 'red']
    if not names:
        names = ['Filled', 'Droppable', 'NA']
    if not sb_kws:
        sb_kws = {'cbar': False}

    cols = _select_cols(data, columns).tolist()
    data_na = data.loc[:, cols].isna().copy()
    if sort:
        data_na.sort_values(by=cols, inplace=True)

    if droppable:
        non_na_mask = ~data_na.values
        na_rows_mask = data_na.any(axis=1).values[:, None]
        droppable_mask = non_na_mask & na_rows_mask
        data_na = data_na.astype(float)
        data_na.values[droppable_mask] = 0.5
        labels = names
    else:
        labels = [names[0], names[-1]]

    if not legend_kws:
        legend_kws = {'bbox_to_anchor': (0.5, 1.15), 'loc': 'upper center', 'ncol': len(labels)}

    ax_heatmap = heatmap(data_na, cmap=cmap, **sb_kws)
    ax_heatmap.yaxis.set_visible(yaxis)
    ax_heatmap.xaxis.set_visible(xaxis)
    legend_elements = [Patch(facecolor=cmap[0]), Patch(facecolor=cmap[-1])]
    if droppable:
        legend_elements.insert(1, Patch(facecolor=cmap[1]))
    ax_heatmap.legend(legend_elements, labels, **legend_kws)

    return ax_heatmap


# def plot_stairs(
#         data: DataFrame,
#         columns: Optional[Union[List, ndarray, Index]] = None,
#         frame: bool = False,
#         grid: bool = False,
#         labels: bool = True,
#         sort: Optional[Union["min", "max", None]] = "min",
#         splt_kws: dict = {},
#         stairs_kws: dict = {},
#         text_kws: dict = {}) -> SubplotBase:
#     """Stairs plot of cumulative changes in rows number
#     after applying :py:meth:`pandas.DataFrame.dropna()` method.

#     Parameters
#     ----------
#     data : DataFrame
#         Input data.
#     columns : Optional[Union[List, ndarray, Index]], optional
#         Column names.
#     frame : bool, optional
#         Draw axes frame.
#     grid : bool, optional
#         Draw grid.
#     sort : Optional[Union["min", "max", None]], optional
#         Sort columns by maximum number of NA values.
#     labels : bool = True
#         Plot data shapes above bars.
#     splt_kws : dict, optional
#         Keyword arguments passed to :py:meth:`matplotlib.pyplot.subplots`.
#     stairs_kws : dict, optional
#         Keyword arguments passed to :py:meth:`matplotlib.pyplot.stairs`.
#     text_kws : dict, optional
#         Keyword arguments passed to :py:meth:`matplotlib.pyplot.text`.

#     Returns
#     -------
#     matplotlib.axes._subplots.AxesSubplot
#         AxesSubplot object.
#     """
#     cols = array(columns)\
#         if columns is not None else data.columns.values
#     _cols = cols.copy().tolist()
#     stairs_values = []
#     stairs_labels = []

#     if sort == 'min':
#         get_func = min
#         arg_func = argmin
#     elif sort == 'max':
#         get_func = max
#         arg_func = argmax
#     else:
#         get_func = lambda x: x[0]
#         arg_func = lambda _: 0

#     while len(_cols) > 0:
#         get_rows = partial(
#             _get_rows_after_cum_dropna, data, stairs_labels)
#         rows_after_dropna = list(map(get_rows, _cols))
#         stairs_values.append(get_func(rows_after_dropna))
#         stairs_labels.append(_cols[arg_func(rows_after_dropna)])
#         _cols.remove(_cols[arg_func(rows_after_dropna)])

#     stairs_values = array([data.shape[0]] + stairs_values)
#     stairs_labels = ["Whole dataset"] + stairs_labels

#     stairs_kws.setdefault('fill', True)
#     stairs_kws.setdefault('linewidth', 2)
#     stairs_kws.setdefault('ec', 'black')
#     text_kws.setdefault('ha', 'center')

#     fig, ax = subplots(**splt_kws)
#     ax.stairs(values=stairs_values, **stairs_kws)

#     # Drawing labels
#     stairs_values_diff = (diff(stairs_values) * -1)
#     stairs_values_diff = insert(
#         stairs_values_diff, 0, stairs_values_diff.mean())
#     labels_y = stairs_values +\
#         max(stairs_values_diff.mean() * 0.2, max(stairs_values) * 0.066)
#     labels_x = arange(len(cols)+1) + 0.5

#     if labels:
#         for text, x, y in zip(stairs_values, labels_x, labels_y):
#             ax.text(x, y, text, **text_kws)

#     # Plot settings
#     ax.set_ylim(0, max(labels_y))
#     ax.yaxis.set_visible(False)
#     ax.set_xticks(labels_x)
#     ax.set_xticklabels(stairs_labels)
#     ax.set_frame_on(frame)
#     ax.grid(grid)
#     fig.tight_layout()

#     return ax


def plot_hist(
        data: DataFrame,
        col: str,
        col_na: str,
        col_na_fmt: str = '"{}" is NA',
        stat: str = "density",
        common_norm: bool = False,
        hist_kws: dict = None) -> SubplotBase:
    """Histogram plot to compare distributions of values in column `col`
    split into two groups (NA/Non-NA) by column `col_na` in input DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame.
    col : str
        Name of column to compare distributions of values.
    col_na : str
        Name of column to group values by (NA/Non-NA).
    col_na_fmt : str
        Legend title format string.
    common_norm : bool, optional
        Use common norm.
    hist_kws : dict, optional
        Keyword arguments passed to :py:meth:`seaborn.histplot`.

    Returns
    -------
    SubplotBase
        AxesSubplot returned by :py:meth:`seaborn.histplot`.
    """
    if not hist_kws:
        hist_kws = {'stat': stat, 'common_norm': common_norm}

    data_copy = data.copy()
    col_na_name = col_na_fmt.format(col_na)
    data_copy[col_na_name] = data_copy.loc[:, col_na].isna()

    return histplot(x=col, hue=col_na_name, data=data_copy, **hist_kws)


def plot_kde(
        data: DataFrame,
        col: str,
        col_na: str,
        col_na_fmt: str = '"{}" is NA',
        common_norm: bool = False,
        kde_kws: dict = None) -> SubplotBase:
    """KDE plot to compare distributions of values in column `col`
    split into two groups (NA/Non-NA) by column `col_na` in input DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame.
    col : str
        Name of column to compare distributions of values.
    col_na : str
        Name of column to group values by (NA/Non-NA).
    col_na_fmt : str
        Legend title format string.
    common_norm : bool, optional
        Use common norm.
    kde_kws : dict, optional
        Keyword arguments passed to :py:meth:`seaborn.kdeplot()`.

    Returns
    -------
    SubplotBase
        AxesSubplot returned by :py:meth:`seaborn.kdeplot()`.
    """
    if not kde_kws:
        kde_kws = {'common_norm': common_norm}

    data_copy = data.copy()
    col_na_name = col_na_fmt.format(col_na)
    data_copy[col_na_name] = data_copy.loc[:, col_na].isna()

    return kdeplot(x=col, hue=col_na_name, data=data_copy, **kde_kws)
