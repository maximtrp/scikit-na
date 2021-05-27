from ._na import _get_rows_after_cum_dropna, correlate
from pandas import DataFrame
from pandas.core.indexes.base import Index
from typing import Optional, Union, List
from numpy import (
    arange, argmin, argmax, array, ndarray, fill_diagonal, nan,
    insert, diff)
from seaborn import heatmap, barplot, diverging_palette
from matplotlib.pyplot import subplots
from matplotlib.axes import SubplotBase
from matplotlib.patches import Patch
from functools import partial


def plot_corr(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        mask_diag: bool = True,
        corr_kws: dict = {},
        heat_kws: dict = {}) -> SubplotBase:
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
        Keyword arguments passed to ``corr()`` method of DataFrame.
    heat_kws : dict, optional
        Keyword arguments passed to ``heatmap()`` method of ``seaborn``
        package.
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Heatmap AxesSubplot object.
    """
    cols = array(columns) if columns is not None else data.columns

    corr_kws.setdefault('method', 'spearman')
    data_corr = correlate(data, cols, **corr_kws)
    if mask_diag:
        fill_diagonal(data_corr.values, nan)

    heat_kws.setdefault('vmin', -1)
    heat_kws.setdefault('vmax', 1)
    heat_kws.setdefault('annot', True)
    heat_kws.setdefault('square', True)
    heat_kws.setdefault('cmap', diverging_palette(240, 10, as_cmap=True))

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
        Keyword arguments passed to :py:meth:`seaborn.barplot` method.

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

    return barplot(x_data, y_data, **kwargs)


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
        columns: Optional[Union[List, ndarray, Index]] = None,
        droppable: bool = True,
        sort: bool = True,
        cmap: Optional[Union[List, ndarray]] = ['green', 'orange', 'red'],
        names: Optional[Union[List, ndarray]] = ['Filled', 'Droppable', 'NA'],
        yaxis: bool = False,
        xaxis: bool = True,
        legend_kws: dict = {},
        sb_kws: dict = {}) -> SubplotBase:
    """NA heatmap. Plots NA values as red lines and normal values
    as black lines.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Union[List, ndarray, Index]], optional
        Columns names.
    droppable : bool, optional
        Show values to be dropped by :py:meth:`pandas.DataFrame.dropna` method.
    sort : bool, optional
        Sort DataFrame by selected columns.
    cmap : Optional[Union[List, ndarray]], optional
        Heatmap and legend colormap: non-missing values, droppable values,
        NA values, correspondingly.
    names : Optional[Union[List, ndarray]], optional
        Legend labels: non-missing values, droppable values,
        NA values, correspondingly.
    yaxis : bool, optional
        Show Y axis.
    xaxis : bool, optional
        Show X axis.
    legend_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`matplotlib.axes._subplots.AxesSubplot` method.
    sb_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`seaborn.heatmap` method.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        AxesSubplot object.
    """
    cols = array(columns) if columns is not None else data.columns.tolist()
    data_copy = data.loc[:, cols].sort_values(by=cols).copy()\
        if sort else data.loc[:, cols].copy()
    data_na = data_copy.isna()
    if droppable:
        non_na_mask = ~data_na.values
        na_rows_mask = data_na.any(axis=1).values[:, None]
        droppable_mask = non_na_mask & na_rows_mask
        data_na = data_na.astype(float)
        data_na.values[droppable_mask] = 0.5
        labels = names
    else:
        labels = [names[0], names[-1]]

    sb_kws.setdefault('cbar', False)
    legend_kws.setdefault('loc', 'upper center')
    legend_kws.setdefault('ncol', len(labels))
    legend_kws.setdefault('bbox_to_anchor', (0.5, 1.15))

    ax = heatmap(data_na, cmap=cmap, **sb_kws)
    ax.yaxis.set_visible(yaxis)
    ax.xaxis.set_visible(xaxis)
    legend_elements = [Patch(facecolor=cmap[0]), Patch(facecolor=cmap[-1])]
    if droppable:
        legend_elements.insert(1, Patch(facecolor=cmap[1]))
    ax.legend(legend_elements, labels, **legend_kws)

    return ax


def plot_stairs(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        frame: bool = False,
        sort: Optional[Union["min", "max", None]] = "min",
        splt_kws: dict = {},
        stairs_kws: dict = {},
        text_kws: dict = {}) -> SubplotBase:
    """Stairs plot of cumulative changes in rows number
    after applying :py:meth:`pandas.DataFrame.dropna()` method.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Union[List, ndarray, Index]], optional
        Column names.
    frame : bool, optional
        Draw axes frame.
    sort : Optional[Union["min", "max", None]], optional
        Sort columns by maximum number of NA values.
    splt_kws : dict, optional
        Keyword arguments passed to :py:meth:`matplotlib.pyplot.subplots`.
    stairs_kws : dict, optional
        Keyword arguments passed to :py:meth:`matplotlib.pyplot.stairs`.
    text_kws : dict, optional
        Keyword arguments passed to :py:meth:`matplotlib.pyplot.text`.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        AxesSubplot object.
    """
    cols = array(columns)\
        if columns is not None else data.columns.values
    _cols = cols.copy().tolist()
    stairs_values = []
    stairs_labels = []

    if sort == 'min':
        get_func = min
        arg_func = argmin
    elif sort == 'max':
        get_func = max
        arg_func = argmax
    else:
        get_func = lambda x: x[0]
        arg_func = lambda _: 0

    while len(_cols) > 0:
        get_rows = partial(
            _get_rows_after_cum_dropna, data, stairs_labels)
        rows_after_dropna = list(map(get_rows, _cols))
        stairs_values.append(get_func(rows_after_dropna))
        stairs_labels.append(_cols[arg_func(rows_after_dropna)])
        _cols.remove(_cols[arg_func(rows_after_dropna)])

    stairs_values = [data.shape[0]] + stairs_values
    stairs_labels = ["Whole dataset"] + stairs_labels

    stairs_kws.setdefault('fill', True)
    stairs_kws.setdefault('linewidth', 2)
    stairs_kws.setdefault('ec', 'black')
    text_kws.setdefault('ha', 'center')

    fig, ax = subplots(**splt_kws)
    ax.stairs(values=stairs_values, **stairs_kws)
    stairs_values_diff = (diff(stairs_values) * -1)
    stairs_values_diff = insert(
        stairs_values_diff, 0, stairs_values_diff.mean())
    labels_y = stairs_values + stairs_values_diff.mean() * 0.2
    labels_x = arange(len(cols)+1) + 0.5

    for text, x, y in zip(stairs_values, labels_x, labels_y):
        ax.text(x, y, text, **text_kws)
    ax.set_ylim(0, max(labels_y))
    ax.yaxis.set_visible(False)
    ax.set_xticks(labels_x)
    ax.set_xticklabels(stairs_labels)
    ax.set_frame_on(False)
    fig.tight_layout()

    return ax
