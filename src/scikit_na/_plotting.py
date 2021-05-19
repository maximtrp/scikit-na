from ._na import correlate
from pandas import DataFrame
from pandas.core.indexes.base import Index
from typing import Optional, Union, List
from numpy import arange, array, argwhere, ndarray
from seaborn import heatmap, barplot
from matplotlib.pyplot import subplots


def plot_corr(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        corr_kwargs: dict = {},
        heat_kwargs: dict = {}) -> DataFrame:
    """Plot a correlation heatmap.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Union[List, ndarray, Index]], optional
        Columns names.
    corr_kwargs : dict, optional
        Keyword arguments passed to ``corr()`` method of DataFrame.
    heat_kwargs : dict, optional
        Keyword arguments passed to ``heatmap()`` method of ``seaborn``
        package.
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Heatmap AxesSubplot object.
    """
    cols = array(columns) if columns else data.columns
    values = correlate(data, cols, **corr_kwargs)
    return heatmap(values, **heat_kwargs)


def plot_stats(
        na_info: DataFrame,
        idxstr: str = None,
        idxint: int = None,
        **kwargs):
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
    if not (idxstr or idxint):
        raise ValueError("Error: `idxstr` or `idxint` must be specified")

    if idxstr:
        y_data = na_info.loc[idxstr, :]
    elif idxint:
        y_data = na_info.iloc[idxint, :]
    x_data = na_info.columns

    return barplot(x_data, y_data, **kwargs)


def plot_raw(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        splt_kwargs: dict = {},
        plt_kwargs: dict = {}):
    """NA eventplot. Plots NA values as red lines and normal values
    as black lines.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Union[List, ndarray, Index]], optional
        Columns names.
    splt_kwargs : dict, optional
        Keyword arguments passed to
        :py:meth:`matplotlib.pyplot.subplots` method.
    plt_kwargs : dict, optional
        Keyword arguments passed to
        :py:meth:`matplotlib.axes._subplots.AxesSubplot.eventplot` method.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        AxesSubplot object.
    """
    cols = array(columns) if columns else data.columns
    fig, ax = subplots(**splt_kwargs)
    non_nas = []
    nas = []

    for i, col in enumerate(cols):
        na_mask = data.loc[:, col].isna().values
        non_na = argwhere(~na_mask).ravel()
        na = argwhere(na_mask).ravel()
        non_nas.append(non_na)
        nas.append(na)

    ax.eventplot(
        non_nas, colors=plt_kwargs.get('colors', ['', 'k'])[0], **plt_kwargs)
    ax.eventplot(
        nas, colors=plt_kwargs.get('colors', ['', 'r'])[1], **plt_kwargs)

    if plt_kwargs.get('orientation', None) == 'vertical':
        ax.set_xticks(arange(len(cols)))
        ax.set_xticklabels(cols)
    else:
        ax.set_yticks(arange(len(cols)))
        ax.set_yticklabels(cols)

    return ax
