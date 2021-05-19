from pandas import concat, DataFrame, Series
from pandas.core.indexes.base import Index
from typing import Union, Optional, List
from numpy import array, ndarray
from functools import partial


def _get_unique_na(data, cols, col):
    return data.dropna(subset=set(cols).difference(col))\
        .loc[:, col].isna().sum()


def _get_rows_after_dropna(data, col):
    return data.dropna(subset=[col]).shape[0]


def _get_abs_na_count(data: DataFrame, cols) -> Series:
    return data.loc[:, cols].isna().sum(axis=0)


def _get_na_perc(data: DataFrame, na_abs: Series) -> Series:
    return na_abs / data.shape[0] * 100


def _get_total_na_count(data: DataFrame, cols) -> int:
    return data.loc[:, cols].isna().sum().sum()


def describe(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        per_column: bool = True):
    """Summary statistics on NA values.

    Parameters
    ----------
    df : DataFrame
        Data object.
    columns : Optional[Union[
            List, numpy.ndarray, pandas.core.indexes.base.Index]] = None
        Columns or indices to observe.
    per_column : bool = True

    Returns
    -------
    DataFrame
        NA descriptive statistics.
    """
    cols = array(columns) if columns else data.columns

    if per_column:
        get_unique_na = partial(_get_unique_na, data, cols)
        get_rows_after_dropna = partial(_get_rows_after_dropna, data)

        na_total = _get_total_na_count(data, cols)
        na_abs_count = _get_abs_na_count(data, cols).rename('NA count')
        na_percentage = _get_na_perc(data, na_abs_count)\
            .rename('NA, % (per column)')
        na_percentage_total = (na_abs_count / na_total * 100)\
            .rename('NA, % (of all NAs)')
        na_unique = Series(
            list(map(get_unique_na, cols)),
            index=cols,
            name='NA unique (per column)')
        na_unique_percentage = (na_unique / na_abs_count * 100)\
            .rename('NA unique, % (per column)')
        rows_after_dropna = Series(
            list(map(get_rows_after_dropna, cols)),
            index=cols,
            name='Rows left after dropna()')
        rows_perc_after_dropna = (rows_after_dropna / data.shape[0] * 100)\
            .rename('Rows, % left after dropna()')
        na_df = concat((
            na_abs_count,
            na_percentage,
            na_percentage_total,
            na_unique,
            na_unique_percentage,
            rows_after_dropna,
            rows_perc_after_dropna), axis=1)
        na_df = na_df.T
    else:
        pass

    return na_df


def correlate(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        **kwargs) -> DataFrame:
    """Calculate correlations between columns in terms of NA values.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[List, ndarray, Index] = None
        Columns names.

    Returns
    -------
    DataFrame
        Correlation values.
    """
    cols = array(columns) if columns else data.columns
    return data.loc[:, cols].isna().corr(**kwargs)
