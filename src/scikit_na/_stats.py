"""Statistical functions."""
__all__ = [
    'describe', 'summary', 'correlate', 'model', 'test_hypothesis']
from functools import partial
from typing import Union, Optional, Dict, Iterable
from pandas import concat, DataFrame, Series, NA
from numpy import array, ndarray, nan, r_, setdiff1d
from statsmodels.discrete.discrete_model import Logit


def _select_cols(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        second_var: list = None) -> ndarray:
    return array(
        list(col for col in columns)
        if columns is not None
        else (data.columns if second_var is None else second_var))


def _get_nominal_cols(
        data: DataFrame,
        columns: Optional[Iterable] = None):
    cols = _select_cols(data, columns)
    return (data[cols].dtypes == object)\
        .replace({False: NA})\
        .dropna()\
        .index.values


def _get_numeric_cols(
        data: DataFrame,
        columns: Optional[Iterable] = None):
    cols = _select_cols(data, columns)
    return ((data[cols].dtypes == float)
            | (data[cols].dtypes == int))\
        .replace({False: NA})\
        .dropna()\
        .index.values


def _get_unique_na(data, cols, col) -> int:
    return data.dropna(subset=set(cols).difference([col]))\
        .loc[:, col].isna().sum()


def _get_rows_after_dropna(data, col=None) -> int:
    return data.dropna(subset=([col] if col else None)).shape[0]


def _get_rows_after_cum_dropna(
        data, cols: list = None, col: str = None) -> int:
    if not cols:
        cols = []
    return data.dropna(subset=(cols + [col] if col else cols)).shape[0]


def _get_abs_na_count(data: DataFrame, cols) -> Series:
    return data.loc[:, cols].isna().sum(axis=0)


def _get_na_perc(
        data: DataFrame,
        na_abs: Union[Series, int]) -> Series:
    return na_abs / data.shape[0] * 100


def _get_total_na_count(data: DataFrame, cols) -> int:
    return data.loc[:, cols].isna().sum().sum()


def summary(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        per_column: bool = True,
        round_dec: int = 2) -> DataFrame:
    """
    Summary statistics on NA values.

    Parameters
    ----------
    data : DataFrame
        Data object.
    columns : Optional[Iterable]
        Columns or indices to observe.
    per_column : bool = True, optional
        Show stats per each selected column.
    round_dec: int = 2, optional
        Number of decimals for rounding.

    Returns
    -------
    DataFrame
        Summary on NA values in the input data.
    """
    cols = _select_cols(data, columns)
    data_copy = data.loc[:, cols].copy()
    na_total = _get_total_na_count(data_copy, cols)

    if per_column:
        get_unique_na = partial(_get_unique_na, data_copy, cols)
        get_rows_after_dropna = partial(_get_rows_after_dropna, data_copy)

        na_abs_count = _get_abs_na_count(data_copy, cols).rename('NA count')
        na_percentage = _get_na_perc(data_copy, na_abs_count)\
            .rename('NA, % (per column)')
        na_percentage_total = (na_abs_count / na_total * 100)\
            .rename('NA, % (of all NAs)').fillna(0)
        na_unique = Series(
            list(map(get_unique_na, cols)),
            index=cols,
            name='NA unique (per column)')
        na_unique_percentage = (na_unique / na_abs_count * 100)\
            .rename('NA unique, % (per column)').fillna(0)
        rows_after_dropna = Series(
            list(map(get_rows_after_dropna, cols)),
            index=cols,
            name='Rows left after dropna()')
        rows_perc_after_dropna = (
            rows_after_dropna / data_copy.shape[0] * 100)\
            .rename('Rows left after dropna(), %')
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
        rows_after_dropna = _get_rows_after_dropna(data_copy.loc[:, cols])
        na_percentage_total = na_total / data_copy.shape[0]\
            / data_copy.shape[1] * 100
        total_cells = data_copy.shape[0] * data_copy.shape[1]
        na_df = DataFrame({
            'Total columns': data_copy.shape[1],
            'Total rows': data_copy.shape[0],
            'Rows with NA': data_copy.shape[0] - rows_after_dropna,
            'Rows without NA': rows_after_dropna,
            'Total cells': total_cells,
            'Cells with NA': na_total,
            'Cells with NA, %': na_percentage_total,
            'Cells with non-missing data':
                total_cells - na_total,
            'Cells with non-missing data, %':
                (total_cells - na_total) / total_cells * 100},
            index=['dataset']
        ).T

    if round_dec:
        na_df = na_df.round(round_dec)

    return na_df


def correlate(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        drop: bool = True,
        **kwargs) -> DataFrame:
    """
    Calculate correlations between columns in terms of NA values.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[List, ndarray, Index] = None
        Columns names.
    drop : bool = True, optional
        Drop columns without NA values.
    kwargs : dict, optional
        Keyword arguments passed to :py:meth:`pandas.DataFrame.corr()` method.

    Returns
    -------
    DataFrame
        Correlation values.
    """
    cols = _select_cols(data, columns)
    kwargs.setdefault('method', 'spearman')
    if drop:
        cols_with_na = data.isna().sum(axis=0).replace({0: nan})\
            .dropna().index.values
        _cols = set(cols).intersection(cols_with_na)
    else:
        _cols = set(cols)
    return data.loc[:, _cols].isna().corr(**kwargs)


def describe(
        data: DataFrame,
        col_na: str,
        columns: Optional[Iterable] = None,
        na_mapping: dict = None) -> DataFrame:
    """
    Describe data grouped by a column with NA values.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col_na : str
        Column with NA values to group the other data by.
    columns : Optional[Iterable]
        Columns to calculate descriptive statistics on.
    na_mapping : dict, optional
        Dictionary with NA mappings. By default,
        it is {True: "NA", False: "Filled"}.

    Returns
    -------
    DataFrame
        Descriptive statistics (mean, median, etc.).
    """
    if not na_mapping:
        na_mapping = {True: "NA", False: "Filled"}
    cols = _select_cols(data, columns).tolist()

    descr_stats = data.loc[:, set(cols).difference([col_na])]\
        .groupby(data[col_na].isna().replace(na_mapping))\
        .describe()\
        .T\
        .unstack(0)
    return descr_stats.swaplevel(axis=1).sort_index(axis=1, level=0)


def model(
        data: DataFrame,
        col_na: str,
        columns: Optional[Iterable] = None,
        intercept: bool = True,
        fit_kws: dict = None,
        logit_kws: dict = None):
    """Logistic regression modeling.

    Fit a logistic regression model to NA values encoded as 0 (non-missing)
    and 1 (NA) in column `col_na` with predictors passed with `columns`
    argument. Statsmodels package is used as a backend for model fitting.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col_na : str
        Column with NA values to use as a dependent variable.
    columns : Optional[Iterable]
        Columns to use as independent variables.
    intercept : bool, optional
        Fit intercept.
    fit_kws : dict, optional
        Keyword arguments passed to `fit()` method of model.
    logit_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`statsmodels.discrete.discrete_model.Logit` class.

    Returns
    -------
    statsmodels.discrete.discrete_model.BinaryResultsWrapper
        Model after applying `fit` method.

    Example
    -------
    >>> import scikit_na as na
    >>> model = na.model(
    ...     data,
    ...     col_na='column_with_NAs',
    ...     columns=['age', 'height', 'weight'])
    >>> model.summary()
    """
    cols = _select_cols(data, columns)
    cols_pred = setdiff1d(cols, [col_na])
    data_copy = data.loc[:, cols_pred.tolist() + [col_na]].copy()

    if not fit_kws:
        fit_kws = {}
    if not logit_kws:
        logit_kws = {}

    if intercept:
        data_copy['(intercept)'] = 1.
        cols_pred = r_[['(intercept)'], cols_pred]

    logit_kws.setdefault('missing', 'drop')
    logit_model = Logit(
        endog=data_copy.loc[:, col_na].isna().astype(int),
        exog=data_copy.loc[:, cols_pred],
        **logit_kws)

    return logit_model.fit(**fit_kws)


def test_hypothesis(
        data: DataFrame,
        col_na: str,
        test_fn: callable,
        test_kws: dict = None,
        columns: Optional[Union[Iterable[str], Dict[str, callable]]] = None,
        dropna: bool = True) -> Dict[str, object]:
    """Test a statistical hypothesis.

    Typically, can be used to compare
    two samples grouped by NA/non-NA values in another column.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col_na : str
        Column to group values by. :py:meth:`pandas.Series.isna()` method
        is applied before grouping.
    columns : Optional[Union[Iterable[str], Dict[str, callable]]]
        Columns to test hypotheses on.
    test_fn : callable, optional
        Function to test hypothesis on NA/non-NA data.
        Must be a two-sample test function that accepts two arrays
        and (optionally) keyword arguments such as
        :py:meth:`scipy.stats.mannwhitneyu`.
    test_kws : dict, optional
        Keyword arguments passed to `test_fn` function.
    dropna: bool = True, optional
        Drop NA values in two samples before running a hypothesis test.

    Returns
    -------
    Dict[str, object]
        Dictionary with tests results as `column` => test function output.

    Example
    -------
    >>> import scikit_na as na
    >>> import pandas as pd
    >>> data = pd.read_csv('some_dataset.csv')
    >>> # Simple example
    >>> na.test_hypothesis(
    ...     data,
    ...     col_na='some_column_with_NAs',
    ...     columns=['age', 'height', 'weight'],
    ...     test_fn=ss.mannwhitneyu)

    >>> # Example with `columns` as a dictionary of column => function pairs
    >>> from functools import partial
    >>> import scipy.stats as st
    >>> # Passing keyword arguments to functions
    >>> kstest_mod = partial(st.kstest, N=100)
    >>> mannwhitney_mod = partial(st.mannwhitneyu, use_continuity=False)
    >>> # Running tests
    >>> results = na.test_hypothesis(
    ...     data,
    ...     col_na='some_column_with_NAs',
    ...     columns={
    ...         'age': kstest_mod,
    ...         'height': mannwhitney_mod,
    ...         'weight': mannwhitney_mod})
    >>> pd.DataFrame(results, index=['statistic', 'p-value'])
    """
    def _get_groups(data, groups, col, dropna=True):
        grp1 = data.loc[groups[False], col]
        grp2 = data.loc[groups[True], col]
        if dropna:
            grp1 = grp1.dropna()
            grp2 = grp2.dropna()
        return grp1, grp2

    # Grouping data by NA/non-NA values in `col_na`
    groups = data.groupby(data[col_na].isna()).groups

    # Initializing
    results = {}
    if not test_kws:
        test_kws = {}

    # Selecting columns with data to test hypothesis on
    if columns is None:
        columns = setdiff1d(data.columns, [col_na])

    # Iterating over columns or column => func pairs and testing hypotheses
    if isinstance(columns, dict):
        for col, func in columns.items():
            result = func(*_get_groups(data, groups, col, dropna))
            results[col] = result

    elif isinstance(columns, Iterable):
        for col in columns:
            result = test_fn(
                *_get_groups(data, groups, col, dropna), **test_kws)
            results[col] = result

    return results
