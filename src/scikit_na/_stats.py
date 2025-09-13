"""Statistical functions."""

__all__ = ["correlate", "describe", "model", "stairs", "summary", "test_hypothesis"]
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from numpy import array, nan, ndarray, r_, setdiff1d
from pandas import NA, DataFrame, Index, Series, concat
from statsmodels.discrete.discrete_model import Logit, BinaryResultsWrapper


def _select_cols(
    data: DataFrame,
    columns: Optional[Iterable[str]] = None,
    second_var: Optional[List[str]] = None,
) -> ndarray:
    return array(
        list(col for col in columns)  # noqa: C400
        if columns is not None
        else (data.columns if second_var is None or len(second_var) == 0 else second_var),
    )


def _get_nominal_cols(data: DataFrame, columns: Optional[Sequence[str]] = None) -> ndarray:
    cols = _select_cols(data, columns)
    return Series(data[cols].dtypes == object).replace({False: NA}).dropna().index.values


def _get_numeric_cols(data: DataFrame, columns: Optional[Sequence[str]] = None) -> ndarray:
    cols = _select_cols(data, columns)
    return Series((data[cols].dtypes == float) | (data[cols].dtypes == int)).replace({False: NA}).dropna().index.values


def _get_unique_na(nas: Series, data: DataFrame, col: str) -> int:
    return (nas & data[col].isna()).sum()


def _get_rows_after_dropna(data: DataFrame, col: Optional[str] = None) -> int:
    return (data.shape[0] - data.loc[:, col].isna().sum()) if col else data.dropna().shape[0]


def _get_rows_after_cum_dropna(data: DataFrame, cols: Optional[List[str]] = None, col: Optional[str] = None) -> int:
    if not cols:
        cols = []
    return data.dropna(subset=(cols + [col] if col else cols)).shape[0]


def _get_abs_na_count(data: DataFrame, cols: Iterable[str]) -> Series:
    return data.loc[:, cols].isna().sum(axis=0)


def _get_na_perc(data: DataFrame, na_abs: Series) -> Series:
    return na_abs / data.shape[0] * 100


def _get_total_na_count(data: DataFrame, cols: Iterable[str]) -> int:
    return data.loc[:, cols].isna().sum().sum()


def summary(
    data: DataFrame,
    columns: Optional[Iterable[str]] = None,
    per_column: bool = True,
    round_dec: int = 2,
) -> DataFrame:
    """Summary statistics on NA values.

    Parameters
    ----------
    data : DataFrame
        Data object.
    columns : Optional[Sequence]
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
    na_by_inst = data_copy.isna().sum(axis=1) == 1
    na_total = _get_total_na_count(data_copy, cols)

    if per_column:
        get_unique_na = partial(_get_unique_na, na_by_inst, data_copy)
        get_rows_after_dropna = partial(_get_rows_after_dropna, data_copy)

        na_abs_count = _get_abs_na_count(data_copy, cols).rename("na_count")
        na_percentage = _get_na_perc(data_copy, na_abs_count).rename("na_pct_per_col")
        na_percentage_total = (na_abs_count / na_total * 100).rename("na_pct_total").fillna(0)
        na_unique = Series(list(map(get_unique_na, cols)), index=cols, name="na_unique_per_col")
        na_unique_percentage = (na_unique / na_abs_count * 100).rename("na_unique_pct_per_col").fillna(0)
        rows_after_dropna = Series(
            list(map(get_rows_after_dropna, cols)),
            index=cols,
            name="rows_after_dropna",
        )
        rows_perc_after_dropna = (rows_after_dropna / data_copy.shape[0] * 100).rename("rows_after_dropna_pct")
        na_df = concat(
            (
                na_abs_count,
                na_percentage,
                na_percentage_total,
                na_unique,
                na_unique_percentage,
                rows_after_dropna,
                rows_perc_after_dropna,
            ),
            axis=1,
        )
        na_df = na_df.T
    else:
        rows_after_dropna = _get_rows_after_dropna(data_copy.loc[:, cols])
        total_cells = data_copy.shape[0] * data_copy.shape[1]
        
        # Handle division by zero for empty datasets
        if total_cells > 0:
            na_percentage_total = na_total / total_cells * 100
            non_na_cells_pct = (total_cells - na_total) / total_cells * 100
        else:
            na_percentage_total = 0.0
            non_na_cells_pct = 0.0
            
        na_col_raw = data_copy.isna().sum()
        na_col_num = na_col_raw[na_col_raw > 0].size
        na_col_only = (na_col_raw == data_copy.shape[0]).sum()
        na_df = DataFrame(
            {
                "total_cols": data_copy.shape[1],
                "na_cols": na_col_num,
                "na_only_cols": na_col_only,
                "total_rows": data_copy.shape[0],
                "na_rows": data_copy.shape[0] - rows_after_dropna,
                "non_na_rows": rows_after_dropna,
                "total_cells": total_cells,
                "na_cells": na_total,
                "na_cells_pct": na_percentage_total,
                "non_na_cells": total_cells - na_total,
                "non_na_cells_pct": non_na_cells_pct,
            },
            index=Index(["dataset"]),
        ).T

    if round_dec:
        na_df = na_df.round(round_dec)

    return na_df


def stairs(
    data: DataFrame,
    columns: Optional[Sequence[str]] = None,
    xlabel: str = "Columns",
    ylabel: str = "Instances",
    tooltip_label: str = "Size difference",
    dataset_label: str = "(Whole dataset)",
) -> DataFrame:
    """DataFrame shrinkage on cumulative :py:meth:`pandas.DataFrame.dropna()`.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Sequence], optional
        Columns names.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    tooltip_label : str, optional
        Tooltip label.
    dataset_label : str, optional
        Label for a whole dataset.

    Returns
    -------
    DataFrame
        Dataset shrinkage results after cumulative
        :py:meth:`pandas.DataFrame.dropna()`.

    """
    cols = _select_cols(data, columns).tolist()
    data_copy = data.loc[:, cols].copy()
    stairs_values = []
    stairs_labels = []

    while len(cols) > 0:
        cols_by_na = data_copy.isna().sum(axis=0).sort_values(ascending=False)
        col_max_na = cols_by_na.head(1)
        col_max_na_name = col_max_na.index.item()
        col_max_na_val = col_max_na.item()
        if not col_max_na_val:
            break
        stairs_values.append(data_copy.shape[0] - col_max_na_val)
        stairs_labels.append(col_max_na_name)
        data_copy = data_copy.dropna(subset=[col_max_na_name])
        cols = data_copy.columns.tolist()

    stairs_values = array([data.shape[0]] + stairs_values)
    stairs_labels = [dataset_label] + stairs_labels
    data_sizes = DataFrame({xlabel: stairs_labels, ylabel: stairs_values})
    data_sizes[tooltip_label] = data_sizes[ylabel].diff().fillna(0)
    return data_sizes


def correlate(data: DataFrame, columns: Optional[Iterable[str]] = None, drop: bool = True, **kwargs: Any) -> DataFrame:
    """Calculate correlations between columns in terms of NA values.

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
    kwargs.setdefault("method", "spearman")
    if drop:
        cols_with_na = data.isna().sum(axis=0).replace({0: nan}).dropna().index.values
        _cols = set(cols).intersection(cols_with_na)
    else:
        _cols = set(cols)
    return data.loc[:, list(_cols)].isna().corr(**kwargs)


def describe(
    data: DataFrame,
    col_na: str,
    columns: Optional[Sequence[str]] = None,
    na_mapping: Optional[Dict[bool, str]] = None,
) -> DataFrame:
    """Describe data grouped by a column with NA values.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col_na : str
        Column with NA values to group the other data by.
    columns : Optional[Sequence]
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

    descr_stats = (
        data.loc[:, list(set(cols).difference([col_na]))]
        .groupby(data[col_na].isna().replace(na_mapping))
        .describe()
        .T.unstack(0)
    )
    return descr_stats.swaplevel(axis=1).sort_index(axis=1, level=0)


def model(
    data: DataFrame,
    col_na: str,
    columns: Optional[Sequence[str]] = None,
    intercept: bool = True,
    fit_kws: Optional[Dict[str, Any]] = None,
    logit_kws: Optional[Dict[str, Any]] = None,
) -> BinaryResultsWrapper:
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
    columns : Optional[Sequence]
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
        data_copy["(intercept)"] = 1.0
        cols_pred = r_[["(intercept)"], cols_pred]

    logit_kws.setdefault("missing", "drop")
    logit_model = Logit(
        endog=data_copy.loc[:, col_na].isna().astype(int),
        exog=data_copy.loc[:, cols_pred],
        **logit_kws,
    )

    return logit_model.fit(**fit_kws)


def test_hypothesis(
    data: DataFrame,
    col_na: str,
    test_fn: Callable[..., Any],
    test_kws: Optional[Dict[str, Any]] = None,
    columns: Optional[Union[Iterable[str], Dict[str, Callable[..., Any]]]] = None,
    dropna: bool = True,
) -> Dict[str, Any]:
    """Test a statistical hypothesis.

    This function can be used to find evidence against missing
    completely at random (MCAR) mechanism by comparing two samples grouped
    by missingness in another column.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col_na : str
        Column to group values by. :py:meth:`pandas.Series.isna()` method
        is applied before grouping.
    columns : Optional[Union[Sequence[str], Dict[str, callable]]]
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

    elif isinstance(columns, Sequence):
        for col in columns:
            result = test_fn(*_get_groups(data, groups, col, dropna), **test_kws)
            results[col] = result

    return results
