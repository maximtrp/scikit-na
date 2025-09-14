"""Statistical functions."""

from __future__ import annotations

__all__ = ["correlate", "describe", "model", "stairs", "summary", "test_hypothesis"]

from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any, Callable, Dict, List

from numpy import array, nan, ndarray, r_, setdiff1d
from pandas import NA, DataFrame, Index, Series, concat
from statsmodels.discrete.discrete_model import BinaryResultsWrapper, Logit


def _select_cols(
    data: DataFrame,
    columns: Iterable[str] | None = None,
    second_var: List[str] | None = None,
) -> ndarray:
    return array(
        list(col for col in columns)  # noqa: C400
        if columns is not None
        else (data.columns if second_var is None or len(second_var) == 0 else second_var),
    )


def _get_nominal_cols(data: DataFrame, columns: Sequence[str] | None = None) -> ndarray:
    cols = _select_cols(data, columns)
    return Series(data[cols].dtypes == object).replace({False: NA}).dropna().index.values


def _get_numeric_cols(data: DataFrame, columns: Sequence[str] | None = None) -> ndarray:
    cols = _select_cols(data, columns)
    return Series((data[cols].dtypes == float) | (data[cols].dtypes == int)).replace({False: NA}).dropna().index.values


def _get_unique_na(nas: Series, data: DataFrame, col: str) -> int:
    return (nas & data[col].isna()).sum()


def _get_rows_after_dropna(data: DataFrame, col: str | None = None) -> int:
    return (data.shape[0] - data.loc[:, col].isna().sum()) if col else data.dropna().shape[0]


def _get_rows_after_cum_dropna(data: DataFrame, cols: List[str] | None = None, col: str | None = None) -> int:
    if not cols:
        cols = []
    return data.dropna(subset=([*cols, col] if col else cols)).shape[0]


def _get_abs_na_count(data: DataFrame, cols: Iterable[str]) -> Series:
    return data.loc[:, cols].isna().sum(axis=0)


def _get_na_perc(data: DataFrame, na_abs: Series) -> Series:
    return na_abs / data.shape[0] * 100


def _get_total_na_count(data: DataFrame, cols: Iterable[str]) -> int:
    return data.loc[:, cols].isna().sum().sum()


def summary(
    data: DataFrame,
    columns: Iterable[str] | None = None,
    per_column: bool = True,
    round_dec: int = 2,
) -> DataFrame:
    """Generate comprehensive summary statistics for missing data patterns.

    Computes detailed statistics about missing values including counts, percentages,
    and the impact of missing data on dataset completeness. This function provides
    both per-column and aggregate statistics to help understand missing data patterns.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame to analyze for missing data patterns.
    columns : Iterable[str], optional
        Specific column names to analyze. If None, analyzes all columns.
    per_column : bool, default True
        If True, returns detailed statistics for each column individually.
        If False, returns aggregate statistics for the entire dataset.
    round_dec : int, default 2
        Number of decimal places for rounding numerical results.

    Returns
    -------
    DataFrame
        Summary statistics with the following metrics:

        When per_column=True:
        - na_count: Absolute count of missing values per column
        - na_pct_per_col: Percentage of missing values per column
        - na_pct_total: Percentage of column's NAs relative to all NAs
        - na_unique_per_col: Count of rows where only this column has NA
        - na_unique_pct_per_col: Percentage of unique NAs for this column
        - rows_after_dropna: Remaining rows after dropping NAs from this column
        - rows_after_dropna_pct: Percentage of rows remaining after dropna

        When per_column=False:
        - Dataset-level aggregated statistics including total cells,
          missing cells, and overall completion rates

    Examples
    --------
    Basic usage with per-column statistics:

    >>> import pandas as pd
    >>> import scikit_na as na
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, None, 4, 5],
    ...     'B': [None, 2, 3, None, 5],
    ...     'C': [1, 2, 3, 4, 5]
    ... })
    >>> na.summary(data)
                        A    B    C
    na_count         1.0  2.0  0.0
    na_pct_per_col  20.0 40.0  0.0
    na_pct_total    33.3 66.7  0.0
    ...

    Dataset-level summary:

    >>> na.summary(data, per_column=False)
                        dataset
    total_cols              3.0
    na_cols                 2.0
    total_rows              5.0
    na_cells                3.0
    na_cells_pct           20.0
    ...

    Analyzing specific columns only:

    >>> na.summary(data, columns=['A', 'B'])
                        A    B
    na_count         1.0  2.0
    na_pct_per_col  20.0 40.0
    ...

    Notes
    -----
    - Empty datasets (zero total cells) are handled gracefully with 0.0% rates
    - The function provides insights into both individual column missingness
      and the overall impact on dataset completeness
    - Use per_column=False for quick dataset-level overview
    - Use per_column=True for detailed column-by-column analysis
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
    columns: Sequence[str] | None = None,
    xlabel: str = "Columns",
    ylabel: str = "Instances",
    tooltip_label: str = "Size difference",
    dataset_label: str = "(Whole dataset)",
) -> DataFrame:
    """Analyze dataset shrinkage from cumulative column-wise dropna operations.

    This function simulates the effect of sequentially applying pandas.DataFrame.dropna()
    to individual columns, starting with the column that has the most missing values.
    It shows how the dataset size decreases as columns are processed, helping to
    understand the cumulative impact of missing data on analysis sample sizes.

    The algorithm:
    1. Identifies the column with the most missing values
    2. Applies dropna() to that column and records remaining dataset size
    3. Repeats with remaining columns until no more missing values exist
    4. Returns results suitable for visualization as a "stairs" plot

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame to analyze for missing data impact.
    columns : Sequence[str], optional
        Specific column names to include in the analysis. If None, uses all columns.
    xlabel : str, default "Columns"
        Label for the x-axis in resulting visualization data.
    ylabel : str, default "Instances"
        Label for the y-axis representing the number of remaining rows.
    tooltip_label : str, default "Size difference"
        Label for the difference column showing row loss at each step.
    dataset_label : str, default "(Whole dataset)"
        Label for the initial state before any dropna operations.

    Returns
    -------
    DataFrame
        Analysis results with columns:
        - {xlabel}: Column names in order of processing (most missing first)
        - {ylabel}: Number of remaining rows after processing each column
        - {tooltip_label}: Number of rows lost at each step (negative values)

        The first row represents the original dataset size before any processing.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> import scikit_na as na
    >>> data = pd.DataFrame({
    ...     'A': [1, None, 3, None, 5],      # 2 missing
    ...     'B': [1, 2, None, 4, 5],        # 1 missing
    ...     'C': [None, None, None, 4, 5]   # 3 missing (most)
    ... })
    >>> na.stairs(data)
          Columns  Instances  Size difference
    0   (Whole dataset)      5              0.0
    1              C         2             -3.0
    2              A         2              0.0
    3              B         2              0.0

    Analyzing specific columns:

    >>> na.stairs(data, columns=['A', 'B'])
          Columns  Instances  Size difference
    0   (Whole dataset)      5              0.0
    1              A         3             -2.0
    2              B         3              0.0

    Custom labels for visualization:

    >>> na.stairs(data,
    ...           xlabel="Features",
    ...           ylabel="Sample Size",
    ...           tooltip_label="Rows Lost")
        Features  Sample Size  Rows Lost
    0   (Whole dataset)       5        0.0
    1              C          2       -3.0
    2              A          2        0.0
    3              B          2        0.0

    Notes
    -----
    - Columns are processed in order of decreasing missing value count
    - This analysis helps identify which columns contribute most to sample size reduction
    - Results are particularly useful for creating "stairs" or "waterfall" visualizations
    - The cumulative approach shows realistic impact when using listwise deletion
    - Use with plot_stairs() for visual representation of the analysis
    - Useful for deciding column inclusion/exclusion strategies in analysis pipelines
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

    stairs_values = array([data.shape[0], *stairs_values])
    stairs_labels = [dataset_label, *stairs_labels]
    data_sizes = DataFrame({xlabel: stairs_labels, ylabel: stairs_values})
    data_sizes[tooltip_label] = data_sizes[ylabel].diff().fillna(0)
    return data_sizes


def correlate(data: DataFrame, columns: Iterable[str] | None = None, drop: bool = True, **kwargs: Any) -> DataFrame:
    """Calculate correlations between missing value patterns across columns.

    Computes correlation coefficients between the missing value indicators (True/False)
    of different columns to identify relationships in missingness patterns. High
    correlations suggest that certain columns tend to have missing values together,
    which can indicate systematic data collection issues or missing data mechanisms.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame to analyze for missing data correlations.
    columns : Iterable[str], optional
        Column names to include in correlation analysis. If None, uses all columns.
    drop : bool, default True
        If True, excludes columns that have no missing values from the analysis.
        If False, includes all specified columns (columns with no NAs will have
        correlations of 0 or NaN with other columns).
    **kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.corr().
        Common options include:
        - method : {'pearson', 'kendall', 'spearman'}, default 'spearman'
        - min_periods : int, minimum number of observations for valid result

    Returns
    -------
    DataFrame
        Correlation matrix showing relationships between missing value patterns.
        Values range from -1 to 1, where:
        - 1 indicates perfect positive correlation (columns miss together)
        - 0 indicates no correlation
        - -1 indicates perfect negative correlation (one misses when other doesn't)

    Examples
    --------
    Basic correlation analysis:

    >>> import pandas as pd
    >>> import scikit_na as na
    >>> data = pd.DataFrame({
    ...     'income': [50000, None, None, 80000, None],
    ...     'bonus': [5000, None, None, 8000, None],
    ...     'age': [25, 30, None, 35, 40],
    ...     'complete': [1, 2, 3, 4, 5]  # no missing values
    ... })
    >>> na.correlate(data)
            income  bonus   age
    income     1.0   1.0   0.5
    bonus      1.0   1.0   0.5
    age        0.5   0.5   1.0

    Include columns without missing values:

    >>> na.correlate(data, drop=False)
            income  bonus   age  complete
    income     1.0   1.0   0.5       NaN
    bonus      1.0   1.0   0.5       NaN
    age        0.5   0.5   1.0       NaN
    complete   NaN   NaN   NaN       NaN

    Using different correlation methods:

    >>> # Pearson correlation (linear relationships)
    >>> na.correlate(data, method='pearson')

    >>> # Kendall's tau (rank-based, robust to outliers)
    >>> na.correlate(data, method='kendall')

    Analyzing specific columns only:

    >>> na.correlate(data, columns=['income', 'bonus', 'age'])

    Notes
    -----
    - Default correlation method is 'spearman' (rank-based), which is often
      appropriate for binary missing value indicators
    - Perfect correlation (1.0) between columns suggests they share the same
      missing data pattern, possibly due to systematic data collection issues
    - This analysis helps identify Missing At Random (MAR) vs Missing Completely
      At Random (MCAR) patterns
    - Use with visualization functions like plot_corr() for easier interpretation
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
    columns: Sequence[str] | None = None,
    na_mapping: Dict[bool, str] | None = None,
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
    columns: Sequence[str] | None = None,
    intercept: bool = True,
    fit_kws: Dict[str, Any] | None = None,
    logit_kws: Dict[str, Any] | None = None,
) -> BinaryResultsWrapper:
    """Fit logistic regression model to predict missing data patterns.

    Creates a logistic regression model where the dependent variable indicates
    whether a value is missing (1) or not (0) in the specified column, using
    other variables as predictors. This is useful for:
    - Understanding factors associated with missingness
    - Testing Missing at Random (MAR) vs Missing Not at Random (MNAR) mechanisms
    - Predicting probability of missingness for imputation or weighting

    The model uses statsmodels' Logit class and automatically handles missing
    values in predictors through the 'missing="drop"' option.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame containing the data to analyze.
    col_na : str
        Column name containing missing values to model as the dependent variable.
        Missing values in this column become 1, non-missing become 0.
    columns : Sequence[str], optional
        Column names to use as independent variables (predictors). If None,
        uses all columns except col_na. Columns with missing values are handled
        automatically by dropping incomplete cases.
    intercept : bool, default True
        If True, includes an intercept term in the model. Recommended for
        most analyses to properly estimate baseline probability.
    fit_kws : dict, optional
        Additional keyword arguments passed to the model's fit() method.
        Common options include:
        - method: 'newton' (default), 'bfgs', 'lbfgs'
        - disp: bool, whether to display convergence messages
        - maxiter: maximum number of iterations
    logit_kws : dict, optional
        Additional keyword arguments passed to statsmodels.discrete.discrete_model.Logit.
        The 'missing' parameter defaults to 'drop' to handle missing predictors.

    Returns
    -------
    statsmodels.discrete.discrete_model.BinaryResultsWrapper
        Fitted logistic regression model with methods for:
        - summary(): Detailed model statistics and coefficients
        - predict(): Predicted probabilities of missingness
        - params: Model coefficients
        - pvalues: Statistical significance of predictors

    Examples
    --------
    Basic logistic regression for missing income data:

    >>> import pandas as pd
    >>> import scikit_na as na
    >>> data = pd.DataFrame({
    ...     'income': [50000, None, 75000, None, 90000, None, 60000],
    ...     'age': [25, 30, 35, 40, 45, 50, 28],
    ...     'education_years': [12, 16, 18, 14, 20, 16, 12],
    ...     'urban': [1, 1, 0, 1, 0, 1, 1]
    ... })
    >>> model = na.model(data, col_na='income',
    ...                  columns=['age', 'education_years', 'urban'])
    >>> print(model.summary())

    Examining model coefficients and significance:

    >>> # Get coefficient estimates
    >>> print("Coefficients:")
    >>> print(model.params)
    >>>
    >>> # Check statistical significance
    >>> print("P-values:")
    >>> print(model.pvalues)
    >>>
    >>> # Predict missingness probabilities
    >>> probabilities = model.predict()

    Advanced model fitting with custom options:

    >>> # Use different optimization method and custom convergence settings
    >>> model = na.model(
    ...     data,
    ...     col_na='income',
    ...     columns=['age', 'education_years'],
    ...     fit_kws={'method': 'bfgs', 'maxiter': 100, 'disp': True},
    ...     logit_kws={'check_rank': False}
    ... )

    Model without intercept (for special cases):

    >>> # When theoretical reasons suggest no baseline probability
    >>> model = na.model(data, col_na='income',
    ...                  columns=['age'], intercept=False)

    Notes
    -----
    - The model automatically converts missing values in col_na to 1 and non-missing to 0
    - Cases with missing values in predictor columns are dropped (listwise deletion)
    - Significant coefficients suggest Missing at Random (MAR) conditional on predictors
    - Non-significant model suggests Missing Completely at Random (MCAR)
    - Use model.summary() for comprehensive output including fit statistics
    - Consider checking model assumptions (linearity, independence, no multicollinearity)
    - Large coefficients may indicate separation issues requiring regularization

    See Also
    --------
    test_hypothesis : Statistical tests for missing data mechanisms
    describe : Descriptive statistics grouped by missingness
    correlate : Correlation analysis of missing data patterns
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
    test_kws: Dict[str, Any] | None = None,
    columns: Iterable[str] | Dict[str, Callable[..., Any]] | None = None,
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
