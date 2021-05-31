__all__ = ['model']
from pandas import DataFrame, Index
from statsmodels.discrete.discrete_model import Logit
from typing import Optional, Union, List
from numpy import array, ndarray, r_


def model(
        data: DataFrame,
        col_na: str,
        columns: Optional[Union[List, ndarray, Index]] = None,
        intercept: bool = True,
        fit_kws: dict = {},
        logit_kws: dict = {}):
    """Fit a logistic regression model to NA values encoded as 0 (non-missing)
    and 1 (NA) in column `col_na` with predictors passed with `columns`
    argument. Statsmodels package is used as a backend for model fitting.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col_na : str
        Column with NA values to use as a dependent variable.
    columns : Optional[Union[List, ndarray, Index]], optional
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
    """
    cols = array(columns)\
        if columns is not None else data.columns.to_numpy(copy=True)
    data_copy = data.loc[:, cols.tolist() + [col_na]].copy()

    if intercept:
        data_copy['(intercept)'] = 1.
        cols = r_[['(intercept)'], cols]

    logit_kws.setdefault('missing', 'drop')
    logit_model = Logit(
        endog=data_copy.loc[:, col_na].isna().astype(int),
        exog=data_copy.loc[:, cols],
        **logit_kws)

    return logit_model.fit(**fit_kws)
