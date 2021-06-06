Regression modeling
===================

The presence of NA values can be used in regression modeling as a dependent
variable encoded as ``0`` and ``1``. Currently, ``scikit_na.model()`` method
runs a logistic model using `statsmodels <https://www.statsmodels.org>`_ package
as a backend.

For demonstration purposes, we will use `Titanic dataset
<https://www.kaggle.com/c/titanic/data>`_. Let's create a regression model with *Age* as a
dependent variable and *Fare*, *Parch*, *Pclass*, *SibSp*, *Survived*
as independent variables. Internally, ``pandas.Series.isna()`` method is called
on *Age* column, and the resulting boolean values are converted to integers
(``True`` and ``False`` become ``1`` and ``0``). Data preprocessing is totally up to
you!

.. code:: python

    import pandas as pd
    import scikit_na as na

    # Loading data
    data = pd.read_csv("titanic_dataset.csv")

    # Selecting columns with numeric data
    # Dropping "PassengerId" column
    subset = data.loc[:, data.dtypes != object].drop(columns=['PassengerId'])

    # Fitting a model
    model = na.model(subset, col_na='Age')
    model.summary()

.. code::

    Optimization terminated successfully.
    Current function value: 0.467801
    Iterations 7
                            Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                    Age   No. Observations:                  891
    Model:                          Logit   Df Residuals:                      885
    Method:                           MLE   Df Model:                            5
    Date:                Sat, 05 Jun 2021   Pseudo R-squ.:                 0.06164
    Time:                        17:51:31   Log-Likelihood:                -416.81
    converged:                       True   LL-Null:                       -444.19
    Covariance Type:            nonrobust   LLR p-value:                 1.463e-10
    ===============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    (intercept)    -2.7294      0.429     -6.369      0.000      -3.569      -1.890
    Fare            0.0010      0.003      0.376      0.707      -0.004       0.006
    Parch          -0.8874      0.223     -3.984      0.000      -1.324      -0.451
    Pclass          0.5953      0.147      4.046      0.000       0.307       0.884
    SibSp           0.2548      0.095      2.684      0.007       0.069       0.441
    Survived       -0.1026      0.198     -0.519      0.604      -0.490       0.285
    ===============================================================================

