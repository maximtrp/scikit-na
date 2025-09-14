Predictive Modeling of Missingness
===================================

Understanding what predicts missingness patterns can reveal important insights about
your data collection process and help determine appropriate missing data mechanisms.
**scikit-na** provides logistic regression modeling to predict the probability of
missingness based on other variables.

Why Model Missingness?
~~~~~~~~~~~~~~~~~~~~~~

Modeling missingness helps you:

* **Test missing data mechanisms**: Distinguish between MCAR, MAR, and MNAR
* **Identify predictors**: Understand which variables are associated with missingness
* **Inform imputation**: Use predictive relationships for better imputation strategies
* **Assess bias**: Evaluate potential selection bias in your analysis

The Model
~~~~~~~~~

The ``scikit_na.model()`` function fits a logistic regression where:

* **Dependent variable**: Missing (1) vs. Non-missing (0) in the target column
* **Independent variables**: Other columns that might predict missingness
* **Backend**: Uses `statsmodels <https://www.statsmodels.org>`_ for robust statistical inference

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import scikit_na as na

   # Load the Titanic dataset
   data = pd.read_csv("titanic_dataset.csv")

   # Select numeric predictors
   predictors = ['Fare', 'Parch', 'Pclass', 'SibSp', 'Survived']

   # Fit logistic regression model
   model = na.model(data, col_na='Age', columns=predictors)

   # Display comprehensive results
   print(model.summary())

Interpreting Results
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Extract key information
   print("Model Coefficients:")
   print(model.params)

   print("\\nStatistical Significance:")
   print(model.pvalues)

   print("\\nConfidence Intervals:")
   print(model.conf_int())

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

