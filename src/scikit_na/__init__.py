"""scikit-na: Comprehensive missing data analysis toolkit for Python.

This package provides statistical functions, interactive visualizations, and
export utilities for analyzing missing data patterns in pandas DataFrames.
It helps researchers and data scientists understand the structure and impact
of missing values through intuitive statistics, correlations, and visual representations.

Key features:
- Statistical summaries of missing data patterns
- Interactive visualizations using Altair and Matplotlib
- Hypothesis testing for missing data mechanisms (MCAR, MAR, MNAR)
- Logistic regression modeling of missingness patterns
- Export functionality for reports and analysis results
- Jupyter notebook integration with interactive widgets

Examples
--------
Basic usage with a DataFrame containing missing values:

>>> import pandas as pd
>>> import scikit_na as na
>>>
>>> # Create sample data with missing values
>>> data = pd.DataFrame({
...     'age': [25, 30, None, 35, 40],
...     'income': [50000, None, 75000, None, 90000],
...     'education': ['HS', 'College', None, 'PhD', 'College']
... })
>>>
>>> # Get summary statistics
>>> na.summary(data)
>>>
>>> # Visualize missing data patterns
>>> na.altair.plot_heatmap(data)
>>>
>>> # Test for missing data mechanisms
>>> from scipy.stats import mannwhitneyu
>>> na.test_hypothesis(data, 'income', mannwhitneyu, columns=['age'])
"""

from . import (
    altair,  # noqa: F401
    mpl,  # noqa: F401
)
from ._export import *  # noqa: F401, F403
from ._report import *  # noqa: F401, F403
from ._stats import *  # noqa: F401, F403

__version__ = "0.3.0"
