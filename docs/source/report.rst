Interactive report
==================

Report interface is based on `ipywidgets
<https://github.com/jupyter-widgets/ipywidgets>`_. It can help you quickly and
interactively explore NA values in your dataset, view patterns, calculate
statistics, show relevant figures and tables. Interface is tested and meant to
be used in Jupyter Lab.

To begin, just load the data and pass your DataFrame to ``scikit_na.report()``
method:

.. code:: python

    import pandas as pd
    import scikit_na as na

    data = pd.read_csv('titanic_dataset.csv')
    na.report(data)

