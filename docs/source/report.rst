Interactive report
==================

Report interface is based on `ipywidgets
<https://github.com/jupyter-widgets/ipywidgets>`_. It can help you quickly and
interactively explore NA values in your dataset, view patterns, calculate
statistics, show relevant figures and tables.

To begin, just load the data and pass your DataFrame to ``scikit_na.report()``
method:

.. code:: python

    import pandas as pd
    import scikit_na as na

    data = pd.read_csv('titanic_dataset.csv')
    na.report(data)

Summary tab
~~~~~~~~~~~

.. image:: _static/report_summary.png
    :alt: Summary tab

Visualization tab
~~~~~~~~~~~~~~~~~

.. image:: _static/report_vis.png
    :alt: Visualization tab

Statistics tab
~~~~~~~~~~~~~~

.. image:: _static/report_stats.png
    :alt: Statistics tab

Correlations tab
~~~~~~~~~~~~~~~~

.. image:: _static/report_correlations.png
    :alt: Correlations tab

Distributions tab
~~~~~~~~~~~~~~~~~

.. image:: _static/report_distributions.png
    :alt: Distributions tab
