Installation
============

Quick Install
~~~~~~~~~~~~~

Install the latest stable version from PyPI:

.. code-block:: bash

   pip install scikit-na

Development Version
~~~~~~~~~~~~~~~~~~~

Install the latest development version directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/maximtrp/scikit-na.git

Verify Installation
~~~~~~~~~~~~~~~~~~~

Test your installation:

.. code-block:: python

   import scikit_na as na
   print(na.__version__)

Dependencies
~~~~~~~~~~~~

**scikit-na** requires the following packages:

Core Dependencies
-----------------
* **pandas** (≥1.0): Data manipulation and analysis
* **numpy** (≥1.18): Numerical computing
* **scipy** (≥1.5): Scientific computing

Statistical Analysis
--------------------
* **statsmodels** (≥0.12): Statistical modeling

Visualization
-------------
* **matplotlib** (≥3.0): Static plotting
* **seaborn** (≥0.11): Statistical data visualization
* **altair** (≥4.0): Interactive visualizations

Interactive Features
--------------------
* **ipywidgets** (≥7.0): Jupyter notebook widgets (for interactive reports)

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For enhanced functionality, consider installing:

* **openpyxl**: Excel file export support
* **xlsxwriter**: Advanced Excel formatting

.. code-block:: bash

   pip install scikit-na[excel]  # Includes Excel support

Jupyter Integration
~~~~~~~~~~~~~~~~~~~

For the best experience with interactive features:

.. code-block:: bash

   # Enable widget extensions
   jupyter nbextension enable --py widgetsnbextension

   # For JupyterLab
   jupyter labextension install @jupyter-widgets/jupyterlab-manager