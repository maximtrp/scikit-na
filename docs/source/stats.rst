Statistics
==========

**scikit-na** includes functions for carrying out dataset- and column-wide
descriptive statistics

Descriptive statistics
~~~~~~~~~~~~~~~~~~~~~~

Per each column
---------------

We will use Titanic dataset (from Kaggle) that contains NA values in three
columns: Age, Cabin, and Embarked.

.. code:: python

   import scikit_na as na
   import pandas as pd

   data = pd.read_csv('titanic_dataset.csv')

   # Excluding three columns without NA to fit the table here
   na.describe(data, columns=data.columns.difference(['SibSp', 'Parch', 'Ticket']))

===========================  ======  =======  ==========  ======  ======  =============  ========  =====  ==========
..                              Age    Cabin    Embarked    Fare    Name    PassengerId    Pclass    Sex    Survived
===========================  ======  =======  ==========  ======  ======  =============  ========  =====  ==========
NA count                     177      687           2          0       0              0         0      0           0
NA, % (per column)            19.87    77.1         0.22       0       0              0         0      0           0
NA, % (of all NAs)            20.44    79.33        0.23       0       0              0         0      0           0
NA unique (per column)        19      529           2          0       0              0         0      0           0
NA unique, % (per column)     10.73    77         100          0       0              0         0      0           0
Rows left after dropna()     714      204         889        891     891            891       891    891         891
Rows left after dropna(), %   80.13    22.9        99.78     100     100            100       100    100         100
===========================  ======  =======  ==========  ======  ======  =============  ========  =====  ==========

Those measures were meant to be self-explanatory:

- *NA count* is the number of NA values in each column.

- *NA unique* is the number of NA values per each column
  that are unique for it, i.e. do not intersect with NA values in the other
  columns (or that will remain in dataset if we drop NA values in the other
  columns).

- *Rows left after dropna()* shows how many rows will be left in dataset
  after applying ``pandas.Series.dropna()`` method to each column.
  
Whole dataset
-------------

We can also calculate descriptive statistics for the whole dataset:

.. code:: python

   na.describe(data, per_column=False)

==============================  =========
..                                dataset
==============================  =========
Total columns                        12
Total rows                          891
Rows with NA                        708
Rows without NA                     183
Total cells                       10692
Cells with NA                       866
Cells with NA, %                      8.1
Cells with non-missing data        9826
Cells with non-missing data, %       91.9
==============================  =========
