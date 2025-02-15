Statistics
==========

In missing data analysis, an important step is calculating simple descriptive
and aggregate statistics for both missing and non-missing data in each column, as
well as for the entire dataset. *Scikit-na* provides useful functions to facilitate these
operations.

Summary
~~~~~~~

We will use Titanic dataset, which contains missing values (NA) in three
columns: *Age*, *Cabin*, and *Embarked*.

Per column
----------

To generate a simple summary for each column, we will load the dataset using ``pandas``
and pass it to the ``summary()`` function. This function supports subsetting the dataset
using the ``columns`` argument, which we will use to reduce the width of the results table.

.. code:: python

    import scikit_na as na
    import pandas as pd

    data = pd.read_csv('titanic_dataset.csv')

    # Excluding three columns without NA to fit the table here
    na.summary(data, columns=data.columns.difference(['SibSp', 'Parch', 'Ticket']))

===========================  ======  =======  ==========  ======  ======  =============  ========  =====  ==========
..                              Age    Cabin    Embarked    Fare    Name    PassengerId    Pclass    Sex    Survived
===========================  ======  =======  ==========  ======  ======  =============  ========  =====  ==========
na_count                     177      687           2          0       0              0         0      0           0
na_pct_per_col                19.87    77.1         0.22       0       0              0         0      0           0
na_pct_total                  20.44    79.33        0.23       0       0              0         0      0           0
na_unique_per_col             19      529           2          0       0              0         0      0           0
na_unique_pct_per_col         10.73    77         100          0       0              0         0      0           0
rows_after_dropna            714      204         889        891     891            891       891    891         891
rows_dropna_pct               80.13    22.9        99.78     100     100            100       100    100         100
===========================  ======  =======  ==========  ======  ======  =============  ========  =====  ==========

Those measures were meant to be self-explanatory:

- *na_count* is the number of NA values in each column.

- *na_unique_per_col* is the number of missing values in each column
  that are unique to it, meaning they do not overlap with NA values in other columns
  (or the number of values that would remain in the dataset if we drop rows with
  NA values from the other columns).

- *rows_after_dropna* is the number of rows that would remain in the dataset
  if we applied ``pandas.Series.dropna()`` method to each column.

Whole dataset
-------------

By default, the ``summary()`` function returns the results for each column. To get
the summary of missing data for the entire dataset, we should set the ``per_column``
argument to ``False``.

.. code:: python

    na.summary(data, per_column=False)

==============================  =========
..                                dataset
==============================  =========
total_columns                          12 
total_rows                            891 
na_rows                               708 
non_na_rows                           183 
total_cells                         10692 
na_cells                              866 
na_cells_pct                          8.1 
non_na_cells                         9826 
non_na_cells_pct                     91.9 
==============================  =========

Descriptive statistics
~~~~~~~~~~~~~~~~~~~~~~

The next step is to calculate descriptive statistics for columns with
quantitative and qualitative data. First, let's filter the columns by data
types:

.. code:: python

    # Presumably, qualitative data, needs checking
    cols_nominal = data.columns[data.dtypes == object]

    # Quantitative data
    cols_numeric = data.columns[(data.dtypes == float) | (data.dtypes == int)]

We should also specify a column with missing values (NAs) to be used for
splitting the data in the selected columns into two groups: NA (missing)
and Filled (non-missing).

Qualitative data
----------------

.. code:: python

    na.describe(data, columns=cols_nominal)

======  ======  ===  ======================  ====================  ======  ====  ======  ======
..        Embarked                        Name                     Sex           Ticket         
------  -----------  --------------------------------------------  ------------  --------------
Cabin   Filled  NA   Filled                  NA                    Filled   NA   Filled    NA  
======  ======  ===  ======================  ====================  ======  ====  ======  ======
count   202     687  204                     687                   204     687      204     687
unique  3       3    204                     687                   2       2        142     549
top     S       S    Levy, Mr. Rene Jacques  Nasser, Mr. Nicholas  male    male  113760  347082
freq    129     515  1                       1                     107     470        4       7
======  ======  ===  ======================  ====================  ======  ====  ======  ======

Let's check the results by hand:

.. code:: python

    data.groupby(
      data['Cabin'].isna().replace({False: 'Filled', True: 'NA'}))['Sex']\
    .value_counts()

======  ======  =====
Cabin   Sex     Count  
======  ======  =====
Filled  male    107  
..      female  97   
NA      male    470  
..      female  217  
======  ======  =====

Here we take *Cabin* column, encode missing/non-missing data as Filled/NA, and
then use it to group and count values in *Sex* column: among the passengers with
missing *cabin* data, 470 were males, while 217 were females.

Quantitative data
-----------------

Now, let's look at the statistics calculated for the numeric data:

.. code:: python

  # Selecting just two columns
  na.describe(data, columns=['Age', 'Fare'], col_na='Cabin')

=====  ========  ========  ========  =========
..     Age                 Fare               
-----  ------------------  -------------------
Cabin  Filled    NA        Filled    NA       
=====  ========  ========  ========  =========
count  185       529       204        687     
mean    35.8293   27.5553   76.1415    19.1573
std     15.6794   13.4726   74.3917    28.6633
min      0.92      0.42      0          0     
25%     24        19        29.4531     7.8771
50%     36        26        55.2208    10.5   
75%     48        35        89.3282    23     
max     80        74       512.329    512.329 
=====  ========  ========  ========  =========

The mean *age* of passengers with missing *cabin* data was 27.6 years.
