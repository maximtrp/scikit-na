Statistics
==========

In missing data analysis, an important step is to calculate simple descriptive
and aggregating statistics of missing and non-missing data for each column and
for the whole dataset. *Scikit-na* attempts to provide useful functions for such operations.

Summary
~~~~~~~

We will use Titanic dataset that contains NA values in three
columns: *Age*, *Cabin*, and *Embarked*.

Per column
----------

To get a simple summary per each column, we will load a dataset using ``pandas``
and pass it to ``summary()`` function. The latter supports subsetting a dataset
with ``columns`` argument. And we will make use of it to cut the width of the
results table.

.. code:: python

    import scikit_na as na
    import pandas as pd

    data = pd.read_csv('titanic_dataset.csv')

    # Excluding three columns without NA to fit the table here
    na.summary(data, columns=data.columns.difference(['SibSp', 'Parch', 'Ticket']))

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

- *NA unique* is the number of NA values in each column
  that are unique for it, i.e. do not intersect with NA values in the other
  columns (or that will remain in dataset if we drop NA values in the other
  columns).

- *Rows left after dropna()* shows how many rows will be left in a dataset
  after applying ``pandas.Series.dropna()`` method to each column.

Whole dataset
-------------

By default, ``summary()`` function returns the results for each column. To get
the summary of missing data for the whole DataFrame, we should set ``per_column`` argument to
``False``.

.. code:: python

    na.summary(data, per_column=False)

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

Descriptive statistics
~~~~~~~~~~~~~~~~~~~~~~

The next step is to calculate simple descriptive statistics for columns with
quantitative and qualitative data. First, let's filter the columns by data
types:

.. code:: python

    # Presumably, qualitative data
    cols_nominal = data.columns[data.dtypes == object]

    # Quantitative data
    cols_numeric = data.columns[(data.dtypes == float) | (data.dtypes == int)]

We should also specify a column with missing values (NAs) that will be used to
split the data in the selected columns in two groups: NA (missing) and Filled
(non-missing).

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

Let's check there results by hand:

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

Now, let's look at the statistics for the numeric data:

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