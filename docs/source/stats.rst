Statistical Analysis
===================

Understanding missing data patterns through statistical analysis is crucial for making
informed decisions about data handling strategies. **scikit-na** provides comprehensive
statistical functions to analyze missing data at both column and dataset levels.

This guide demonstrates key statistical functions using the Titanic dataset, which
contains missing values in three columns: *Age*, *Cabin*, and *Embarked*.

Getting Started
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import scikit_na as na

   # Load the Titanic dataset
   data = pd.read_csv('titanic_dataset.csv')

   # Quick overview of missing data
   print(f"Dataset shape: {data.shape}")
   print(f"Missing values per column:")
   print(data.isnull().sum())

Summary Statistics
~~~~~~~~~~~~~~~~~~

Column-Level Analysis
---------------------

Generate detailed statistics for each column to understand individual patterns:

.. code-block:: python

   # Comprehensive per-column summary
   summary_stats = na.summary(data, per_column=True)
   print(summary_stats)

   # Focus on columns with missing data only
   na.summary(data, columns=['Age', 'Cabin', 'Embarked'])

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

Understanding the Summary Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The summary provides several key metrics for missing data analysis:

**Missing Data Counts**
  - *na_count*: Absolute number of missing values in each column
  - *na_pct_per_col*: Percentage of missing values within each column
  - *na_pct_total*: This column's missing values as percentage of all missing values

**Missing Data Patterns**
  - *na_unique_per_col*: Missing values unique to this column (don't overlap with other columns)
  - *na_unique_pct_per_col*: Percentage of this column's missing values that are unique

**Impact Analysis**
  - *rows_after_dropna*: Rows remaining after dropping missing values from this column
  - *rows_after_dropna_pct*: Percentage of original rows that would remain

Dataset-Level Analysis
----------------------

For an overall dataset perspective, use aggregate statistics:

.. code-block:: python

   # Dataset-level summary
   dataset_summary = na.summary(data, per_column=False)
   print(dataset_summary)

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
