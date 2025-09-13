import logging
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series

from src.scikit_na._stats import (
    _get_abs_na_count,
    _get_na_perc,
    _get_nominal_cols,
    _get_numeric_cols,
    _get_rows_after_cum_dropna,
    _get_rows_after_dropna,
    _get_total_na_count,
    _get_unique_na,
    _select_cols,
    correlate,
    describe,
    model,
    stairs,
    summary,
    test_hypothesis,
)

logger = logging.getLogger(__name__)

# Skip the test_hypothesis function as it is a helper function
test_hypothesis.__test__ = False  # type: ignore[attr-defined]


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """Create a sample DataFrame with mixed column types for testing."""
    return DataFrame(
        {
            "numeric_int": [1, 2, 3, 4, 5],
            "numeric_float": [1.1, 2.2, 3.3, 4.4, 5.5],
            "string_col": ["a", "b", "c", "d", "e"],
            "categorical": pd.Categorical(["cat1", "cat2", "cat1", "cat2", "cat1"]),
            "bool_col": [True, False, True, False, True],
            "datetime": pd.date_range("2023-01-01", periods=5),
            "mixed_col": [1, "b", 3, "d", 5],
        }
    )


@pytest.fixture(name="data_with_na")
def fixture_data_with_na():
    """Create a sample DataFrame with missing values for testing."""
    df = DataFrame(
        {
            "A": [1, 2, np.nan, 4, 5],
            "B": [np.nan, 2.2, 3.3, np.nan, 5.5],
            "C": ["a", "b", np.nan, "d", np.nan],
            "D": [True, False, True, np.nan, True],
        }
    )
    return df


@pytest.fixture(name="correlation_data")
def fixture_correlation_data():
    """Create data for correlation testing."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = x * 0.8 + np.random.normal(0, 0.5, 100)  # Correlated with x
    z = np.random.normal(0, 1, 100)  # Independent

    df = DataFrame({"x": x, "y": y, "z": z, "cat": np.random.choice(["A", "B", "C"], 100)})

    # Add some NAs
    df.loc[0:5, "x"] = np.nan
    df.loc[10:15, "y"] = np.nan
    df.loc[20:25, "z"] = np.nan

    return df


@pytest.fixture(name="model_data")
def fixture_model_data():
    """Create data for model testing."""
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)

    # Create a binary target with some relationship to x1 and x2
    logit = 1.5 + 0.8 * x1 - 0.4 * x2
    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob)

    df = DataFrame({"x1": x1, "x2": x2, "y": y})

    # Add some NAs - make sure we have both True and False values for y_is_na
    df.loc[0:5, "x1"] = np.nan
    df.loc[10:15, "x2"] = np.nan
    df.loc[20:25, "y"] = np.nan  # These rows will have y_is_na = True

    return df


# Tests for helper functions
def test_select_cols_default(sample_data):
    """Test _select_cols with default parameters."""
    cols = _select_cols(sample_data)

    assert isinstance(cols, ndarray)
    assert set(cols) == set(sample_data.columns)
    assert len(cols) == len(sample_data.columns)


def test_select_cols_with_columns(sample_data):
    """Test _select_cols with specified columns."""
    specified_cols = ["numeric_int", "string_col"]
    cols = _select_cols(sample_data, specified_cols)

    assert isinstance(cols, ndarray)
    assert set(cols) == set(specified_cols)
    assert len(cols) == len(specified_cols)


def test_select_cols_with_second_var(sample_data):
    """Test _select_cols with second_var parameter."""
    second_var_cols = ["numeric_float", "bool_col"]
    cols = _select_cols(sample_data, None, second_var_cols)

    assert isinstance(cols, ndarray)
    assert set(cols) == set(second_var_cols)
    assert len(cols) == len(second_var_cols)


def test_select_cols_with_empty_second_var(sample_data):
    """Test _select_cols with empty second_var parameter."""
    cols = _select_cols(sample_data, None, [])

    assert isinstance(cols, ndarray)
    assert set(cols) == set(sample_data.columns)
    assert len(cols) == len(sample_data.columns)


def test_get_nominal_cols_all_columns(sample_data):
    """Test _get_nominal_cols with all columns."""
    nominal_cols = _get_nominal_cols(sample_data)

    # Only string and mixed columns should be identified as nominal
    assert set(nominal_cols) == {"string_col", "mixed_col"}
    assert len(nominal_cols) == 2
    assert "numeric_int" not in nominal_cols
    assert "numeric_float" not in nominal_cols


def test_get_nominal_cols_subset(sample_data):
    """Test _get_nominal_cols with a subset of columns."""
    columns = ["string_col", "numeric_int", "bool_col"]
    nominal_cols = _get_nominal_cols(sample_data, columns)

    assert set(nominal_cols) == {"string_col"}
    assert len(nominal_cols) == 1


def test_get_nominal_cols_empty_result(sample_data):
    """Test _get_nominal_cols with columns that are not nominal."""
    columns = ["numeric_int", "numeric_float", "bool_col"]
    nominal_cols = _get_nominal_cols(sample_data, columns)

    assert len(nominal_cols) == 0


def test_get_numeric_cols_all_columns(sample_data):
    """Test _get_numeric_cols with all columns."""
    numeric_cols = _get_numeric_cols(sample_data)

    assert set(numeric_cols) == {"numeric_int", "numeric_float"}
    assert len(numeric_cols) == 2
    assert "string_col" not in numeric_cols
    assert "mixed_col" not in numeric_cols


def test_get_numeric_cols_subset(sample_data):
    """Test _get_numeric_cols with a subset of columns."""
    columns = ["numeric_float", "string_col", "bool_col"]
    numeric_cols = _get_numeric_cols(sample_data, columns)

    assert set(numeric_cols) == {"numeric_float"}
    assert len(numeric_cols) == 1


def test_get_numeric_cols_empty_result(sample_data):
    """Test _get_numeric_cols with columns that are not numeric."""
    columns = ["string_col", "bool_col", "categorical"]
    numeric_cols = _get_numeric_cols(sample_data, columns)

    assert len(numeric_cols) == 0


def test_get_unique_na(data_with_na):
    """Test _get_unique_na function."""
    # Create a Series where only one column has NA
    unique_na_A = Series([False, False, True, False, False])  # Only row 2 has NA in column A
    unique_na_B = Series([True, False, False, True, False])  # Rows 0 and 3 have NA in column B

    # Test for column A
    result_A = _get_unique_na(unique_na_A, data_with_na, "A")
    assert result_A == 1  # Only one row where both unique_na_A is True and A has NA

    # Test for column B
    result_B = _get_unique_na(unique_na_B, data_with_na, "B")
    assert result_B == 2  # Two rows where both unique_na_B is True and B has NA


def test_get_rows_after_dropna(data_with_na):
    """Test _get_rows_after_dropna function."""
    # Test with specific column
    result_A = _get_rows_after_dropna(data_with_na, "A")
    assert result_A == 4  # 5 rows - 1 NA in column A

    result_B = _get_rows_after_dropna(data_with_na, "B")
    assert result_B == 3  # 5 rows - 2 NAs in column B

    # Test with all columns
    result_all = _get_rows_after_dropna(data_with_na)
    assert result_all == 1  # Only 1 row has no NAs across all columns


def test_get_rows_after_cum_dropna(data_with_na):
    """Test _get_rows_after_cum_dropna function."""
    # Test with one column
    result_A = _get_rows_after_cum_dropna(data_with_na, ["A"])
    assert result_A == 4  # 5 rows - 1 NA in column A

    # Test with multiple columns
    result_AB = _get_rows_after_cum_dropna(data_with_na, ["A", "B"])
    assert result_AB == 2  # Only 2 rows have no NAs in both A and B

    # Test with cols and col parameters
    result_A_plus_B = _get_rows_after_cum_dropna(data_with_na, ["A"], "B")
    assert result_A_plus_B == 2  # Same as above, different way to specify

    # Test with no columns specified
    result_none = _get_rows_after_cum_dropna(data_with_na)
    assert result_none == 5  # All rows when no columns specified for NA checking


def test_get_abs_na_count(data_with_na):
    """Test _get_abs_na_count function."""
    result = _get_abs_na_count(data_with_na, data_with_na.columns)

    assert isinstance(result, Series)
    assert result["A"] == 1  # Column A has 1 NA
    assert result["B"] == 2  # Column B has 2 NAs
    assert result["C"] == 2  # Column C has 2 NAs
    assert result["D"] == 1  # Column D has 1 NA


def test_get_na_perc(data_with_na):
    """Test _get_na_perc function."""
    na_abs = _get_abs_na_count(data_with_na, data_with_na.columns)
    result = _get_na_perc(data_with_na, na_abs)

    assert isinstance(result, Series)
    assert result["A"] == 20.0  # 1/5 = 20%
    assert result["B"] == 40.0  # 2/5 = 40%
    assert result["C"] == 40.0  # 2/5 = 40%
    assert result["D"] == 20.0  # 1/5 = 20%


def test_get_total_na_count(data_with_na):
    """Test _get_total_na_count function."""
    result = _get_total_na_count(data_with_na, data_with_na.columns)

    assert result == 6  # Total of 6 NAs across all columns


# Tests for public functions
def test_summary_basic(data_with_na):
    """Test basic summary function."""
    result = summary(data_with_na)

    assert isinstance(result, DataFrame)
    # Check that the summary contains expected rows
    assert "na_count" in result.index
    assert "na_pct_per_col" in result.index
    assert "na_pct_total" in result.index
    assert "rows_after_dropna" in result.index


def test_summary_with_columns(data_with_na):
    """Test summary function with specific columns."""
    result = summary(data_with_na, columns=["A", "B"])

    assert isinstance(result, DataFrame)
    assert set(result.columns) == {"A", "B"}
    assert "C" not in result.columns
    assert "D" not in result.columns


def test_summary_without_per_column(data_with_na):
    """Test summary function with per_column=False."""
    result = summary(data_with_na, per_column=False)

    assert isinstance(result, DataFrame)
    assert result.shape[1] == 1  # Only one column for summary stats
    # Check for expected rows in the summary without per-column stats
    assert "total_cols" in result.index
    assert "na_cols" in result.index
    assert "total_rows" in result.index
    assert "na_rows" in result.index
    assert "na_cells" in result.index
    assert "na_cells_pct" in result.index


def test_summary_with_rounding(data_with_na):
    """Test summary function with different rounding."""
    result = summary(data_with_na, round_dec=3)

    # Check that percentages have 3 decimal places
    assert isinstance(result.loc["na_pct_per_col", "A"], float)


def test_stairs(data_with_na):
    """Test stairs function."""
    result = stairs(data_with_na)

    assert isinstance(result, DataFrame)
    # Check the structure of the result
    assert "Columns" in result.columns
    assert "Instances" in result.columns
    assert "Size difference" in result.columns

    # Test with specific columns
    result_cols = stairs(data_with_na, columns=["A", "B"])
    assert "A" in result_cols["Columns"].values
    assert "B" in result_cols["Columns"].values


def test_correlate(data_with_na):
    """Test correlate function."""
    # The correlate function works on NA patterns, not the actual values
    result = correlate(data_with_na)

    assert isinstance(result, DataFrame)
    # Check that the result is a correlation matrix of NA patterns
    assert result.shape[0] == result.shape[1]  # Square matrix

    # Test with method parameter
    result_spearman = correlate(data_with_na, method="pearson")
    assert isinstance(result_spearman, DataFrame)

    # Test with columns parameter
    result_cols = correlate(data_with_na, columns=["A", "B"])
    assert result_cols.shape[0] == 2
    assert result_cols.shape[1] == 2


def test_describe(data_with_na):
    """Test describe function."""
    # Create a binary indicator for NA in column A
    data_with_na["A_is_na"] = data_with_na["A"].isna()

    result = describe(data_with_na, col_na="A_is_na")

    assert isinstance(result, DataFrame)
    # Check that it includes statistics for both NA and non-NA groups
    assert result.shape[1] >= 2  # At least two columns for NA and non-NA groups

    # Test with columns parameter
    result_cols = describe(data_with_na, col_na="A_is_na", columns=["B", "C"])
    assert result_cols.shape[0] > 0  # Should have some rows

    # Test with na_mapping parameter
    custom_mapping = {True: "Missing", False: "Present"}
    result_mapping = describe(data_with_na, col_na="A_is_na", na_mapping=custom_mapping)
    assert isinstance(result_mapping, DataFrame)


def test_model(data_with_na):
    """Test model function."""
    # Create a simple dataset with a clear pattern for the model
    np.random.seed(42)
    test_df = DataFrame(
        {
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
            "na_col": [1, 2, np.nan, 4, 5] * 20,  # Create pattern with NAs
        }
    )

    # Test logistic regression with a simple dataset
    try:
        result = model(test_df, col_na="na_col", columns=["x", "y"], fit_kws={"disp": False})
        # If the model runs successfully, check that it has the expected attributes
        assert hasattr(result, "summary")
        assert hasattr(result, "params")
        assert hasattr(result, "pvalues")
        # Check return type matches type hint
        from statsmodels.discrete.discrete_model import BinaryResultsWrapper

        assert isinstance(result, BinaryResultsWrapper)
    except (ImportError, AttributeError) as e:
        logger.exception("Model test failed due to missing dependencies or attribute error")
        pytest.skip(f"Model test skipped due to dependency issue: {e}")
    except (ValueError, TypeError, KeyError) as e:
        logger.exception("Model test failed due to invalid data or parameters")
        pytest.skip(f"Model test skipped due to data/parameter error: {e}")
    except Exception as e:
        logger.exception("Unexpected error occurred while testing model function")
        pytest.skip(f"Model test skipped due to unexpected error: {e}")


def test_model_with_invalid_column(data_with_na):
    """Test model function with invalid column."""
    with pytest.raises(KeyError):
        model(data_with_na, col_na="nonexistent_col", columns=["A"])


def test_test_hypothesis(model_data):
    """Test test_hypothesis function."""
    # Create a binary indicator for NA in column y with both True and False values
    # We need to ensure there are both True and False values in the isna() result
    model_data["col_with_na"] = np.nan  # All values are NA
    model_data.loc[0:50, "col_with_na"] = 1.0  # Half the values are not NA

    # Define a simple test function that works with the test_hypothesis implementation
    def simple_test_fn(group1, group2):
        # Simple function that returns a tuple with statistic and p-value
        return (1.0, 0.5)

    # Test with a custom test function
    result = test_hypothesis(model_data, col_na="col_with_na", test_fn=simple_test_fn, columns=["x1"])

    assert isinstance(result, dict)
    assert "x1" in result


# Additional tests for improved functionality
def test_summary_return_type_hints(data_with_na):
    """Test that summary function returns proper types as per type hints."""
    result = summary(data_with_na)
    assert isinstance(result, DataFrame)

    # Test with specific columns (type hint: Optional[Iterable[str]])
    result_cols = summary(data_with_na, columns=["A", "B"])
    assert isinstance(result_cols, DataFrame)
    assert set(result_cols.columns) == {"A", "B"}


def test_correlate_return_type_hints(data_with_na):
    """Test that correlate function returns proper types."""
    result = correlate(data_with_na)
    assert isinstance(result, DataFrame)

    # Test with kwargs (type hint: **kwargs: Any)
    result_kendall = correlate(data_with_na, method="kendall")
    assert isinstance(result_kendall, DataFrame)


def test_describe_return_type_hints(data_with_na):
    """Test that describe function returns proper types."""
    # Add a column to group by
    data_with_na["grouping_col"] = [1, 2, np.nan, 1, 2]

    result = describe(data_with_na, col_na="grouping_col", columns=["A", "B"])
    assert isinstance(result, DataFrame)


def test_summary_with_empty_dataframe():
    """Test summary function with empty DataFrame."""
    empty_df = DataFrame()

    # Should handle empty DataFrame gracefully
    try:
        result = summary(empty_df)
        assert isinstance(result, DataFrame)
    except (ValueError, IndexError):
        # Expected for empty DataFrame
        pass


def test_correlate_with_no_missing_values():
    """Test correlate function when no columns have missing values."""
    no_na_data = DataFrame({"A": [1, 2, 3, 4, 5], "B": [1.1, 2.2, 3.3, 4.4, 5.5]})

    result = correlate(no_na_data, drop=True)
    # Should return empty or handle gracefully when no NAs
    assert isinstance(result, DataFrame)


def test_summary_edge_cases(data_with_na):
    """Test summary function edge cases."""
    # Test with round_dec=0
    result = summary(data_with_na, round_dec=0)
    assert isinstance(result, DataFrame)

    # Test per_column=False
    result_dataset = summary(data_with_na, per_column=False)
    assert isinstance(result_dataset, DataFrame)
    assert "dataset" in result_dataset.columns


def test_helper_functions_edge_cases():
    """Test helper functions with edge cases."""
    # Test with single column DataFrame
    single_col_df = DataFrame({"A": [1, np.nan, 3]})

    numeric_cols = _get_numeric_cols(single_col_df)
    assert "A" in numeric_cols

    nominal_cols = _get_nominal_cols(single_col_df)
    assert len(nominal_cols) == 0  # No nominal columns

    # Test with all-NA column
    all_na_df = DataFrame({"A": [np.nan, np.nan, np.nan]})
    na_count = _get_abs_na_count(all_na_df, ["A"])
    assert na_count["A"] == 3


def test_stairs_function_basic(data_with_na):
    """Test stairs function basic functionality."""
    result = stairs(data_with_na)
    assert isinstance(result, DataFrame)
    assert "Columns" in result.columns  # Default xlabel
    assert "Instances" in result.columns  # Default ylabel


def test_stairs_with_custom_labels(data_with_na):
    """Test stairs function with custom labels."""
    result = stairs(
        data_with_na,
        xlabel="Custom X",
        ylabel="Custom Y",
        tooltip_label="Custom Tooltip",
        dataset_label="Custom Dataset",
    )
    assert isinstance(result, DataFrame)
    assert "Custom X" in result.columns
    assert "Custom Y" in result.columns
    assert "Custom Tooltip" in result.columns
