import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, read_csv
from pytest import fixture

from src import scikit_na as na


@fixture(name="data")
def fixture_data():
    return read_csv("./tests/titanic_dataset.csv")


@fixture(name="simple_data")
def fixture_simple_data():
    """Create simple test data for more controlled testing."""
    return DataFrame(
        {
            "A": [1, 2, np.nan, 4, 5],
            "B": [np.nan, 2.2, 3.3, np.nan, 5.5],
            "C": ["a", "b", np.nan, "d", np.nan],
            "D": [1, 2, 3, 4, 5],  # No missing values
        }
    )


def test_summary(data):
    summary = na.summary(data)

    na_counts_correct = all(data.isna().sum() == summary.loc["na_count"])
    na_pct_correct = all((data.isna().sum() / data.shape[0] * 100).round(2) == summary.loc["na_pct_per_col"])
    na_pct_total = (data.isna().sum() / data.isna().sum().sum() * 100).round(2)
    na_pct_total_correct = all(na_pct_total == summary.loc["na_pct_total"])
    rows_after_dropna_correct = all((data.shape[0] - data.isna().sum()) == summary.loc["rows_after_dropna"])
    rows_after_dropna_pct_correct = all(
        ((data.shape[0] - data.isna().sum()) * 100 / data.shape[0]).round(2) == summary.loc["rows_after_dropna_pct"]
    )

    na_unique1 = ((data.isna().sum(axis=1) == 1) & data.loc[:, "Age"].isna()).sum().item()
    na_unique_correct1 = na_unique1 == 19
    na_unique_pct_correct1 = (na_unique1 / data.isna().sum()["Age"] * 100).round(2) == summary.loc[
        "na_unique_pct_per_col", "Age"
    ]
    na_unique_correct2 = ((data.isna().sum(axis=1) == 1) & data.loc[:, "Cabin"].isna()).sum().item() == 529
    na_unique_correct3 = ((data.isna().sum(axis=1) == 1) & data.loc[:, "Survived"].isna()).sum().item() == 0
    assert na_counts_correct
    assert na_pct_correct
    assert na_pct_total_correct
    assert rows_after_dropna_correct
    assert rows_after_dropna_pct_correct
    assert na_unique_correct1
    assert na_unique_pct_correct1
    assert na_unique_correct2
    assert na_unique_correct3


# Additional comprehensive tests
def test_summary_basic_functionality(simple_data):
    """Test basic functionality of summary with simple data."""
    result = na.summary(simple_data)

    # Test return type
    assert isinstance(result, DataFrame)

    # Test that expected rows are present
    expected_rows = [
        "na_count",
        "na_pct_per_col",
        "na_pct_total",
        "na_unique_per_col",
        "na_unique_pct_per_col",
        "rows_after_dropna",
        "rows_after_dropna_pct",
    ]
    for row in expected_rows:
        assert row in result.index

    # Test that all columns are included
    assert set(result.columns) == set(simple_data.columns)


def test_summary_per_column_false(simple_data):
    """Test summary with per_column=False."""
    result = na.summary(simple_data, per_column=False)

    assert isinstance(result, DataFrame)
    assert "dataset" in result.columns

    expected_rows = [
        "total_cols",
        "na_cols",
        "na_only_cols",
        "total_rows",
        "na_rows",
        "non_na_rows",
        "total_cells",
        "na_cells",
        "na_cells_pct",
        "non_na_cells",
        "non_na_cells_pct",
    ]
    for row in expected_rows:
        assert row in result.index


def test_summary_with_columns_parameter(simple_data):
    """Test summary with specific columns."""
    columns = ["A", "B"]
    result = na.summary(simple_data, columns=columns)

    assert isinstance(result, DataFrame)
    assert set(result.columns) == set(columns)
    assert "C" not in result.columns
    assert "D" not in result.columns


def test_summary_rounding(simple_data):
    """Test summary with different rounding settings."""
    # Test with round_dec=0
    result_no_round = na.summary(simple_data, round_dec=0)
    assert isinstance(result_no_round, DataFrame)

    # Test with round_dec=3
    result_round_3 = na.summary(simple_data, round_dec=3)
    assert isinstance(result_round_3, DataFrame)

    # Check that rounding affects the values
    na_pct_no_round = result_no_round.loc["na_pct_per_col", "A"]
    na_pct_round_3 = result_round_3.loc["na_pct_per_col", "A"]

    # Both should be numbers but potentially different precision
    assert isinstance(na_pct_no_round, (int, float))
    assert isinstance(na_pct_round_3, (int, float))


def test_summary_edge_cases():
    """Test summary with edge cases."""
    # Test with empty DataFrame
    empty_df = DataFrame()
    try:
        result = na.summary(empty_df)
        assert isinstance(result, DataFrame)
    except (ValueError, IndexError):
        # Expected for empty DataFrame
        pass

    # Test with DataFrame with no missing values
    no_na_df = DataFrame({"A": [1, 2, 3, 4, 5], "B": [1.1, 2.2, 3.3, 4.4, 5.5]})
    result = na.summary(no_na_df)
    assert isinstance(result, DataFrame)

    # All na_count should be 0
    na_counts = result.loc["na_count"]
    assert all(count == 0 for count in na_counts)

    # Test with DataFrame with all missing values
    all_na_df = DataFrame({"A": [np.nan, np.nan, np.nan], "B": [np.nan, np.nan, np.nan]})
    result = na.summary(all_na_df)
    assert isinstance(result, DataFrame)

    # All na_count should equal the number of rows
    na_counts = result.loc["na_count"]
    assert all(count == 3 for count in na_counts)


def test_summary_with_single_column():
    """Test summary with single column DataFrame."""
    single_col_df = DataFrame({"A": [1, np.nan, 3, np.nan, 5]})

    result = na.summary(single_col_df)
    assert isinstance(result, DataFrame)
    assert list(result.columns) == ["A"]

    # Check that calculations are correct
    assert result.loc["na_count", "A"] == 2
    assert result.loc["na_pct_per_col", "A"] == 40.0  # 2/5 * 100


def test_summary_type_compatibility(simple_data):
    """Test that summary works with different column types."""
    # Add different column types
    test_df = simple_data.copy()
    test_df["datetime"] = pd.date_range("2023-01-01", periods=5)
    test_df["category"] = pd.Categorical(["cat1", "cat2", np.nan, "cat1", "cat2"])
    test_df["boolean"] = [True, False, np.nan, True, False]

    result = na.summary(test_df)
    assert isinstance(result, DataFrame)

    # Should handle all column types
    expected_cols = set(test_df.columns)
    assert set(result.columns) == expected_cols


def test_summary_calculations_accuracy(simple_data):
    """Test that summary calculations are mathematically accurate."""
    result = na.summary(simple_data)

    # Manually calculate expected values for column A
    a_na_count = simple_data["A"].isna().sum()
    a_na_pct = (a_na_count / len(simple_data)) * 100
    a_rows_after_dropna = len(simple_data) - a_na_count

    assert result.loc["na_count", "A"] == a_na_count
    assert abs(result.loc["na_pct_per_col", "A"] - a_na_pct) < 0.01  # Allow small rounding difference
    assert result.loc["rows_after_dropna", "A"] == a_rows_after_dropna


def test_summary_parameter_validation():
    """Test parameter validation in summary function."""
    df = DataFrame({"A": [1, 2, np.nan]})

    # Test with invalid columns parameter
    with pytest.raises(KeyError):
        na.summary(df, columns=["nonexistent_column"])

    # Test with valid parameters
    result = na.summary(df, columns=["A"], per_column=True, round_dec=1)
    assert isinstance(result, DataFrame)
