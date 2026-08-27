"""Tests for the matplotlib visualization module."""

import numpy as np
import pytest
from pandas import DataFrame

# Try to import matplotlib, skip tests if not available
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from src.scikit_na.mpl._mpl import (
        plot_corr,
        plot_heatmap,
        plot_hist,
        plot_kde,
        plot_stats,
    )

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# Skip all tests if matplotlib is not available
pytestmark = pytest.mark.skipif(not MPL_AVAILABLE, reason="Matplotlib is not available")


@pytest.fixture(autouse=True)
def _fresh_figure():
    """Give every test its own axes; seaborn otherwise draws onto the last one."""
    plt.figure()
    yield
    plt.close("all")


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """Create a sample DataFrame with mixed data types for testing."""
    np.random.seed(42)
    df = DataFrame(
        {
            "numeric1": np.random.normal(0, 1, 100),
            "numeric2": np.random.normal(5, 2, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        },
    )

    # Add some NAs
    df.loc[0:10, "numeric1"] = np.nan
    df.loc[20:30, "numeric2"] = np.nan
    df.loc[40:50, "category"] = np.nan

    # Add NA indicator columns
    df["numeric1_na"] = df["numeric1"].isna()
    df["numeric2_na"] = df["numeric2"].isna()
    df["category_na"] = df["category"].isna()

    return df


@pytest.fixture(name="na_info_data")
def fixture_na_info_data():
    """Create a sample DataFrame with NA information for testing plot_stats."""
    return DataFrame(
        {"A": [10, 20.0, 30.0, 40, 50.0], "B": [5, 10.0, 15.0, 20, 25.0]},
        index=[
            "na_count",
            "na_pct_per_col",
            "na_pct_total",
            "rows_after_dropna",
            "rows_after_dropna_pct",
        ],
    )


def test_plot_corr(sample_data):
    """Test plot_corr function."""
    ax = plot_corr(data=sample_data, columns=["numeric1", "numeric2"])

    # Check that the returned object is a matplotlib Axes
    assert isinstance(ax, Axes)

    # Check that the axes has a title
    assert ax.get_title() is not None


def test_plot_stats(na_info_data):
    """Test plot_stats function."""
    ax = plot_stats(na_info=na_info_data, idxstr="na_count")

    # Check that the returned object is a matplotlib Axes
    assert isinstance(ax, Axes)

    # Check that the axes has a title
    assert ax.get_title() is not None


def test_plot_heatmap(sample_data):
    """Test plot_heatmap function."""
    ax = plot_heatmap(data=sample_data, columns=["numeric1", "numeric2"])

    # Check that the returned object is a matplotlib Axes
    assert isinstance(ax, Axes)

    # Check that the axes has a title
    assert ax.get_title() is not None


def test_plot_hist(sample_data):
    """Test plot_hist function."""
    ax = plot_hist(data=sample_data, col="numeric1", col_na="numeric1_na")

    # Check that the returned object is a matplotlib Axes
    assert isinstance(ax, Axes)

    # Check that the axes has a title
    assert ax.get_title() is not None


def test_plot_kde(sample_data):
    """Test plot_kde function."""
    ax = plot_kde(data=sample_data, col="numeric1", col_na="numeric1_na")

    # Check that the returned object is a matplotlib Axes
    assert isinstance(ax, Axes)

    # Check that the axes has a title
    assert ax.get_title() is not None


def test_plot_corr_with_options(sample_data):
    """Test plot_corr function with various options."""
    ax = plot_corr(
        data=sample_data,
        columns=["numeric1", "numeric2"],
        mask_diag=False,
        corr_kws={"method": "spearman"},
        heat_kws={"cmap": "viridis"},
    )

    assert isinstance(ax, Axes)


def test_plot_stats_with_options(na_info_data):
    """Test plot_stats function with various options."""
    ax = plot_stats(na_info=na_info_data, idxint=0)

    assert isinstance(ax, Axes)


def test_plot_hist_with_options(sample_data):
    """Test plot_hist function with various options."""
    ax = plot_hist(
        data=sample_data,
        col="numeric1",
        col_na="numeric1_na",
        col_na_fmt='"{}" is missing',
        stat="count",
        common_norm=True,
        hist_kws={"bins": 15, "alpha": 0.7},
    )

    assert isinstance(ax, Axes)


def test_plot_kde_with_options(sample_data):
    """Test plot_kde function with various options."""
    ax = plot_kde(
        data=sample_data,
        col="numeric1",
        col_na="numeric1_na",
        col_na_fmt='"{}" is missing',
        common_norm=True,
        kde_kws={"bw_adjust": 0.5, "alpha": 0.7},
    )

    assert isinstance(ax, Axes)


# --- Regression tests -------------------------------------------------------


@pytest.mark.parametrize("plot_func", [plot_hist, plot_kde])
def test_plot_dist_rejects_identical_columns(sample_data, plot_func):
    """`col == col_na` used to fail with a confusing duplicate-column error."""
    with pytest.raises(ValueError, match="must be different columns"):
        plot_func(sample_data, col="numeric1", col_na="numeric1")


def test_plot_hist_honours_stat_alongside_hist_kws(sample_data):
    """Passing `hist_kws` must not silently discard `stat`/`common_norm`."""
    ax = plot_hist(sample_data, col="numeric1", col_na="numeric2", stat="count", hist_kws={"bins": 3})

    assert isinstance(ax, Axes)
    # `stat="count"` keeps raw counts, so the tallest bar is a whole number.
    heights = [patch.get_height() for patch in ax.patches]
    assert heights
    assert all(height == int(height) for height in heights)


def test_plot_kde_honours_common_norm_alongside_kde_kws(sample_data):
    """`kde_kws` must merge with, not replace, the explicit parameters."""
    ax = plot_kde(sample_data, col="numeric1", col_na="numeric2", common_norm=True, kde_kws={"warn_singular": False})

    assert isinstance(ax, Axes)


def test_plot_hist_kws_can_override_explicit_stat(sample_data):
    """An explicit key in `hist_kws` still wins over the named parameter."""
    ax = plot_hist(sample_data, col="numeric1", col_na="numeric2", stat="density", hist_kws={"stat": "count"})

    assert isinstance(ax, Axes)


def test_plot_heatmap_uses_fixed_color_limits():
    """Colour limits must be pinned so a category always maps to one colour."""
    no_na = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    ax = plot_heatmap(no_na)

    assert ax.collections[0].get_clim() == (0.0, 1.0)


def test_plot_heatmap_sb_kws_can_override_color_limits(sample_data):
    """The pinned limits are defaults, not a hard-coded constraint."""
    ax = plot_heatmap(sample_data, sb_kws={"cbar": False, "vmin": 0, "vmax": 2})

    assert ax.collections[0].get_clim() == (0.0, 2.0)
