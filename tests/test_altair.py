"""Tests for the altair visualization module."""

import numpy as np
import pytest
from pandas import DataFrame

# Try to import altair, skip tests if not available
try:
    import altair as alt
    from src.scikit_na.altair._altair import (
        plot_hist,
        plot_kde,
        plot_corr,
        plot_scatter,
        plot_stairs,
        plot_stairbars,
        plot_heatmap,
        view_dist,
    )

    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False


# Skip all tests if altair is not available
pytestmark = pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """Create a sample DataFrame with mixed data types for testing."""
    np.random.seed(42)
    df = DataFrame(
        {
            "numeric1": np.random.normal(0, 1, 100),
            "numeric2": np.random.normal(5, 2, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
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


def test_plot_hist(sample_data):
    """Test plot_hist function."""
    chart = plot_hist(data=sample_data, col="numeric1", col_na="numeric1_na")

    # Check that the chart is an Altair Chart object
    assert isinstance(chart, alt.Chart)

    # Check that the chart has the expected encoding channels
    assert hasattr(chart.encoding, "x")
    assert hasattr(chart.encoding, "y")
    assert hasattr(chart.encoding, "color")


def test_plot_kde(sample_data):
    """Test plot_kde function."""
    chart = plot_kde(data=sample_data, col="numeric1", col_na="numeric1_na")

    # Check that the chart is an Altair Chart object
    assert isinstance(chart, alt.Chart)

    # Check that the chart has the expected encoding channels
    assert hasattr(chart.encoding, "x")
    assert hasattr(chart.encoding, "y")
    assert hasattr(chart.encoding, "color")


def test_plot_scatter(sample_data):
    """Test plot_scatter function."""
    chart = plot_scatter(
        data=sample_data, x_col="numeric1", y_col="numeric2", col_na="numeric1_na"
    )

    # Check that the chart is an Altair Chart object
    assert isinstance(chart, alt.Chart)

    # Check that the chart has the expected encoding channels
    assert hasattr(chart.encoding, "x")
    assert hasattr(chart.encoding, "y")
    assert hasattr(chart.encoding, "color")


def test_plot_stairs(sample_data):
    """Test plot_stairs function."""
    chart = plot_stairs(data=sample_data)

    # Check that the chart is an Altair Chart object
    assert isinstance(chart, alt.Chart)

    # Check that the chart has the expected encoding channels
    assert hasattr(chart.encoding, "x")
    assert hasattr(chart.encoding, "y")


def test_plot_stairbars(sample_data):
    """Test plot_stairbars function."""
    chart = plot_stairbars(data=sample_data)

    # Check that the chart is an Altair Chart object
    assert isinstance(chart, alt.Chart)

    # Check that the chart has the expected encoding channels
    assert hasattr(chart.encoding, "x")
    assert hasattr(chart.encoding, "y")


def test_plot_heatmap(sample_data):
    """Test plot_heatmap function."""
    chart = plot_heatmap(data=sample_data)

    # Check that the chart is an Altair Chart object
    assert isinstance(chart, alt.Chart)

    # Check that the chart has the expected encoding channels
    assert hasattr(chart.encoding, "x")
    assert hasattr(chart.encoding, "y")
    assert hasattr(chart.encoding, "color")


def test_plot_corr(sample_data):
    """Test plot_corr function."""
    chart = plot_corr(data=sample_data, columns=["numeric1", "numeric2"])

    # Check that the chart is an Altair LayerChart object
    assert isinstance(chart, alt.LayerChart)

    # Check that the chart has the expected layers
    assert len(chart.layer) >= 1


def test_view_dist(sample_data):
    """Test view_dist function."""
    # This function returns an ipywidgets.VBox, which is difficult to test directly
    # Instead, we'll just check that it runs without errors
    try:
        widget = view_dist(data=sample_data, columns=["numeric1", "numeric2"])
        assert widget is not None
    except Exception as e:
        pytest.fail(f"view_dist raised an exception: {e}")


def test_plot_hist_with_options(sample_data):
    """Test plot_hist function with various options."""
    chart = plot_hist(
        data=sample_data,
        col="numeric1",
        col_na="numeric1_na",
        na_label="Missing",
        step=True,
        norm=False,
        font_size=12,
        xlabel="Custom X Label",
        ylabel="Custom Y Label",
    )

    assert isinstance(chart, alt.Chart)


def test_plot_kde_with_options(sample_data):
    """Test plot_kde function with various options."""
    chart = plot_kde(
        data=sample_data,
        col="numeric1",
        col_na="numeric1_na",
        na_label="Missing",
        font_size=12,
        xlabel="Custom X Label",
        ylabel="Custom Y Label",
    )

    assert isinstance(chart, alt.Chart)


def test_plot_corr_with_options(sample_data):
    """Test plot_corr function with various options."""
    chart = plot_corr(
        data=sample_data,
        columns=["numeric1", "numeric2"],
        mask_diag=False,
        annot_color="red",
        round_sgn=3,
        font_size=12,
        opacity=0.7,
    )

    assert isinstance(chart, alt.LayerChart)
