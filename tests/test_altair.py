"""Tests for the altair visualization module."""

import logging

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

# Try to import altair, skip tests if not available
try:
    import altair as alt

    from src.scikit_na.altair._altair import (
        plot_corr,
        plot_heatmap,
        plot_hist,
        plot_kde,
        plot_scatter,
        plot_stairbars,
        plot_stairs,
        view_dist,
    )

    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    chart = plot_scatter(data=sample_data, x_col="numeric1", y_col="numeric2", col_na="numeric1_na")

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
    except (ImportError, AttributeError) as e:
        pytest.skip(f"view_dist skipped due to dependency issue: {e}")
    except (ValueError, TypeError) as e:
        pytest.fail(f"view_dist failed with data/parameter error: {e}")
    except Exception as e:
        logger.exception("Unexpected error occurred while testing view_dist function")
        pytest.fail(f"view_dist raised an unexpected exception: {e}")


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


def test_plot_hist_applies_chart_options_and_merges_mark_defaults(sample_data):
    chart = plot_hist(
        data=sample_data,
        col="numeric1",
        col_na="numeric1_na",
        chart_kws={"title": "Custom histogram"},
        markbar_kws={"color": "blue"},
    )

    spec = chart.to_dict(validate=True)
    assert spec["title"] == "Custom histogram"
    assert spec["mark"]["color"] == "blue"
    assert spec["mark"]["opacity"] == 0.5


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


def test_custom_marks_without_opacity_are_valid(sample_data):
    kde = plot_kde(sample_data, "numeric1", "numeric1_na", markarea_kws={"clip": True})
    scatter = plot_scatter(sample_data, "numeric1", "numeric2", "numeric1_na", circle_kws={"size": 30})

    assert kde.to_dict(validate=True)["mark"]["opacity"] == 0.5
    assert scatter.to_dict(validate=True)["mark"]["opacity"] == 0.5


def test_plot_stairbars_serializes_with_valid_bar_options(sample_data):
    spec = plot_stairbars(sample_data).to_dict(validate=True)

    assert spec["mark"]["type"] == "bar"


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


# --- Regression tests -------------------------------------------------------


@pytest.fixture(name="mixed_dtypes_data")
def fixture_mixed_dtypes_data():
    """DataFrame covering the dtypes the histogram heuristic must distinguish."""
    return DataFrame(
        {
            "na_col": [1.0, None, 3.0, 4.0, None, 6.0],
            "floats": [0.1, 1.7, 2.3, 3.9, 4.4, 5.2],
            "small_ints": [1, 2, 3, 2, 1, 3],
            "strings": pd.array(["x", "y", None, "z", "x", "y"], dtype="string"),
            "categories": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
            "objects": ["p", "q", "r", "p", "q", "r"],
            "bools": [True, False, True, False, True, False],
        }
    )


@pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")
def test_plot_hist_does_not_mutate_caller_x_kws(mixed_dtypes_data):
    """The heuristic must not write back into the dict the caller passed in."""
    x_kws = {"title": "Custom title"}
    plot_hist(mixed_dtypes_data, col="floats", col_na="na_col", x_kws=x_kws)

    assert x_kws == {"title": "Custom title"}


@pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")
def test_plot_hist_explicit_x_kws_override_heuristic(mixed_dtypes_data):
    """Explicitly passed `bin`/`type` win over whatever the heuristic picked."""
    chart = plot_hist(
        mixed_dtypes_data,
        col="floats",
        col_na="na_col",
        x_kws={"bin": False, "type": "ordinal"},
    )
    encoding = chart.encoding.x.to_dict()

    assert encoding["bin"] is False
    assert encoding["type"] == "ordinal"


@pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")
@pytest.mark.parametrize(
    ("col", "expected_type", "expected_bin"),
    [
        ("strings", "nominal", False),
        ("categories", "nominal", False),
        ("objects", "nominal", False),
        ("bools", "nominal", False),
        ("small_ints", "ordinal", False),
        ("floats", "quantitative", True),
    ],
)
def test_plot_hist_heuristic_classifies_dtypes(mixed_dtypes_data, col, expected_type, expected_bin):
    """Nominal dtypes must never be binned onto a quantitative axis."""
    encoding = plot_hist(mixed_dtypes_data, col=col, col_na="na_col").encoding.x.to_dict()

    assert encoding["type"] == expected_type
    assert encoding["bin"] is expected_bin


@pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")
@pytest.mark.parametrize("columns_name", ["index", "variable", "value", "anything", None])
def test_plot_corr_handles_named_columns_index(sample_data, columns_name):
    """Reshaping must not clash with a name carried on the columns index."""
    data = sample_data.copy()
    data.columns.name = columns_name

    assert isinstance(plot_corr(data), (alt.Chart, alt.LayerChart))


@pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")
@pytest.mark.parametrize("colliding_name", ["Rows", "Columns", "Values"])
def test_plot_heatmap_handles_columns_named_like_labels(sample_data, colliding_name):
    """A column named after an axis label must not break the melt."""
    data = sample_data.rename(columns={sample_data.columns[0]: colliding_name})

    assert isinstance(plot_heatmap(data), alt.Chart)


@pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")
@pytest.mark.parametrize(("droppable", "expected"), [(True, 3), (False, 2)])
def test_plot_heatmap_color_scale_domain_matches_range(sample_data, droppable, expected):
    """A colour scale with more range entries than domain entries is invalid."""
    scale = plot_heatmap(sample_data, droppable=droppable).encoding.color.to_dict()["scale"]

    assert len(scale["domain"]) == expected
    assert len(scale["range"]) == expected


@pytest.mark.skipif(not ALTAIR_AVAILABLE, reason="Altair is not available")
def test_view_dist_restricts_na_columns_to_selection():
    """The NA dropdown must only offer columns from the `columns` selection."""
    data = DataFrame(
        {
            "selected_na": [1.0, None, 3.0],
            "selected_full": [1.0, 2.0, 3.0],
            "excluded_na": [None, None, 3.0],
        }
    )

    widget = view_dist(data, columns=["selected_na", "selected_full"])
    na_options = widget.widget.children[1].options

    assert "excluded_na" not in na_options
    assert "selected_na" in na_options
