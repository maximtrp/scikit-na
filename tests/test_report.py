"""Tests for the report module."""

import pandas as pd
import numpy as np
import pytest
from pandas import DataFrame
from unittest.mock import patch

# Try to import ipywidgets, skip tests if not available
try:
    from ipywidgets import widgets
    from src.scikit_na import report
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

# Skip all tests if ipywidgets is not available
pytestmark = pytest.mark.skipif(
    not IPYWIDGETS_AVAILABLE,
    reason="ipywidgets is not available"
)


@pytest.fixture
def sample_data():
    """Create a sample DataFrame with mixed data types for testing."""
    np.random.seed(42)
    df = DataFrame({
        'numeric1': np.random.normal(0, 1, 100),
        'numeric2': np.random.normal(5, 2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some NAs
    df.loc[0:10, 'numeric1'] = np.nan
    df.loc[20:30, 'numeric2'] = np.nan
    df.loc[40:50, 'category'] = np.nan
    
    # Add NA indicator columns
    df['numeric1_na'] = df['numeric1'].isna()
    df['numeric2_na'] = df['numeric2'].isna()
    df['category_na'] = df['category'].isna()
    
    return df


@patch('src.scikit_na._report.describe')
@patch('src.scikit_na._report.summary')
def test_report_basic(mock_summary, mock_describe, sample_data):
    """Test basic functionality of the report function."""
    # Mock return values
    mock_summary.return_value = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    mock_describe.return_value = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    
    # Call the report function
    tab = report(data=sample_data)
    
    # Check that the returned object is an ipywidgets Tab
    assert isinstance(tab, widgets.Tab)
    
    # Check that the tab has 5 children (Summary, Visualizations, Statistics, Correlations, Distributions)
    assert len(tab.children) == 5
    
    # Check tab titles
    assert tab.get_title(0) == "Summary"
    assert tab.get_title(1) == "Visualizations"
    assert tab.get_title(2) == "Statistics"
    assert tab.get_title(3) == "Correlations"
    assert tab.get_title(4) == "Distributions"


@patch('src.scikit_na._report.describe')
@patch('src.scikit_na._report.summary')
@patch('src.scikit_na._report.plot_corr')
@patch('src.scikit_na._report.plot_stairs')
@patch('src.scikit_na._report.plot_heatmap')
@patch('src.scikit_na._report.plot_hist')
@patch('src.scikit_na._report.plot_kde')
def test_report_with_options(mock_plot_kde, mock_plot_hist, mock_plot_heatmap, 
                            mock_plot_stairs, mock_plot_corr, mock_summary, 
                            mock_describe, sample_data):
    """Test report function with various options."""
    # Mock return values
    mock_summary.return_value = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    mock_describe.return_value = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    
    # Create a custom layout
    layout = widgets.Layout(grid_template_columns="1fr 1fr", justify_items="center")
    
    # Call the report function with custom options
    tab = report(
        data=sample_data,
        columns=['numeric1', 'numeric2'],
        layout=layout,
        round_dec=3,
        corr_kws={'mask_diag': False},
        heat_kws={'cmap': 'viridis'},
        dist_kws={'na_label': 'Missing'}
    )
    
    # Check that the returned object is an ipywidgets Tab
    assert isinstance(tab, widgets.Tab)
    
    # Check that the tab has 5 children
    assert len(tab.children) == 5
    
    # Verify that the functions were called with the correct arguments
    mock_summary.assert_called()
    mock_describe.assert_called()
    mock_plot_corr.assert_called()
    mock_plot_stairs.assert_called()
    mock_plot_heatmap.assert_called() 