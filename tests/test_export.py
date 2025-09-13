"""Tests for export functionality."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import pytest
from pandas import DataFrame

from src.scikit_na._export import export_summary, export_report


@pytest.fixture(name="sample_data_with_na")
def fixture_sample_data_with_na():
    """Create sample data with missing values for testing."""
    np.random.seed(42)  # For reproducible tests
    data = {
        'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'B': [np.nan, 2.2, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, 10.1],
        'C': ['a', 'b', 'c', np.nan, 'e', 'f', 'g', np.nan, 'i', 'j'],
        'D': [1, 1, 0, 1, 0, np.nan, 1, 0, 1, 0],
        'E': [10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8, 90.9, 100.0]  # No NAs
    }
    return DataFrame(data)


class TestExportSummary:
    """Test cases for export_summary function."""
    
    def test_export_summary_csv(self, sample_data_with_na):
        """Test exporting summary to CSV format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.csv"
            
            export_summary(sample_data_with_na, output_path, format="csv")
            
            assert output_path.exists()
            
            # Read back and verify content
            result_df = pd.read_csv(output_path, index_col=0)
            assert not result_df.empty
            assert 'A' in result_df.columns
            assert 'B' in result_df.columns
            assert 'na_count' in result_df.index
    
    def test_export_summary_json(self, sample_data_with_na):
        """Test exporting summary to JSON format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.json"
            
            export_summary(sample_data_with_na, output_path, format="json")
            
            assert output_path.exists()
            
            # Read back and verify content
            with open(output_path, 'r') as f:
                result_data = json.load(f)
            assert isinstance(result_data, dict)
            # JSON structure has row names as keys and columns as nested dict
            assert 'na_count' in result_data
            assert 'A' in result_data['na_count']
    
    def test_export_summary_html(self, sample_data_with_na):
        """Test exporting summary to HTML format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.html"
            
            export_summary(sample_data_with_na, output_path, format="html")
            
            assert output_path.exists()
            
            # Read back and verify content
            with open(output_path, 'r') as f:
                content = f.read()
            assert '<table' in content
            assert 'na-summary' in content
    
    def test_export_summary_xlsx(self, sample_data_with_na):
        """Test exporting summary to XLSX format."""
        pytest.importorskip("openpyxl")  # Skip if openpyxl not available
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.xlsx"
            
            export_summary(sample_data_with_na, output_path, format="xlsx")
            
            assert output_path.exists()
            
            # Read back and verify content
            result_df = pd.read_excel(output_path, sheet_name="NA Summary", index_col=0)
            assert not result_df.empty
            assert 'A' in result_df.columns
    
    def test_export_summary_with_columns(self, sample_data_with_na):
        """Test exporting summary with specific columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.csv"
            
            export_summary(
                sample_data_with_na, 
                output_path, 
                format="csv", 
                columns=['A', 'B']
            )
            
            assert output_path.exists()
            result_df = pd.read_csv(output_path, index_col=0)
            assert 'A' in result_df.columns
            assert 'B' in result_df.columns
            assert 'C' not in result_df.columns
    
    def test_export_summary_dataset_level(self, sample_data_with_na):
        """Test exporting dataset-level summary."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.csv"
            
            export_summary(
                sample_data_with_na, 
                output_path, 
                format="csv", 
                per_column=False
            )
            
            assert output_path.exists()
            result_df = pd.read_csv(output_path, index_col=0)
            assert 'dataset' in result_df.columns
            assert 'total_cols' in result_df.index
    
    def test_export_summary_invalid_format(self, sample_data_with_na):
        """Test error handling for invalid format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.txt"
            
            with pytest.raises(ValueError, match="Unsupported format"):
                export_summary(sample_data_with_na, output_path, format="txt")
    
    def test_export_summary_string_path(self, sample_data_with_na):
        """Test using string path instead of Path object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/test_summary.csv"
            
            export_summary(sample_data_with_na, output_path, format="csv")
            
            assert Path(output_path).exists()


class TestExportReport:
    """Test cases for export_report function."""
    
    def test_export_report_basic(self, sample_data_with_na):
        """Test basic export_report functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(sample_data_with_na, output_dir)
            
            # Check that files were created
            assert isinstance(exported_files, dict)
            assert "summary" in exported_files
            assert "dataset_summary" in exported_files
            assert "report_summary" in exported_files
            
            # Verify files exist
            for file_path in exported_files.values():
                assert Path(file_path).exists()
    
    def test_export_report_with_correlations(self, sample_data_with_na):
        """Test export_report with correlations included."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(
                sample_data_with_na, 
                output_dir, 
                include_correlations=True
            )
            
            # Should include correlations if there are columns with NAs
            if "correlations" in exported_files:
                assert Path(exported_files["correlations"]).exists()
    
    def test_export_report_with_descriptions(self, sample_data_with_na):
        """Test export_report with descriptive statistics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(
                sample_data_with_na, 
                output_dir, 
                include_descriptions=True
            )
            
            # Should include descriptive stats if there are columns with NAs
            if "descriptive_stats" in exported_files:
                assert Path(exported_files["descriptive_stats"]).exists()
    
    def test_export_report_specific_columns(self, sample_data_with_na):
        """Test export_report with specific columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(
                sample_data_with_na, 
                output_dir, 
                columns=['A', 'B', 'C']
            )
            
            # Verify summary file contains only specified columns
            summary_df = pd.read_csv(exported_files["summary"], index_col=0)
            assert 'A' in summary_df.columns
            assert 'B' in summary_df.columns
            assert 'C' in summary_df.columns
            assert 'D' not in summary_df.columns
    
    def test_export_report_no_correlations(self, sample_data_with_na):
        """Test export_report with correlations disabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(
                sample_data_with_na, 
                output_dir, 
                include_correlations=False
            )
            
            assert "correlations" not in exported_files
    
    def test_export_report_no_descriptions(self, sample_data_with_na):
        """Test export_report with descriptions disabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(
                sample_data_with_na, 
                output_dir, 
                include_descriptions=False
            )
            
            assert "descriptive_stats" not in exported_files
    
    def test_export_report_creates_directory(self, sample_data_with_na):
        """Test that export_report creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "new_directory"
            assert not output_dir.exists()
            
            exported_files = export_report(sample_data_with_na, output_dir)
            
            assert output_dir.exists()
            assert output_dir.is_dir()
            assert len(exported_files) > 0
    
    def test_export_report_summary_json(self, sample_data_with_na):
        """Test that report_summary.json contains expected structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(sample_data_with_na, output_dir)
            
            # Read and verify report summary
            with open(exported_files["report_summary"], 'r') as f:
                report_summary = json.load(f)
            
            assert "analysis_date" in report_summary
            assert "dataset_shape" in report_summary
            assert "columns_analyzed" in report_summary
            assert "total_missing_values" in report_summary
            assert "missing_percentage" in report_summary
            assert "exported_files" in report_summary
            
            # Verify data types and values
            assert isinstance(report_summary["dataset_shape"], list)
            assert len(report_summary["dataset_shape"]) == 2
            assert isinstance(report_summary["total_missing_values"], int)
            assert isinstance(report_summary["missing_percentage"], float)
            assert report_summary["total_missing_values"] > 0
    
    def test_export_report_string_path(self, sample_data_with_na):
        """Test export_report with string path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            exported_files = export_report(sample_data_with_na, tmp_dir)
            
            assert len(exported_files) > 0
            for file_path in exported_files.values():
                assert Path(file_path).exists()


class TestExportEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_export_summary_empty_dataframe(self):
        """Test export_summary with empty DataFrame."""
        empty_df = DataFrame()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "empty_summary.csv"
            
            # Should not raise an error but create an empty file
            export_summary(empty_df, output_path, format="csv")
            assert output_path.exists()
    
    def test_export_summary_no_missing_values(self):
        """Test export_summary with DataFrame that has no missing values."""
        df_no_na = DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "no_na_summary.csv"
            
            export_summary(df_no_na, output_path, format="csv")
            assert output_path.exists()
            
            result_df = pd.read_csv(output_path, index_col=0)
            # Should show 0 missing values
            na_counts = result_df.loc['na_count']
            assert all(count == 0 for count in na_counts)
    
    def test_export_report_no_missing_values(self):
        """Test export_report with DataFrame that has no missing values."""
        df_no_na = DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            exported_files = export_report(df_no_na, output_dir)
            
            # Should still create basic files
            assert "summary" in exported_files
            assert "dataset_summary" in exported_files
            assert "report_summary" in exported_files
            
            # Read report summary
            with open(exported_files["report_summary"], 'r') as f:
                report_summary = json.load(f)
            
            assert report_summary["total_missing_values"] == 0
            assert report_summary["missing_percentage"] == 0.0
    
    def test_export_summary_nonexistent_columns(self, sample_data_with_na):
        """Test export_summary with non-existent columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.csv"
            
            # Should handle non-existent columns gracefully
            with pytest.raises(KeyError):
                export_summary(
                    sample_data_with_na, 
                    output_path, 
                    format="csv",
                    columns=['nonexistent_column']
                )
    
    def test_export_functions_with_single_column(self):
        """Test export functions with single column DataFrame."""
        single_col_df = DataFrame({'A': [1, np.nan, 3, np.nan, 5]})
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test export_summary
            summary_path = Path(tmp_dir) / "single_col_summary.csv"
            export_summary(single_col_df, summary_path, format="csv")
            assert summary_path.exists()
            
            # Test export_report
            export_dir = Path(tmp_dir) / "single_col_report"
            exported_files = export_report(single_col_df, export_dir)
            assert len(exported_files) > 0
    
    def test_round_decimals_parameter(self, sample_data_with_na):
        """Test that round_dec parameter works correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_summary.csv"
            
            export_summary(
                sample_data_with_na, 
                output_path, 
                format="csv",
                round_dec=1
            )
            
            result_df = pd.read_csv(output_path, index_col=0)
            
            # Check that percentages are rounded to 1 decimal place
            na_pct_row = result_df.loc['na_pct_per_col']
            for val in na_pct_row:
                if not pd.isna(val) and val != 0:
                    # Check that value has at most 1 decimal place
                    assert len(str(val).split('.')[-1]) <= 1


class TestExportIntegration:
    """Integration tests for export functionality."""
    
    def test_export_with_real_dataset(self):
        """Test export functionality with a more realistic dataset."""
        # Create a dataset similar to Titanic with various data types and patterns
        np.random.seed(42)
        n_rows = 100
        
        data = {
            'passenger_id': range(1, n_rows + 1),
            'age': np.random.normal(30, 10, n_rows),
            'fare': np.random.exponential(30, n_rows),
            'sex': np.random.choice(['male', 'female'], n_rows),
            'embarked': np.random.choice(['S', 'C', 'Q'], n_rows),
            'survived': np.random.choice([0, 1], n_rows)
        }
        
        # Introduce missing values with patterns
        df = DataFrame(data)
        df.loc[np.random.choice(df.index, 20, replace=False), 'age'] = np.nan
        df.loc[np.random.choice(df.index, 5, replace=False), 'embarked'] = np.nan
        df.loc[np.random.choice(df.index, 2, replace=False), 'fare'] = np.nan
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Export comprehensive report
            exported_files = export_report(
                df, 
                output_dir,
                include_correlations=True,
                include_descriptions=True,
                round_dec=3
            )
            
            # Verify all expected files were created
            expected_files = ['summary', 'dataset_summary', 'report_summary']
            for file_key in expected_files:
                assert file_key in exported_files
                assert Path(exported_files[file_key]).exists()
            
            # Verify report summary content
            with open(exported_files["report_summary"], 'r') as f:
                report_summary = json.load(f)
            
            assert report_summary["dataset_shape"] == [n_rows, 6]
            assert report_summary["total_missing_values"] > 0
            assert 0 < report_summary["missing_percentage"] < 100
            assert len(report_summary["columns_analyzed"]) == 6