"""Export functionality for missing data analysis reports and visualizations.

This module provides comprehensive export capabilities for missing data analysis results,
supporting multiple file formats and automated report generation. It enables researchers
to save their analysis outputs for documentation, sharing, and further processing.
"""

from __future__ import annotations

__all__ = ["export_report", "export_summary"]

import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame

from ._stats import correlate, describe, summary

logger = logging.getLogger(__name__)

ExportFormat = Literal["csv", "json", "html", "xlsx"]


def export_summary(
    data: DataFrame,
    output_path: str | Path,
    format: ExportFormat = "csv",  # noqa: A002
    columns: list | None = None,
    per_column: bool = True,
    round_dec: int = 2,
) -> None:
    """Export missing data summary statistics to various file formats.

    Generates and saves comprehensive summary statistics about missing data patterns
    in multiple formats for documentation, reporting, or further analysis. The function
    uses the summary() function internally and saves results in the specified format.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame to analyze and export summary statistics for.
    output_path : str or Path
        File path where the summary statistics will be saved. The file extension
        is automatically added if not provided, based on the format parameter.
    format : {'csv', 'json', 'html', 'xlsx'}, default 'csv'
        Output file format:
        - 'csv': Comma-separated values, easily imported into spreadsheet applications
        - 'json': JavaScript Object Notation, suitable for web applications
        - 'html': HTML table format, ready for web display or documentation
        - 'xlsx': Excel format, preserves formatting and supports multiple sheets
    columns : list, optional
        Specific column names to include in the summary analysis. If None,
        includes all columns from the DataFrame.
    per_column : bool, default True
        Controls summary granularity:
        - True: Detailed statistics for each individual column
        - False: Aggregate statistics for the entire dataset
    round_dec : int, default 2
        Number of decimal places for rounding numerical results in the output.

    Raises
    ------
    ValueError
        If the specified format is not supported. Supported formats are:
        'csv', 'json', 'html', 'xlsx'.
    IOError
        If the output path is not writable or the directory doesn't exist.

    Examples
    --------
    Export basic CSV summary:

    >>> import pandas as pd
    >>> import scikit_na as na
    >>> data = pd.DataFrame({
    ...     'A': [1, None, 3, 4, 5],
    ...     'B': [1, 2, None, None, 5],
    ...     'C': [1, 2, 3, 4, 5]
    ... })
    >>> na.export_summary(data, 'missing_data_summary.csv')

    Export detailed HTML report:

    >>> na.export_summary(data, 'summary.html', format='html', per_column=True)

    Export aggregate Excel summary:

    >>> na.export_summary(data,
    ...                   'dataset_overview.xlsx',
    ...                   format='xlsx',
    ...                   per_column=False,
    ...                   round_dec=3)

    Export JSON for specific columns:

    >>> na.export_summary(data,
    ...                   'selected_columns.json',
    ...                   format='json',
    ...                   columns=['A', 'B'])

    Notes
    -----
    - CSV format is most compatible across platforms and applications
    - HTML format includes table styling and is ready for web display
    - JSON format preserves data types and is suitable for programmatic access
    - XLSX format supports rich formatting but requires pandas Excel dependencies
    - The function automatically creates parent directories if they don't exist
    """
    writers = {
        "csv": lambda df, path: df.to_csv(path),
        "json": lambda df, path: df.to_json(path, orient="index", indent=2),
        "html": lambda df, path: df.to_html(path, table_id="na-summary"),
        "xlsx": lambda df, path: df.to_excel(path, sheet_name="NA Summary"),
    }
    if format not in writers:
        raise ValueError(f"Unsupported format: {format}. Use one of: {', '.join(writers)}")

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate summary
    summary_df = summary(data, columns=columns, per_column=per_column, round_dec=round_dec)
    writers[format](summary_df, output_path)


def export_report(
    data: DataFrame,
    output_dir: str | Path,
    columns: list | None = None,
    round_dec: int = 2,
    include_correlations: bool = True,
    include_descriptions: bool = True,
) -> dict[str, Path]:
    """Export comprehensive missing data report to multiple files.

    Parameters
    ----------
    data : DataFrame
        Input data.
    output_dir : Union[str, Path]
        Directory to save exported files.
    columns : Optional[list], optional
        Columns to include in analysis.
    round_dec : int, optional
        Number of decimals for rounding.
    include_correlations : bool, optional
        Whether to include correlation analysis.
    include_descriptions : bool, optional
        Whether to include descriptive statistics.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping report sections to file paths.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzed_columns = data.columns.tolist() if columns is None else list(columns)
    analyzed_data = data.loc[:, analyzed_columns]

    exported_files = {}

    # Summary statistics
    summary_path = output_dir / "na_summary.csv"
    export_summary(data, summary_path, format="csv", columns=columns, round_dec=round_dec)
    exported_files["summary"] = summary_path

    # Dataset-level summary
    dataset_summary_path = output_dir / "dataset_summary.csv"
    export_summary(data, dataset_summary_path, format="csv", columns=columns, per_column=False, round_dec=round_dec)
    exported_files["dataset_summary"] = dataset_summary_path

    if include_correlations:
        # Correlation matrix
        try:
            corr_df = correlate(data, columns=columns)
            if not corr_df.empty:
                corr_path = output_dir / "na_correlations.csv"
                corr_df.round(round_dec).to_csv(corr_path)
                exported_files["correlations"] = corr_path
        except Exception as e:
            logger.warning("Could not generate correlations", exc_info=True)
            print(f"Warning: Could not generate correlations: {e}")

    if include_descriptions:
        # Descriptive statistics for columns with most NAs
        try:
            cols = analyzed_columns

            # Find column with most NAs
            na_counts = data[cols].isna().sum()
            if na_counts.sum() > 0:
                col_most_na = na_counts.idxmax()

                # Descriptive stats
                desc_df = describe(data, col_na=col_most_na, columns=cols)
                desc_path = output_dir / f"descriptive_stats_{col_most_na}.csv"
                desc_df.round(round_dec).to_csv(desc_path)
                exported_files["descriptive_stats"] = desc_path
        except Exception as e:
            logger.warning("Could not generate descriptive statistics", exc_info=True)
            print(f"Warning: Could not generate descriptive statistics: {e}")

    # Generate a summary report
    total_cells = analyzed_data.shape[0] * analyzed_data.shape[1]
    na_total = int(analyzed_data.isna().sum().sum())
    report_summary = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "dataset_shape": analyzed_data.shape,
        "columns_analyzed": analyzed_columns,
        "total_missing_values": na_total,
        "missing_percentage": float(na_total / total_cells * 100) if total_cells > 0 else 0.0,
        "exported_files": {k: str(v) for k, v in exported_files.items()},
    }

    report_path = output_dir / "report_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_summary, f, indent=2)
    exported_files["report_summary"] = report_path

    return exported_files
