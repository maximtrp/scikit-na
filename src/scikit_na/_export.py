"""Export functionality for reports and visualizations."""

from __future__ import annotations

__all__ = ["export_report", "export_summary"]

import json
from pathlib import Path
from typing import Dict, Literal

import pandas as pd
from pandas import DataFrame

from ._stats import correlate, describe, summary

ExportFormat = Literal["csv", "json", "html", "xlsx"]


def export_summary(
    data: DataFrame,
    output_path: str | Path,
    format: ExportFormat = "csv",
    columns: list | None = None,
    per_column: bool = True,
    round_dec: int = 2,
) -> None:
    """Export summary statistics to various formats.

    Parameters
    ----------
    data : DataFrame
        Input data.
    output_path : Union[str, Path]
        Path to save the exported file.
    format : ExportFormat, optional
        Output format: 'csv', 'json', 'html', 'xlsx'.
    columns : Optional[list], optional
        Columns to include in summary.
    per_column : bool, optional
        Show stats per each selected column.
    round_dec : int, optional
        Number of decimals for rounding.

    Raises
    ------
    ValueError
        If format is not supported.

    """
    output_path = Path(output_path)

    # Generate summary
    summary_df = summary(data, columns=columns, per_column=per_column, round_dec=round_dec)

    if format == "csv":
        summary_df.to_csv(output_path)
    elif format == "json":
        summary_df.to_json(output_path, orient="index", indent=2)
    elif format == "html":
        summary_df.to_html(output_path, table_id="na-summary")
    elif format == "xlsx":
        summary_df.to_excel(output_path, sheet_name="NA Summary")
    else:
        raise ValueError(f"Unsupported format: {format}. Use one of: csv, json, html, xlsx")


def export_report(
    data: DataFrame,
    output_dir: str | Path,
    columns: list | None = None,
    round_dec: int = 2,
    include_correlations: bool = True,
    include_descriptions: bool = True,
) -> Dict[str, Path]:
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
    Dict[str, Path]
        Dictionary mapping report sections to file paths.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            print(f"Warning: Could not generate correlations: {e}")

    if include_descriptions:
        # Descriptive statistics for columns with most NAs
        try:
            if columns is None:
                cols = data.columns.tolist()
            else:
                cols = columns

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
            print(f"Warning: Could not generate descriptive statistics: {e}")

    # Generate a summary report
    report_summary = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "dataset_shape": data.shape,
        "columns_analyzed": columns if columns else data.columns.tolist(),
        "total_missing_values": int(data.isna().sum().sum()),
        "missing_percentage": float(data.isna().sum().sum() / (data.shape[0] * data.shape[1]) * 100),
        "exported_files": {k: str(v) for k, v in exported_files.items()},
    }

    report_path = output_dir / "report_summary.json"
    with open(report_path, "w") as f:
        json.dump(report_summary, f, indent=2)
    exported_files["report_summary"] = report_path

    return exported_files

