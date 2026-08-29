"""
Interactive visualization functions using Altair for missing data analysis.

This module provides comprehensive interactive visualizations for exploring missing data
patterns using the Altair/Vega-Lite grammar of graphics. All functions return Chart
objects that can be displayed in Jupyter notebooks or saved to various formats.

Key visualization types:
- Heatmaps for missing data patterns
- Correlation plots for missingness relationships
- Distribution plots (histograms and KDE) grouped by missingness
- Stairs plots showing cumulative impact of missing data
- Interactive widgets for dynamic exploration
"""

from __future__ import annotations

__all__ = [
    "plot_corr",
    "plot_heatmap",
    "plot_hist",
    "plot_kde",
    "plot_scatter",
    "plot_stairbars",
    "plot_stairs",
    "view_dist",
]
from collections.abc import Iterable, Sequence

from altair import (
    Axis,
    Chart,
    Color,
    LayerChart,
    Scale,
    Text,
    X,
    Y,
    condition,
    selection_point,
    value,
)
from ipywidgets import interact, widgets
from numpy import arange, fill_diagonal, nan
from pandas import DataFrame, Series
from pandas.api.types import is_bool_dtype

from .._stats import _is_nominal_series, _select_cols, correlate, stairs


def _hist_x_heuristic(values: Series, thres_uniq: int) -> dict:
    """
    Pick Altair `bin`/`type` settings for the X axis of a histogram.

    Nominal data (object, string, categorical, boolean) is never binned. Numeric
    data is binned unless it is integral with few enough distinct values, in
    which case an ordinal axis reads better.
    """
    if _is_nominal_series(values) or is_bool_dtype(values):
        return {"bin": False, "type": "nominal"}

    non_na = values.dropna()
    # `kind` is "i"/"u" for integer dtypes; a float column holding only whole
    # numbers should be treated the same way.
    is_integral = non_na.dtype.kind in "iu" or (non_na.dtype.kind == "f" and (non_na % 1 == 0).all())

    if is_integral and non_na.nunique() < thres_uniq:
        return {"bin": False, "type": "ordinal"}
    return {"bin": True, "type": "quantitative"}


def plot_hist(
    data: DataFrame,
    col: str,
    col_na: str,
    na_label: str | None = None,
    na_replace: dict | None = None,
    heuristic: bool = True,
    thres_uniq: int = 20,
    step: bool = False,
    norm: bool = True,
    font_size: int = 14,
    xlabel: str | None = None,
    ylabel: str = "Frequency",
    chart_kws: dict | None = None,
    markarea_kws: dict | None = None,
    markbar_kws: dict | None = None,
    joinagg_kws: dict | None = None,
    calc_kws: dict | None = None,
    x_kws: dict | None = None,
    y_kws: dict | None = None,
    color_kws: dict | None = None,
) -> Chart:
    """
    Histogram plot.

    Plots a histogram of values in a column `col` grouped by NA/non-NA values
    in column `col_na`.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col : str
        Column to display distribution of values.
    col_na : str
        Column to group values by.
    na_label : str, optional
        Legend title.
    na_replace : dict, optional
        Dictionary to replace values returned by
        :py:meth:`pandas.Series.isna()` method.
    heuristic : bool, default True
        Infer the Altair axis type and whether numeric values should be binned.
    thres_uniq : int, default 20
        Maximum unique-value count for treating integral data as ordinal.
    step : bool, optional
        Draw step plot.
    norm : bool, optional
        Normalize values in groups.
    font_size : int, default 14
        Font size for axis and legend labels and titles.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()`.
    markarea_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_area()`.
    markbar_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_bar()`.
    joinagg_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`altair.Chart.transform_joinaggregate()`.
    calc_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`altair.Chart.transform_calculate()`.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()`.

    Returns
    -------
    Chart
        Altair Chart object.

    """
    if not chart_kws:
        chart_kws = {}
    markarea_kws = {"opacity": 0.5, "interpolate": "step", **(markarea_kws or {})}
    markbar_kws = {"opacity": 0.5, **(markbar_kws or {})}
    if not joinagg_kws:
        joinagg_kws = {"total": "count()", "groupby": [col_na]}
    if not calc_kws:
        calc_kws = {"y": "1 / datum.total"}
    x_defaults = {"title": xlabel or col}
    if not y_kws:
        y_kws = {"type": "quantitative", "stack": None, "title": ylabel}
    if not color_kws:
        color_kws = {"title": na_label or col_na}
    if not na_replace:
        na_replace = {True: "NA", False: "Filled"}

    # Simple heuristic for choosing histplot parameters
    if heuristic:
        x_defaults.update(_hist_x_heuristic(data[col], thres_uniq))

    # Explicitly passed `x_kws` always win over the heuristic, and the caller's
    # dictionary is never mutated.
    x_kws = {**x_defaults, **(x_kws or {})}

    data_copy = data.loc[:, [col, col_na]].copy()
    data_copy[col_na] = data_copy.loc[:, col_na].isna().replace(na_replace)

    # Chart creation
    chart = Chart(data_copy, **chart_kws)

    chart = chart.mark_area(**markarea_kws) if step else chart.mark_bar(**markbar_kws)

    # Normed vs non-normed histogram
    if norm:
        y_shorthand = "sum(y)"
        chart = chart.transform_joinaggregate(**joinagg_kws)
        chart = chart.transform_calculate(**calc_kws)
    else:
        y_shorthand = "count()"

    selection = selection_point(fields=[col_na], bind="legend")
    chart = chart.encode(
        x=X(col, **x_kws),
        y=Y(y_shorthand, **y_kws),
        color=Color(col_na, **color_kws),
        tooltip=["count()"],
        opacity=condition(
            selection,
            value(markarea_kws["opacity"] if step else markbar_kws["opacity"]),
            value(0),
        ),
    ).add_params(selection)

    return chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
    )


def plot_kde(
    data: DataFrame,
    col: str,
    col_na: str,
    na_label: str | None = None,
    na_replace: dict | None = None,
    font_size: int = 14,
    xlabel: str | None = None,
    ylabel: str = "Density",
    chart_kws: dict | None = None,
    markarea_kws: dict | None = None,
    density_kws: dict | None = None,
    x_kws: dict | None = None,
    y_kws: dict | None = None,
    color_kws: dict | None = None,
) -> Chart:
    """
    Density plot.

    Plots distribution of values in a column `col` grouped by
    NA/non-NA values in column `col_na`.

    Parameters
    ----------
    data : DataFrame
        Input data.
    col : str
        Column to display distribution of values.
    col_na : str
        Column to group values by.
    na_label : str, optional
        Legend title.
    na_replace : dict, optional
        Dictionary to replace values returned by
        :py:meth:`pandas.Series.isna()` method.
    font_size : int, default 14
        Font size for axis and legend labels and titles.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()`.
    markarea_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_area()`.
    density_kws : dict, optional
        Keyword arguments passed to
        :py:meth:`altair.Chart.transform_density()`.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()`.

    Returns
    -------
    Chart
        Altair Chart object.

    """
    if not chart_kws:
        chart_kws = {}
    markarea_kws = {"opacity": 0.5, **(markarea_kws or {})}
    if not density_kws:
        density_kws = {"density": col, "groupby": [col_na], "as_": [col, ylabel]}
    if not x_kws:
        x_kws = {"title": xlabel or col}
    if not y_kws:
        y_kws = {"type": "quantitative", "stack": None, "title": ylabel}
    if not color_kws:
        color_kws = {"title": na_label or col_na}
    if not na_replace:
        na_replace = {True: "NA", False: "Filled"}

    y_shorthand = ylabel

    data_copy = data.loc[:, [col, col_na]].copy()
    data_copy[col_na] = data_copy.loc[:, col_na].isna().replace(na_replace)

    # Chart creation
    chart = Chart(data_copy, **chart_kws).mark_area(**markarea_kws)
    chart = chart.transform_density(**density_kws)

    selection = selection_point(fields=[col_na], bind="legend")
    chart = chart.encode(
        x=X(col, **x_kws),
        y=Y(y_shorthand, **y_kws),
        color=Color(col_na, **color_kws),
        opacity=condition(selection, value(markarea_kws["opacity"]), value(0)),
    ).add_params(selection)

    return chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
    )


def plot_scatter(
    data: DataFrame,
    x_col: str,
    y_col: str,
    col_na: str,
    na_label: str | None = None,
    na_replace: dict | None = None,
    font_size: int = 14,
    xlabel: str | None = None,
    ylabel: str | None = None,
    circle_kws: dict | None = None,
    color_kws: dict | None = None,
    x_kws: dict | None = None,
    y_kws: dict | None = None,
):
    """
    Scatter plot.

    Parameters
    ----------
    data : DataFrame
        Input data.
    x_col : str
        Column name corresponding to X axis.
    y_col : str
        Column name corresponding to Y axis.
    col_na : str
        Column name
    na_label : str, optional
        Label for NA values in legend.
    na_replace : dict, optional
        NA replacement mapping, by default {True: 'NA', False: 'Filled'}.
    font_size : int, optional
        Font size for plotting, by default 14.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    circle_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_circle()`.
    color_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Color()`.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()`.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()`.

    Returns
    -------
    altair.Chart
        Scatter plot.

    """
    circle_kws = {"opacity": 0.5, **(circle_kws or {})}
    if not color_kws:
        color_kws = {"title": na_label or col_na}
    if not x_kws:
        x_kws = {"title": xlabel or x_col}
    if not y_kws:
        y_kws = {"title": ylabel or y_col}
    if not na_replace:
        na_replace = {True: "NA", False: "Filled"}

    data_copy = data.loc[:, [x_col, y_col, col_na]].copy()
    data_copy[col_na] = data_copy[col_na].isna().replace(na_replace)
    base = Chart(data_copy)

    selection = selection_point(fields=[col_na], bind="legend")
    points = (
        base.mark_circle(**circle_kws)
        .encode(
            x=X(x_col, **x_kws),
            y=Y(y_col, **y_kws),
            color=Color(col_na, **color_kws),
            opacity=condition(selection, value(circle_kws["opacity"]), value(0)),
        )
        .add_params(selection)
    )

    return points.configure_axis(labelFontSize=font_size, titleFontSize=font_size).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
    )


def plot_stairs(
    data: DataFrame,
    columns: Sequence[str] | None = None,
    xlabel: str = "Columns",
    ylabel: str = "Instances",
    tooltip_label: str = "Size difference",
    dataset_label: str = "(Whole dataset)",
    font_size: int = 14,
    area_kws: dict | None = None,
    chart_kws: dict | None = None,
    x_kws: dict | None = None,
    y_kws: dict | None = None,
):
    """
    Stairs plot.

    Plots changes in dataset size (rows/instances number) after applying
    :py:meth:`pandas.DataFrame.dropna()` to each column cumulatively.

    Columns are sorted by maximum influence on dataset size.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Sequence[str]], optional
        Columns that are to be displayed on a plot.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    tooltip_label : str, optional
        Label for differences in dataset size that is displayed on a tooltip.
    dataset_label : str, optional
        Label for the whole dataset (before dropping any NAs).
    font_size : int, default 14
        Font size for axis and legend labels and titles.
    area_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_area()` method.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()` class.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()` class.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()` class.

    Returns
    -------
    altair.Chart
        Chart object.

    """
    if not area_kws:
        area_kws = {"interpolate": "step-after", "line": True}
    if not chart_kws:
        chart_kws = {}
    if not x_kws:
        x_kws = {"sort": "-y", "shorthand": xlabel}
    if not y_kws:
        y_kws = {"shorthand": ylabel}

    data_sizes = stairs(data, columns, xlabel, ylabel, tooltip_label, dataset_label)

    chart = (
        Chart(data_sizes, **chart_kws)
        .mark_area(**area_kws)
        .encode(x=X(**x_kws), y=Y(**y_kws), tooltip=[xlabel, ylabel, tooltip_label])
    )
    return chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
    )


def plot_stairbars(
    data: DataFrame,
    columns: Sequence[str] | None = None,
    xlabel: str = "Columns",
    ylabel: str = "Instances",
    tooltip_label: str = "Size difference",
    dataset_label: str = "(Whole dataset)",
    font_size: int = 14,
    area_kws: dict | None = None,
    chart_kws: dict | None = None,
    x_kws: dict | None = None,
    y_kws: dict | None = None,
):
    """
    Stairbars.

    Plots the changes in dataset size (rows/instances number) after applying
    :py:meth:`pandas.DataFrame.dropna()` to each column cumulatively.

    Columns are sorted by maximum influence on dataset size.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Sequence[str]], optional
        Columns that are to be displayed on a plot.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    tooltip_label : str, optional
        Label for differences in dataset size that is displayed on a tooltip.
    dataset_label : str, optional
        Label for the whole dataset (before dropping any NAs).
    font_size : int, default 14
        Font size for axis and legend labels and titles.
    area_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart.mark_bar()` method.
    chart_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Chart()` class.
    x_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.X()` class.
    y_kws : dict, optional
        Keyword arguments passed to :py:meth:`altair.Y()` class.

    Returns
    -------
    altair.Chart
        Chart object.

    """
    if not area_kws:
        area_kws = {}
    if not chart_kws:
        chart_kws = {}
    if not x_kws:
        x_kws = {"sort": "-y", "shorthand": xlabel}
    if not y_kws:
        y_kws = {"shorthand": ylabel}

    data_sizes = stairs(data, columns, xlabel, ylabel, tooltip_label, dataset_label)

    chart = (
        Chart(data_sizes, **chart_kws)
        .mark_bar(**area_kws)
        .encode(x=X(**x_kws), y=Y(**y_kws), tooltip=[xlabel, ylabel, tooltip_label])
    )
    return chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
    )


def plot_heatmap(
    data: DataFrame,
    columns: Sequence[str] | None = None,
    names: list | None = None,
    sort: bool = True,
    droppable: bool = True,
    font_size: int = 14,
    xlabel: str = "Columns",
    ylabel: str = "Rows",
    zlabel: str = "Values",
    chart_kws: dict | None = None,
    rect_kws: dict | None = None,
    x_kws: dict | None = None,
    y_kws: dict | None = None,
    color_kws: dict | None = None,
) -> Chart:
    """
    Create interactive heatmap visualization of missing data patterns.

    Generates a color-coded heatmap where each cell represents a data point,
    showing the pattern of missing values across rows and columns. This
    visualization is essential for understanding:
    - Overall distribution of missing values
    - Systematic patterns in data collection
    - Which rows would be affected by listwise deletion
    - Relationships between missing values in different columns

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame to visualize missing data patterns.
    columns : Sequence[str], optional
        Specific column names to include in the visualization. If None,
        includes all columns in the DataFrame.
    names : list, optional
        Custom labels for the legend categories, provided as a list with:
        - names[0]: Label for non-missing values (default: "Filled")
        - names[1]: Label for missing values (default: "NA")
        - names[2]: Label for droppable values (default: "Droppable")
        Only first two elements are used if droppable=False.
    sort : bool, default True
        If True, sorts columns by number of missing values (most missing first)
        and rows by missing value patterns for better visual clustering.
    droppable : bool, default True
        If True, highlights non-missing values in rows that contain at least
        one missing value (i.e., values that would be lost with listwise deletion).
        This helps understand the impact of complete case analysis.
    font_size : int, default 14
        Font size for axis labels and legend text.
    xlabel : str, default "Columns"
        Label for the x-axis (column names).
    ylabel : str, default "Rows"
        Label for the y-axis (row indices).
    zlabel : str, default "Values"
        Title for the color legend showing value categories.
    chart_kws : dict, optional
        Additional keyword arguments passed to altair.Chart() constructor.
        Common options: {'width': int, 'height': int, 'title': str}
    rect_kws : dict, optional
        Keyword arguments for altair.Chart.mark_rect() to customize rectangles.
        Common options: {'stroke': str, 'strokeWidth': float}
    x_kws : dict, optional
        Keyword arguments for altair.X() encoding of the x-axis.
    y_kws : dict, optional
        Keyword arguments for altair.Y() encoding of the y-axis.
    color_kws : dict, optional
        Keyword arguments for altair.Color() encoding, including custom color scales.

    Returns
    -------
    altair.Chart
        Interactive Altair Chart object that can be:
        - Displayed directly in Jupyter notebooks
        - Saved to various formats (PNG, SVG, HTML, JSON)
        - Further customized with additional Altair methods

    Examples
    --------
    Basic missing data heatmap:

    >>> import pandas as pd
    >>> import scikit_na as na
    >>> data = pd.DataFrame({
    ...     'A': [1, None, 3, None, 5],
    ...     'B': [1, 2, None, 4, None],
    ...     'C': [None, None, 3, 4, 5]
    ... })
    >>> chart = na.altair.plot_heatmap(data)
    >>> chart.show()

    Focus on specific columns without sorting:

    >>> chart = na.altair.plot_heatmap(data,
    ...                                columns=['A', 'B'],
    ...                                sort=False)

    Simplified view without droppable values:

    >>> chart = na.altair.plot_heatmap(data,
    ...                                droppable=False,
    ...                                names=['Available', 'Missing'])

    Customized appearance:

    >>> chart = na.altair.plot_heatmap(
    ...     data,
    ...     chart_kws={'width': 400, 'height': 300, 'title': 'Missing Data Pattern'},
    ...     color_kws={'scale': {'range': ['lightblue', 'red', 'orange']}},
    ...     font_size=12
    ... )

    Save to file:

    >>> chart = na.altair.plot_heatmap(data)
    >>> chart.save('missing_data_heatmap.png')

    Notes
    -----
    - Green typically represents filled/non-missing values
    - Red represents missing (NA) values
    - Orange represents "droppable" values (non-missing values in incomplete rows)
    - Sorting helps identify systematic missing data patterns
    - The droppable category shows the collateral damage from listwise deletion
    - Interactive features allow zooming and tooltips for detailed inspection
    - Large datasets may require adjusting chart dimensions via chart_kws

    See Also
    --------
    plot_stairs : Visualize cumulative impact of missing data
    plot_corr : Correlation heatmap for missing value patterns
    summary : Numerical summary of missing data patterns

    """
    if not chart_kws:
        chart_kws = {"height": 300}
    if not x_kws:
        x_kws = {"sort": None, "shorthand": xlabel, "type": "nominal"}
    if not y_kws:
        y_kws = {
            "sort": None,
            "shorthand": ylabel,
            "type": "ordinal",
            "axis": Axis(labelOverlap="greedy"),
        }
    if not names:
        names = ["Filled", "NA", "Droppable"]
    if not color_kws:
        colors = ["green", "red", "orange"]
        domain = names if droppable else names[0:2]
        color_kws = {
            "shorthand": zlabel,
            "type": "nominal",
            "scale": Scale(domain=domain, range=colors[: len(domain)]),
        }
    if not rect_kws:
        rect_kws = {"clip": True}

    cols = _select_cols(data, columns)

    data_copy = data.loc[:, cols].isna()
    if sort:
        cols_sorted = data_copy.sum().sort_values(ascending=False).index.tolist()
        data_copy = data_copy.sort_values(by=cols_sorted)
        x_kws = {**x_kws, "sort": cols_sorted}

    if droppable:
        non_na_mask = ~data_copy.to_numpy()
        na_rows_mask = data_copy.any(axis=1).to_numpy()[:, None]
        droppable_mask = non_na_mask & na_rows_mask
        data_copy = data_copy.astype(int).mask(droppable_mask, other=2)
    else:
        data_copy = data_copy.astype(int)

    data_copy = data_copy.replace(dict(zip([0, 1, 2], names)))

    # Reshape under private names first: a selected column may well be called
    # "Rows"/"Columns"/"Values", which melt() would reject as a clash.
    row_key = "__row__"
    data_copy[row_key] = arange(data_copy.shape[0])
    data_copy = data_copy.melt(
        id_vars=[row_key],
        value_vars=list(cols),
        var_name="__column__",
        value_name="__value__",
    )
    data_copy.columns = [ylabel, xlabel, zlabel]

    chart = (
        Chart(data_copy, **chart_kws)
        .mark_rect(**rect_kws)
        .encode(
            x=X(**x_kws),
            y=Y(**y_kws),
            color=Color(**color_kws),
        )
    )

    return chart.configure_axis(labelFontSize=font_size, titleFontSize=font_size).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
    )


def plot_corr(
    data: DataFrame,
    columns: Iterable[str] | None = None,
    mask_diag: bool = True,
    annot_color: str = "black",
    round_sgn: int = 2,
    font_size: int = 14,
    opacity: float = 0.5,
    corr_kws: dict | None = None,
    chart_kws: dict | None = None,
    x_kws: dict | None = None,
    y_kws: dict | None = None,
    color_kws: dict | None = None,
    text_kws: dict | None = None,
) -> LayerChart:
    """
    Correlation heatmap.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Sequence[str]]
        Columns names.
    mask_diag : bool = True
        Mask diagonal on heatmap.
    annot_color : str, default "black"
        Color of the correlation-value annotations.
    round_sgn : int, default 2
        Number of decimal places displayed in annotations.
    font_size : int, default 14
        Font size for labels, titles, and annotations.
    opacity : float, default 0.5
        Opacity of heatmap rectangles.
    corr_kws : dict, optional
        Keyword arguments passed to :py:meth:`pandas.DataFrame.corr()` method.
    chart_kws : dict, optional
        Keyword arguments passed to :py:class:`altair.Chart`.
    x_kws : dict, optional
        Keyword arguments passed to :py:class:`altair.X`.
    y_kws : dict, optional
        Keyword arguments passed to :py:class:`altair.Y`.
    color_kws : dict, optional
        Keyword arguments passed to :py:class:`altair.Color`.
    text_kws : dict, optional
        Keyword arguments passed to :py:class:`altair.Text`.

    Returns
    -------
    altair.Chart
        Altair Chart object.

    """
    if not corr_kws:
        corr_kws = {"method": "spearman"}
    if not chart_kws:
        chart_kws = {}
    if not x_kws:
        x_kws = {"shorthand": "variable", "title": ""}
    if not y_kws:
        y_kws = {"shorthand": "index", "title": ""}
    if not color_kws:
        color_kws = {
            "shorthand": "value:Q",
            "title": "Correlation",
            "scale": Scale(scheme="redblue", domain=[-1, 1], reverse=True),
        }
    if not text_kws:
        text_kws = {"shorthand": "value:Q", "format": f".{round_sgn}f"}

    cols = _select_cols(data, columns)

    data_corr = correlate(data, columns=cols, **corr_kws)

    if mask_diag:
        corr_values = data_corr.to_numpy(copy=True)
        fill_diagonal(corr_values, nan)
        data_corr = DataFrame(corr_values, index=data_corr.index, columns=data_corr.columns)
    # Drop any index/column names first: they would otherwise collide with the
    # "index"/"variable"/"value" names that the reshaping below relies on.
    data_corr = data_corr.rename_axis(index=None, columns=None)
    data_corr_melt = (
        data_corr.rename_axis(index="index")
        .reset_index()
        .melt(
            id_vars=["index"],
            var_name="variable",
            value_name="value",
        )
    )

    base = Chart(data_corr_melt, **chart_kws).encode(x=X(**x_kws), y=Y(**y_kws))

    heatmap = base.mark_rect().encode(color=Color(**color_kws))
    text = base.mark_text(baseline="middle").encode(text=Text(**text_kws))

    # Draw the chart
    return (
        (heatmap + text)
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
        .configure_text(fontSize=font_size, color=annot_color)
        .configure_rect(opacity=opacity)
    )


def view_dist(data: DataFrame, columns: Sequence[str] | None = None, **kwargs):
    """
    Interactive distribution widget.

    Interactively observe distribution of values in a selected column
    grouped by NA/non-NA values in another column.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Sequence[str]]
        Column names.

    Returns
    -------
    _InteractFactory
        Interactive widget.

    """
    cols = _select_cols(data, columns)
    na_counts = data.loc[:, cols].isna().sum(axis=0)
    na_cols = na_counts.index[na_counts > 0].to_numpy()

    return interact(
        lambda Column, NA: (  # noqa: N803
            plot_hist(data, col=Column, col_na=NA, **kwargs)
            if Column != NA
            else widgets.HTML('<em style="color: red">Note: select different columns</em>')
        ),
        Column=cols,
        NA=na_cols,
    )
