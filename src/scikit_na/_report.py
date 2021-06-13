"""Interactive report."""
__all__ = ['report']
from typing import Optional, Iterable
from ipywidgets import widgets
from pandas import DataFrame
from numpy import array, random, setdiff1d
from .altair import plot_corr, plot_hist, plot_stairs, plot_heatmap, plot_kde
from ._stats import (
    describe, summary, _select_cols, _get_nominal_cols, _get_numeric_cols)


def report(
        data: DataFrame,
        columns: Optional[Iterable] = None,
        layout: widgets.Layout = None,
        round_dec: int = 2,
        corr_kws: dict = None,
        heat_kws: dict = None,
        dist_kws: dict = None):
    """Interactive report.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Optional[Iterable], optional
        Columns names.
    layout : widgets.Layout, optional
        Layout object for use in GridBox.
    round_dec : int, optional
        Number of decimals for rounding.
    corr_kws : dict, optional
        Keyword arguments passed to :py:meth:`scikit_na.altair.plot_corr()`.
    heat_kws : dict, optional
        Keyword arguments passed to :py:meth:`scikit_na.altair.plot_heatmap()`.
    hist_kws : dict, optional
        Keyword arguments passed to :py:meth:`scikit_na.altair.plot_hist()`.

    Returns
    -------
    widgets.Tab
        Interactive report with multiple tabs.
    """
    from IPython.display import display

    if not corr_kws:
        corr_kws = {}
    if not heat_kws:
        heat_kws = {}
    if not dist_kws:
        dist_kws = {}

    cols = _select_cols(data, columns).tolist()
    na_cols = data.loc[:, cols].isna()\
        .sum(axis=0)\
        .rename('na_num')\
        .to_frame()\
        .query('na_num > 0')\
        .index.values

    layout = widgets.Layout(
        grid_template_columns='1fr 1fr',
        justify_items='center') if not layout else layout
    tab = widgets.Tab()

    # SUMMARY TAB
    # Table with per column summary
    summary_table = widgets.Output()
    summary_table.append_display_data(
        summary(data, columns=cols, per_column=True).round(round_dec))
    summary_table_accordion = widgets.Accordion(children=[summary_table])
    summary_table_accordion.set_title(0, 'NA summary (per each column)')
    summary_table_accordion.selected_index = 0

    # Columns selection
    def _on_col_select(names):
        summary_table.clear_output(wait=False)
        total_summary_table.clear_output(wait=False)
        with summary_table:
            display(
                summary(data, columns=array(names['new']), per_column=True)
                .round(round_dec))
        with total_summary_table:
            display(
                summary(data, columns=array(names['new']), per_column=False)
                .round(round_dec))
    select_cols = widgets.SelectMultiple(options=cols, rows=6)
    select_cols.observe(_on_col_select, names='value')
    select_accordion = widgets.Accordion(children=[select_cols])
    select_accordion.set_title(0, 'Columns selection')
    select_accordion.selected_index = 0

    # Table with total summary
    total_summary_table = widgets.Output()
    total_summary_table.append_display_data(
        summary(data, columns=cols, per_column=False).round(round_dec))
    total_summary_accordion = widgets.Accordion(children=[total_summary_table])
    total_summary_accordion.set_title(
        0, 'NA summary for the whole dataset (across selected columns)')

    # Finalizing summary tab
    summary_tab = widgets.VBox(
        [select_accordion, summary_table_accordion, total_summary_accordion])

    # VISUALIZATION TAB
    # Columns selection
    def _on_vis_col_select(names):
        stairs_plot.clear_output(wait=False)
        heatmap_plot.clear_output(wait=False)
        with stairs_plot:
            display(plot_stairs(data, columns=names['new']))
        with heatmap_plot:
            display(plot_heatmap(data, columns=names['new']))

    select_vis_cols = widgets.SelectMultiple(options=cols, rows=6)
    select_vis_cols.observe(_on_vis_col_select, names='value')
    select_vis_accordion = widgets.Accordion(children=[select_cols])
    select_vis_accordion.set_title(0, 'Columns selection')
    select_vis_accordion.selected_index = 0

    # Creating plots
    stairs_plot = widgets.Output()
    stairs_plot.append_display_data(
        plot_stairs(data, columns=cols))
    heatmap_plot = widgets.Output()
    heatmap_plot.append_display_data(
        plot_heatmap(data, columns=cols))

    # Joining two plots into one container
    vis_box = widgets.HBox(
        [stairs_plot, heatmap_plot])

    # Making an accordion with both plots
    vis_accordion = widgets.Accordion(children=[vis_box])
    vis_accordion.set_title(0, 'Stairs plot and heatmap of NA values')
    vis_accordion.selected_index = 0

    # Finalizing tab
    vis_tab = widgets.VBox(
        [select_vis_accordion, vis_accordion])

    # SUMMARY TAB
    # Choosing a column with most NAs
    col_with_most_nas = data.loc[:, cols].isna().sum()\
        .sort_values().tail(1).index.item()

    def _on_stats_col_select(_):
        # Clearing display
        stats_table.clear_output(wait=False)
        # Getting selected column with NA values
        _col_na = select_stats_col_na.value
        # Getting selected columns
        _cols_tmp = select_stats_cols.value\
            if select_stats_cols.value else cols
        _cols = setdiff1d(_cols_tmp, [_col_na])
        _cols_numeric = _get_numeric_cols(data, _cols)
        _cols_nominal = _get_nominal_cols(data, _cols)

        with stats_table:
            display(
                describe(
                    data, col_na=_col_na, columns=_cols_numeric)
                .round(round_dec))
        with stats_table2:
            display(
                describe(
                    data, col_na=_col_na, columns=_cols_nominal)
                .round(round_dec))

    # Setting up dropdown and select elements for choosing columns
    select_stats_col_na_header = widgets.HTML(
        '<b>Select a column with NAs to group values by</b>')
    select_stats_col_na = widgets.Dropdown(options=cols)
    select_stats_col_na.value = col_with_most_nas
    select_stats_cols_header = widgets.HTML(
        '<b>Select columns to calculate descriptive statistics</b>')
    select_stats_cols = widgets.SelectMultiple(options=cols, rows=6)
    select_stats_col_na.observe(_on_stats_col_select, names='value')
    select_stats_cols.observe(_on_stats_col_select, names='value')

    selects_stats = widgets.GridBox([
        widgets.VBox(
            [select_stats_cols_header, select_stats_cols],
            layout={'align_items': 'center'}),
        widgets.VBox(
            [select_stats_col_na_header, select_stats_col_na],
            layout={'align_items': 'center'}),
    ], layout=layout)

    select_accordion = widgets.Accordion(children=[selects_stats])
    select_accordion.set_title(0, 'Columns selection')
    select_accordion.selected_index = 0

    # Statistics for numeric data
    cols_with_num_data = _get_numeric_cols(data, cols)
    stats_table = widgets.Output()
    stats_table.append_display_data(
        describe(
            data, col_na=col_with_most_nas,
            columns=setdiff1d(cols_with_num_data, [col_with_most_nas]))
        .round(round_dec))
    stats_table_accordion = widgets.Accordion(children=[stats_table])
    stats_table_accordion.set_title(
        0, 'Descriptive statistics for numeric data')

    # Statistics for nominal data
    cols_with_nom_data = _get_nominal_cols(data, cols)
    stats_table2 = widgets.Output()
    stats_table2.append_display_data(
        describe(
            data,
            col_na=col_with_most_nas,
            columns=setdiff1d(cols_with_nom_data, [col_with_most_nas]))
        .round(round_dec))
    stats_table2_accordion = widgets.Accordion(children=[stats_table2])
    stats_table2_accordion.set_title(
        0, 'Descriptive statistics for nominal data')

    # Finalizing tab
    stats_tab = widgets.VBox(
        [select_accordion, stats_table_accordion, stats_table2_accordion])

    # CORRELATION TAB
    # Columns selection
    def _on_corr_col_select(names):
        corr_image.clear_output(wait=False)
        with corr_image:
            display(
                plot_corr(data, columns=names['new'], **corr_kws)
                .properties(width=400, height=400))

    corr_select_cols = widgets.SelectMultiple(
        options=na_cols, rows=10)
    corr_select_cols.observe(_on_corr_col_select, names='value')
    corr_select_header = widgets.HTML(
        '<b>Select columns to calculate correlations</b>')
    corr_select_box = widgets.VBox(
        [corr_select_header, corr_select_cols],
        layout={'align_items': 'center'})

    # Correlations heatmap
    corr_image = widgets.Output()
    corr_image.append_display_data(
        plot_corr(data, columns=random.choice(na_cols, 5), **corr_kws)
        .properties(width=400, height=400))
    corr_image_header = widgets.HTML('<b>NA values correlations</b>')
    corr_image_box = widgets.VBox(
        [corr_image_header, corr_image], layout={'align_items': 'center'})

    # Finalizing correlations tab
    # corr_tab = widgets.GridBox(
    #     [corr_image_box, corr_select_box], layout=layout)
    corr_tab = widgets.HBox(
        [corr_image_box, corr_select_box])

    # DISTRIBUTIONS TAB
    def _on_dist_col_select(_):
        col = dist_col_select.value
        na_col = na_col_select.value
        dist_kind = dist_kind_select.value
        plot_func = plot_hist if dist_kind == 'hist' else plot_kde
        dist_image.clear_output(wait=False)
        with dist_image:
            if col == na_col:
                display(widgets.HTML("Select different columns"))
            else:
                display(
                    plot_func(
                        data,
                        col=col,
                        col_na=na_col,
                        **dist_kws).properties(width=400))

    # Choosing a random column (except the one with most NAs)
    random_col = random.choice(
        setdiff1d(cols, [col_with_most_nas]))

    # Preparing widgets
    dist_image = widgets.Output()
    dist_image.append_display_data(
        plot_hist(data, col=random_col, col_na=col_with_most_nas, **dist_kws)
        .properties(width=400))
    dist_image_header = widgets.HTML('<b>Distributions of values</b>')
    dist_image_box = widgets.VBox(
        [dist_image_header, dist_image], layout={'align_items': 'center'})

    dist_kind_header = widgets.HTML('<b>Plot kind</b>')
    dist_kind_select = widgets.Dropdown(
        options=[('Histogram', 'hist'), ('Density', 'kde')])
    dist_kind_select.observe(_on_dist_col_select, names='value')

    na_col_header = widgets.HTML('<b>Column with NA values</b>')
    na_col_select = widgets.Dropdown(options=cols)
    na_col_select.value = col_with_most_nas
    na_col_select.observe(_on_dist_col_select, names='value')

    dist_col_header = widgets.HTML(
        '<b>Column to explore distributions of values</b>')
    dist_col_select = widgets.Dropdown(options=cols)
    dist_col_select.value = random_col
    dist_col_select.observe(_on_dist_col_select, names='value')

    selects_box = widgets.VBox([
        na_col_header, na_col_select,
        dist_col_header, dist_col_select,
        dist_kind_header, dist_kind_select],
        layout={'align_items': 'center'})
    dist_tab = widgets.HBox(
        [dist_image_box, selects_box])

    # FINALIZING REPORT INTERFACE
    tab.children = [summary_tab, vis_tab, stats_tab, corr_tab, dist_tab]
    tab.set_title(0, 'Summary')
    tab.set_title(1, 'Visualizations')
    tab.set_title(2, 'Statistics')
    tab.set_title(3, 'Correlations')
    tab.set_title(4, 'Distributions')

    return tab
