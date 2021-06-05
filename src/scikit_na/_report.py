__all__ = ['report']
from .altair import plot_corr, plot_dist
from ._stats import describe
from pandas import DataFrame, Index
from ipywidgets import widgets
from typing import Optional, Union, List
from numpy import array, ndarray, random


def report(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        layout: widgets.Layout = None,
        round_sgn: int = 2,
        corr_kws: dict = {},
        heat_kws: dict = {},
        dist_kws: dict = {}):

    from IPython.display import display
    cols = array(columns) if columns is not None else array(data.columns)
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

    # STATISTICS TAB
    # Table with per column stats
    stats_table = widgets.Output()
    stats_table.append_display_data(
        describe(data, columns=cols, per_column=True).round(round_sgn))
    stats_table_accordion = widgets.Accordion(children=[stats_table])
    stats_table_accordion.set_title(0, 'NA statistics (per column)')
    stats_table_accordion.selected_index = 0

    # Columns selection
    def _on_col_select(names):
        stats_table.clear_output(wait=False)
        total_stats_table.clear_output(wait=False)
        with stats_table:
            display(
                describe(data, columns=array(names['new']), per_column=True)
                .round(round_sgn))
        with total_stats_table:
            display(
                describe(data, columns=array(names['new']), per_column=False)
                .round(round_sgn))
    select_cols = widgets.SelectMultiple(options=cols, rows=6)
    select_cols.observe(_on_col_select, names='value')
    select_accordion = widgets.Accordion(children=[select_cols])
    select_accordion.set_title(0, 'Select columns to describe')
    select_accordion.selected_index = 0

    # Table with total stats
    total_stats_table = widgets.Output()
    total_stats_table.append_display_data(
        describe(data, columns=cols, per_column=False).round(round_sgn))
    total_stats_accordion = widgets.Accordion(children=[total_stats_table])
    total_stats_accordion.set_title(
        0, 'NA statistics (in total, for selected columns)')

    # Finalizing stats tab
    stats_tab = widgets.VBox(
        [select_accordion, stats_table_accordion, total_stats_accordion])

    # CORRELATION TAB
    # Columns selection
    def _on_corr_col_select(names):
        corr_image.clear_output(wait=False)
        with corr_image:
            display(plot_corr(data, columns=names['new'], **corr_kws))
        pass

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
        plot_corr(data, columns=random.choice(na_cols, 5), **corr_kws))
    corr_image_header = widgets.HTML('<b>NA values correlations</b>')
    corr_image_box = widgets.VBox(
        [corr_image_header, corr_image], layout={'align_items': 'center'})

    # Finalizing correlations tab
    corr_tab = widgets.GridBox(
        [corr_image_box, corr_select_box], layout=layout)

    # DISTRIBUTIONS TAB
    def _on_dist_col_select(names):
        col = dist_col_select.value
        na_col = na_col_select.value
        dist_kind = dist_kind_select.value
        dist_image.clear_output(wait=False)
        with dist_image:
            if col == na_col:
                display("Select different columns")
            else:
                display(
                    plot_dist(
                        data,
                        col=col,
                        col_na=na_col,
                        kind=dist_kind,
                        **dist_kws))

    random_cols = random.choice(cols, 2)
    dist_image = widgets.Output()
    dist_image.append_display_data(
        plot_dist(data, col=random_cols[0], col_na=random_cols[1], **dist_kws))
    dist_image_header = widgets.HTML('<b>Distributions of values</b>')
    dist_image_box = widgets.VBox(
        [dist_image_header, dist_image], layout={'align_items': 'center'})

    dist_kind_select = widgets.Dropdown(options=['hist', 'kde'])
    dist_kind_select.observe(_on_dist_col_select, names='value')

    na_col_header = widgets.HTML('<b>Column with NA values</b>')
    na_col_select = widgets.Dropdown(options=cols)
    na_col_select.observe(_on_dist_col_select, names='value')

    dist_col_header = widgets.HTML(
        '<b>Column to explore distributions of values</b>')
    dist_col_select = widgets.Dropdown(options=cols)
    dist_col_select.observe(_on_dist_col_select, names='value')

    selects_box = widgets.VBox(
        [na_col_header, na_col_select, dist_col_header, dist_col_select],
        layout={'align_items': 'center'})
    dist_tab = widgets.HBox(
        [dist_image_box, selects_box])

    # FINALIZING REPORT INTERFACE
    tab.children = [stats_tab, corr_tab, dist_tab]
    tab.set_title(0, 'Statistics')
    tab.set_title(1, 'Correlations')
    tab.set_title(2, 'Distributions')
    return tab
