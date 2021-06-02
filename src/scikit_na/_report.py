__all__ = ['report', 'view_dist']
from .mpl import plot_corr
from .altair import plot_dist
from ._descr import describe
from pandas import DataFrame, Index
from ipywidgets import widgets, interact
from typing import Optional, Union, List
from numpy import array, ndarray
from io import BytesIO
from matplotlib.pyplot import close


def view_dist(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        **kwargs):
    """Interactively observe distribution of values in a selected column
    grouped by NA/non-NA values in another column.

    Parameters
    ----------
    data : DataFrame
        Input data.
    columns : Union[list, ndarray, Index] = None
        Column names.

    Returns
    -------
    _InteractFactory
        Interactive widget.
    """
    cols = array(columns) if columns is not None else data.columns
    na_cols = data.isna().sum(axis=0)\
        .rename('na_num')\
        .to_frame()\
        .query('na_num > 0')\
        .index.values

    return interact(
        lambda Column, NA:
            plot_dist(data, col=Column, col_na=NA, **kwargs)
            if Column != NA
            else widgets.HTML(
                '<em style="color: red">Note: select different columns</em>'),
        Column=cols, NA=na_cols)


def report(
        data: DataFrame,
        columns: Optional[Union[List, ndarray, Index]] = None,
        layout: widgets.Layout = None,
        corr_kws: dict = {},
        heat_kws: dict = {}):

    from IPython.display import display
    cols = array(columns) if columns is not None else array(data.columns)

    layout = widgets.Layout(
        grid_template_columns='1fr 1fr',
        justify_items='center') if not layout else layout
    tab = widgets.Tab()

    # STATISTICS TAB
    # Table with per column stats
    stats_table = widgets.Output()
    stats_table.append_display_data(
        describe(data, columns=cols, per_column=True))
    stats_table_accordion = widgets.Accordion(children=[stats_table])
    stats_table_accordion.set_title(0, 'NA statistics (per column)')
    stats_table_accordion.selected_index = 0

    # Columns selection
    def _on_col_select(names):
        stats_table.clear_output(wait=False)
        total_stats_table.clear_output(wait=False)
        with stats_table:
            display(
                describe(data, columns=array(names['new']), per_column=True))
        with total_stats_table:
            display(
                describe(data, columns=array(names['new']), per_column=False))
    select_cols = widgets.SelectMultiple(options=cols, rows=4)
    select_cols.observe(_on_col_select, names='value')
    select_accordion = widgets.Accordion(children=[select_cols])
    select_accordion.set_title(0, 'Select columns to describe')
    select_accordion.selected_index = 0

    # Table with total stats
    total_stats_table = widgets.Output()
    total_stats_table.append_display_data(
        describe(data, columns=cols, per_column=False))
    total_stats_accordion = widgets.Accordion(children=[total_stats_table])
    total_stats_accordion.set_title(0, 'NA statistics (in total)')

    # Finalizing stats tab
    stats_tab = widgets.VBox(
        [select_accordion, stats_table_accordion, total_stats_accordion])

    # CORRELATION TAB
    # Columns selection
    def _on_corr_col_select(names):
        # stats_table.clear_output(wait=False)
        # total_stats_table.clear_output(wait=False)
        # with stats_table:
        #     display(describe(data, columns=names['new'], per_column=True))
        pass

    corr_select_cols = widgets.SelectMultiple(options=cols, rows=4)
    corr_select_cols.observe(_on_corr_col_select, names='value')

    # Correlations heatmap
    corr_image_svg = BytesIO()
    ax_corr = plot_corr(
        data, columns=cols,
        corr_kws=corr_kws, heat_kws=heat_kws)
    ax_corr.figure.tight_layout()
    ax_corr.figure.savefig(corr_image_svg, format='png', dpi=300)
    close(ax_corr.figure)
    corr_image_svg.seek(0)
    corr_image = widgets.Image(value=corr_image_svg.read())
    corr_image_header = widgets.HTML('<b>NA values correlations</b>')
    corr_image_box = widgets.VBox(
        [corr_image_header, corr_image], layout={'align_items': 'center'})

    # Finalizing correlations tab
    corr_tab = widgets.GridBox(
        [corr_image_box], layout=layout)

    # DISTRIBUTIONS TAB
    # TODO: interactivity
    dist_image_header = widgets.HTML('<b>NA values correlations</b>')
    dist_image_box = widgets.GridBox(
        [dist_image_header, corr_image], grid_row='span 4')

    na_col_header = widgets.HTML('<b>Column with NA values</b>')
    na_col_select = widgets.Select(options=cols)

    dist_col_header = widgets.HTML(
        '<b>Column to explore distributions of values:</b>')
    dist_col_select = widgets.Select(options=cols)

    selects_box = widgets.GridBox(
        [na_col_header, na_col_select, dist_col_header, dist_col_select])
    dist_tab = widgets.GridBox(
        [dist_image_box, selects_box], layout=layout)

    # FINALIZING REPORT INTERFACE
    tab.children = [stats_tab, corr_tab, dist_tab]
    tab.set_title(0, 'Statistics')
    tab.set_title(1, 'Correlations')
    tab.set_title(2, 'Distributions')
    return tab
