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
        corr_kwargs: dict = {},
        heat_kwargs: dict = {}):

    from IPython.display import display

    layout = widgets.Layout(
        grid_template_columns='1fr 1fr',
        justify_items='center') if not layout else layout
    tab = widgets.Tab()

    # STATISTICS TAB
    # Table with per column stats
    stats_table = widgets.Output()
    stats_table.append_display_data(
        describe(data, columns=columns, per_column=True))
    stats_header = widgets.HTML('<b>NA statistics (per column)</b>')
    stats_table_box = widgets.VBox(
        [stats_header, stats_table], layout={'align_items': 'center'})

    # Columns selection
    def _on_col_select(names):
        stats_table.clear_output(wait=False)
        with stats_table:
            display(data.loc[:, names['new']])
    select = widgets.SelectMultiple(options=columns)
    select.observe(_on_col_select, names='value')
    select_header = widgets.HTML('<b>Select columns to display statistics</b>')
    select_box = widgets.VBox(
        [select_header, select], layout={'align_items': 'center'})

    # Table with total stats
    total_stats_header = widgets.HTML('<b>NA statistics (in total)</b>')
    total_stats_table = widgets.Output()
    total_stats_table.append_display_data(
        describe(data, columns=columns, per_column=True))
    total_stats_box = widgets.VBox(
        [total_stats_header, total_stats_table],
        layout={'align_items': 'center'})

    # Finalizing stats tab
    stats_tab = widgets.GridBox(
        [stats_table_box, select_box, total_stats_box], layout=layout)

    # CORRELATION TAB
    # Correlations heatmap
    corr_image_svg = BytesIO()
    ax_corr = plot_corr(
        data, columns=columns,
        orr_kwargs=corr_kwargs, heat_kwargs=heat_kwargs)
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
    na_col_select = widgets.Select(options=columns)

    dist_col_header = widgets.HTML(
        '<b>Column to explore distributions of values:</b>')
    dist_col_select = widgets.Select(options=columns)

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
