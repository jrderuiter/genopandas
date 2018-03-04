"""Module containing additional functions for drawing clustermaps."""

from collections import OrderedDict
from itertools import cycle, islice

import pandas as pd

from ._constants import NUMERICAL_CMAPS, CATEGORICAL_COLORS


def color_annotation(df, colors=None, bg_color='#ffffff'):
    """Converts a data frame of annotations to colors."""

    # TODO: Fix boolean/nan case.

    if colors is None:
        colors = generate_colors(df)

    colored_cols = OrderedDict()
    color_maps = OrderedDict()

    for (col_name, values), col_colors in zip(df.items(), colors):
        colored, color_map = _color_column(values, col_colors, bg_color)
        colored_cols[col_name] = colored
        color_maps[col_name] = color_map

    return pd.DataFrame(colored_cols), color_maps


def _color_column(series, colors, bg_color):
    """Converts an annotation column to colors."""

    if series.dtype == bool:
        # Boolean case.
        return _color_bool(series, colors, bg_color)

#     elif series.dtype.kind in 'biufc':
#         # Numeric case.
#         return _color_numeric(series, colors, bg_color)
    elif series.dtype.kind == 'O':
        # Strings and categorical case.
        return _color_categorical(series, colors, bg_color)
    else:
        raise ValueError('Unsupported dtype: {}'.format(series.dtype))


def _color_bool(series, color, bg_color):
    """Converts a boolean annotation column to colors."""

    color_map = OrderedDict(zip([True, False], [color, bg_color]))
    mapped = series.map(color_map).fillna(bg_color)

    return mapped, color_map


def _color_categorical(series, colors, bg_color):
    """Converts a categorical annotation column to colors."""

    if series.dtype.name == 'category':
        str_values = series.cat.categories
    else:
        str_values = set(series.dropna())

    if isinstance(colors, dict):
        color_map = {k: v for k, v in colors.items() if k in str_values}
    else:
        color_map = OrderedDict(zip(str_values, colors))

    colored = series.map(color_map).fillna(bg_color)

    return colored, color_map


# def _color_numeric(series, color, bg_color, n=200):
#     """Converts a numeric annotation column to colors."""

#     cmap = LinearSegmentedColormap.from_list(
#         name='custom_cmap', colors=[bg_color, color], N=n)
#     norm = Normalize(vmin=series.min(), vmax=series.max())
#     mappable = ScalarMappable(norm=norm, cmap=cmap)

#     rgba_colors = mappable.to_rgba(series.values)
#     hex_colors = (rgb2hex(rgba) for rgba in rgba_colors)

#     mapped = pd.Series(hex_colors, index=series.index, name=series.name)

#     return mapped, color


def _rgb_to_hex(rgb, normalized=True):
    if normalized:
        rgb = tuple(map(lambda x: int(x * 255), rgb))
    return '#%02x%02x%02x' % rgb


def generate_colors(annotation,
                    categorical_colors=CATEGORICAL_COLORS,
                    numerical_colors=NUMERICAL_CMAPS):
    """Generate colors for given annotation."""

    # Convert to iterators and cycle.
    categorical_colors = cycle(categorical_colors)
    numerical_colors = cycle(numerical_colors)

    # Assign colors.
    colors = []
    for _, values in annotation.items():
        if values.dtype == bool:
            colors.append(next(categorical_colors))
        elif values.dtype.kind == 'O':
            n_levels = values.nunique()
            colors.append(list(islice(categorical_colors, n_levels)))
        else:
            raise NotImplementedError()

    return colors


def draw_legends(cm, color_maps, margin=1, y=0.6, **kwargs):
    """Draws annotation legends on the given cluster map."""

    fig = cm.fig
    fig.draw(fig.canvas.get_renderer())

    # Draw 'dummy' legends.
    for name, color_map in color_maps.items():
        _draw_legend(fig, color_map, title=name, loc='upper right', **kwargs)

    # Draw to get proper extents.
    fig.draw(fig.canvas.get_renderer())

    # Resize figure to fit legend and labels.
    labels = cm.ax_heatmap.get_yticklabels()

    if cm.ax_col_colors is not None:
        labels += cm.ax_col_colors.get_yticklabels()

    x_inv = _resize_to_fit_legends(
        fig, fig.legends, labels=labels, margin=margin)

    # Get heights and clear legends.
    legend_heights = [l.get_window_extent().height for l in fig.legends]
    fig.legends = []

    # Re-draw final legends.
    inv = fig.transFigure.inverted()
    y_scale = inv.transform([[0, 1], [0, 0]])[0, 1]

    for (name, color_map), height in zip(color_maps.items(), legend_heights):
        _draw_legend(fig, color_map, title=name, loc=(x_inv, y), **kwargs)
        y = y - (height * y_scale)


def _draw_legend(fig, color_map, **kwargs):
    from matplotlib.patches import Patch

    patches = [Patch(color=color, label=label)
               for label, color in color_map.items()]  # yapf: disable

    return fig.legend(patches, color_map.keys(), **kwargs)


def _resize_to_fit_legends(fig, legends, labels, margin=0.5):
    legend_width = max(l.get_window_extent().width for l in legends) / fig.dpi

    if labels:
        label_width = max(l.get_window_extent().width
                          for l in labels) / fig.dpi
    else:
        label_width = 0

    orig_width = fig.get_figwidth()
    required_width = orig_width + label_width + legend_width + margin

    fig.set_figwidth(required_width)
    fig.draw(fig.canvas.get_renderer())

    fig.subplots_adjust(right=orig_width / required_width)
    fig.draw(fig.canvas.get_renderer())

    x_offset = (orig_width + label_width + (margin / 2)) / required_width

    return x_offset
