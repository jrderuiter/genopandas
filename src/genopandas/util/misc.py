import re

import numpy as np
import pandas as pd


def with_defaults(kws, defaults):
    """Returns merged dict with defaults for missing values."""
    return {**defaults, **(kws or {})}


def lookup(mapping, key, label='key'):
    """Looks-up key in dict, raising a readable exception if not found."""

    try:
        value = mapping[key]
    except KeyError:
        raise ValueError('Unknown {} type: {}. Valid values are {}'
                         .format(label, key, list(mapping.keys())))

    return value


def reorder_columns(df, order):
    """Reorders dataframe columns, sorting any extra columns alphabetically."""
    extra_cols = set(df.columns) - set(order)
    return df[list(order) + sorted(extra_cols)]


def expand_index(index, expression=None, one_based=False):
    """Expands index with position values."""

    # Compile regex.
    if expression is None:
        expression = r'(?P<chromosome>\w+):(?P<position>\d+)'

    regex = re.compile(expression)

    # Extract chromosome, start, end positions.
    group_dicts = (regex.match(el).groupdict() for el in index)

    tups = ((grp['chromosome'], int(grp['position'])) for grp in group_dicts)

    chrom, positions = zip(*tups)

    # Correct for one-base to match Python conventions.
    positions = np.array(positions)

    if one_based:
        positions -= 1

    # Build index.
    index = pd.MultiIndex.from_arrays(
        [chrom, positions], names=['chromosome', 'position'])

    return index


def expand_index_ranged(index,
                        expression=None,
                        one_based=False,
                        inclusive=False):
    """Expands index with ranged values."""

    # Compile regex.
    if expression is None:
        expression = r'(?P<chromosome>\w+):(?P<start>\d+)-(?P<end>\d+)'

    regex = re.compile(expression)

    # Extract chromosome, start, end positions.
    group_dicts = (regex.match(el).groupdict() for el in index)

    tups = ((grp['chromosome'], int(grp['start']), int(grp['end']))
            for grp in group_dicts)

    chrom, starts, ends = zip(*tups)

    # Correct for one-base and/or inclusive-ness to
    # match Python conventions.
    starts = np.array(starts)
    ends = np.array(ends)

    if one_based:
        starts -= 1

    if inclusive:
        ends += 1

    # Build index.
    index = pd.MultiIndex.from_arrays(
        [chrom, starts, ends], names=['chromosome', 'start', 'end'])

    return index
