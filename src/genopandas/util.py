import re

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


def expand_index(index, expression):
    regex = re.compile(expression)

    if '?P<start>' in expression:
        group_dicts = (regex.match(el).groupdict() for el in index)
        tups = ((grp['chromosome'], int(grp['start']), int(grp['end']))
                for grp in group_dicts)
        names = ['chromosome', 'start', 'end']
    elif '?P<position>' in expression:
        group_dicts = (regex.match(el).groupdict() for el in index)
        tups = ((grp['chromosome'], int(grp['position']))
                for grp in group_dicts)
        names = ['chromosome', 'position']
    else:
        raise ValueError('Invalid expression')

    try:
        index = pd.MultiIndex.from_tuples(list(tups), names=names)
    except AttributeError:
        raise ValueError('Some entries did not match the given expression')

    return index
