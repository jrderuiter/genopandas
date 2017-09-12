import re

import pandas as pd


REGION_REGEX = re.compile(r'(?P<chromosome>\w+):(?P<start>\d+)-(?P<end>\d+)')


def expand_region_index(df,
                        regex=REGION_REGEX,
                        names=('chromosome', 'start', 'end')):

    regions = [parse_region_str(i, regex=regex) for i in df.index]

    expanded = df.copy()
    expanded.index = pd.MultiIndex.from_tuples(regions, names=names)

    return expanded


def parse_region_str(region_str, regex):
    match = regex.search(region_str)

    if match is None:
        raise ValueError('Unable to parse region {!r}'.format(region_str))

    groups = match.groupdict()

    return (groups['chromosome'], int(groups['start']), int(groups['end']))
