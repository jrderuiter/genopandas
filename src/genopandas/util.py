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
