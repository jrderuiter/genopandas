import functools

import pandas as pd


class DfWrapper(object):
    """Utility class used to build wrapper classes for pandas DataFrames."""

    def __init__(self, values):
        self._values = values

    @property
    def values(self):
        """Internal DataFrame containing values."""
        return self._values

    @property
    def loc(self):
        """Label-based indexer (similar to pandas .loc)."""
        return DfLocWrapper(self._values.loc, constructor=self._constructor)

    @property
    def iloc(self):
        """Index-based indexer (similar to pandas .iloc)."""
        return DfLocWrapper(self._values.iloc, constructor=self._constructor)

    def _constructor(self, values):
        """Constructor that attempts to build new instance
           from given values."""

        if isinstance(values, pd.DataFrame):
            return self.__class__(values.copy())

        return values

    def __getitem__(self, item):
        values = self._values[item]
        return self._constructor(values)

    def __getattr__(self, name):
        if hasattr(self._values, name):
            attr = getattr(self._values, name)
        else:
            raise AttributeError('{!r} object has no attribute {!r}'.format(
                self.__class__.__name__, name))

        if callable(attr):
            attr = self._wrap_function(attr)
        else:
            attr = self._constructor(attr)

        return attr

    def _wrap_function(self, func):
        """Wrap functions to call _constructor on returned value."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return self._constructor(result)

        return wrapper


class DfLocWrapper(object):
    """Wrapper class that wraps an objects loc/iloc accessor."""

    def __init__(self, loc, constructor=None):
        if constructor is None:
            constructor = lambda x: x

        self._loc = loc
        self._constructor = constructor

    def __getitem__(self, item):
        result = self._loc[item]
        return self._constructor(result)
