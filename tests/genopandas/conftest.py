import os
from pathlib import Path

import numpy as np
import random as rand

pytest_plugins = ['helpers_namespace']

import pytest

BASE_DIR = Path(__file__).parent


@pytest.helpers.register
def data_path(relative_path, relative_to=None):
    """Returns data path to test file."""

    if relative_to is None:
        # Use BASE_DIR as default.
        relative_to = BASE_DIR
    elif not isinstance(relative_to, Path):
        # Ensure relative_to is a Path.
        relative_to = Path(relative_to)

    # If relative_to is not a path, move up one level.
    if not relative_to.is_dir():
        relative_to = relative_to.parent

    return relative_to / 'data' / relative_path


@pytest.fixture
def random():
    rand.seed(0)
    np.random.seed(0)
