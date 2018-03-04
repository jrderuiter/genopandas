from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

from .core import *

__author__ = 'Julian de Ruiter'
__email__ = 'julianderuiter@gmail.com'
