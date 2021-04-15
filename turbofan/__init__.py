"""All of turbofan' functions."""
from datetime import datetime
import logging
from pathlib import Path

from ._version import get_versions

# Import submodules.
from . import data
from . import viz

__all__ = ['data',
           'viz']

# Logging configuration.
# Logs to stdout and to file.
formatter = logging.Formatter('%(asctime)s [%(levelname)-7s] %(name)s: %(message)s')

# File logger.
logfile = Path('/tmp').joinpath(f"coffeevision_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Stream logger.
stream_handler = logging.StreamHandler(None)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

# Logging configuration.
FORMAT = '%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

__version__ = get_versions()['version']
del get_versions
