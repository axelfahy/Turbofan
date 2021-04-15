"""Data module of Turbofan.

Handle dataset creation and exploration.
"""
from .dataset import create_test_set, create_train_set

# Public object of the module.
__all__ = [
    'create_test_set',
    'create_train_set',
]
