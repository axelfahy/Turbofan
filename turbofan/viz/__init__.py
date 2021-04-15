"""Visualization module of turbofan.

This module contains functions relative to data visualization.
"""
from .plot import (
    plot_distribution,
    plot_performance_by_threshold,
)

# Public object of the module.
__all__ = [
    'plot_distribution',
    'plot_performance_by_threshold',
]
