"""
Data source adapters for MASS
"""
from .base import DataAdapter
from .ydb_adapter import YDBAdapter

__all__ = ['DataAdapter', 'YDBAdapter']

