"""
Napistu Binding - PyTorch-based toolkit for working with protein and metabolite structures
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napistu_binding")
except PackageNotFoundError:
    # package is not installed
    pass
