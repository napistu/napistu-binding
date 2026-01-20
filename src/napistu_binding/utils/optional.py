"""
Utilities for handling optional dependencies in Napistu-Binding.

Decorators
----------
require_rdkit
    Decorator ensuring rdkit is available before calling *func*.

Public Functions
----------------
import_rdkit:
    Import and return rdkit, raising an informative error if missing.
"""

from __future__ import annotations

import importlib
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from napistu_binding.constants import OPTIONAL_DEFS

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def import_rdkit():
    """Import and return rdkit, raising an informative error if missing."""

    try:
        return importlib.import_module(OPTIONAL_DEFS.RDKIT_PACKAGE)
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - executed when dependency missing
        raise ImportError(
            "This functionality requires `rdkit`. "
            f"Install with `pip install 'napistu-binding[{OPTIONAL_DEFS.RDKIT_EXTRA}]'`."
        ) from exc


def require_rdkit(func: _F) -> _F:
    """Decorator ensuring rdkit is available before calling *func*.

    Use this decorator for functions that require rdkit.

    Examples
    --------
    >>> @require_rdkit
    >>> def extract_molecules_from_sdf(sdf_path):
    ...     # Uses rdkit
    ...     pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_rdkit()
        return func(*args, **kwargs)

    return cast(_F, wrapper)
