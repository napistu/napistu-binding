"""Constants for the napistu-binding package."""

from types import SimpleNamespace

OPTIONAL_DEPENDENCIES = SimpleNamespace(
    RDKIT="rdkit",
)

OPTIONAL_DEFS = SimpleNamespace(
    RDKIT_PACKAGE="rdkit",
    RDKIT_EXTRA=OPTIONAL_DEPENDENCIES.RDKIT,
)
