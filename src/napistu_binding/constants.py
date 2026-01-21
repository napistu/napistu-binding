"""Constants for the napistu-binding package."""

from types import SimpleNamespace

OPTIONAL_DEPENDENCIES = SimpleNamespace(
    METABOLITE="metabolite",
    PROTEIN="protein",
)

OPTIONAL_DEFS = SimpleNamespace(
    RDKIT_PACKAGE="rdkit",
    RDKIT_EXTRA=OPTIONAL_DEPENDENCIES.METABOLITE,
    BIOPYTHON_PACKAGE="biopython",
    BIOPYTHON_EXTRA=OPTIONAL_DEPENDENCIES.PROTEIN,
    ESM_PACKAGE="esm",
    ESM_EXTRA=OPTIONAL_DEPENDENCIES.PROTEIN,
)
