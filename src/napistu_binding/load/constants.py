"""Constants for the load subpackage."""

from types import SimpleNamespace

CHEBI_SDF_DEFS = SimpleNamespace(
    URL_ROOT="https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/",
    VARIANTS=SimpleNamespace(
        CHEBI="chebi",
        CHEBI_3_STARS="chebi_3_stars",
        CHEBI_LITE="chebi_lite",
        CHEBI_LITE_3_STARS="chebi_lite_3_stars",
    ),
    FILE_EXTENSION="sdf.gz",
)
