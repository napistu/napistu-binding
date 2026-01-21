"""Constants for the load subpackage."""

from types import SimpleNamespace

# sdf constants

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

# diffdock constants
DIFFDOCK_CONSTANTS = SimpleNamespace(
    SPACE_ID="reginabarzilaygroup/DiffDock-Web",
    API_NAME="/predict",
    BACKENDS=SimpleNamespace(
        HUGGINFACE="huggingface",
        DOCKER="docker",
    ),
)
