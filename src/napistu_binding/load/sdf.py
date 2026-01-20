"""
Functions for loading .sdf files containing metabolite structures and metadata.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import List, Union

from napistu.utils import download_wget
from napistu_torch.utils.base_utils import ensure_path

from napistu_binding.load.constants import CHEBI_SDF_DEFS
from napistu_binding.load.nmol import nMol
from napistu_binding.utils.logging import restore_rdkit_logging, suppress_rdkit_logging
from napistu_binding.utils.optional import require_rdkit

logger = logging.getLogger(__name__)


def download_chebi_sdf(
    target_uri: str, chebi_variant: str = CHEBI_SDF_DEFS.VARIANTS.CHEBI_3_STARS
) -> None:
    """
    Download the CHEBI SDF file for a given variant.

    Parameters
    ----------
    target_uri : str
        The URI where the CHEBI SDF file should be saved.
    chebi_variant : str
        The variant of CHEBI to download.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the CHEBI variant is not supported.
    """

    allowed_variants = CHEBI_SDF_DEFS.VARIANTS.__dict__.values()
    if chebi_variant not in allowed_variants:
        raise ValueError(
            f"Invalid CHEBI version: {chebi_variant}. Supported versions are: {allowed_variants}"
        )

    chebi_sdf_url = (
        f"{CHEBI_SDF_DEFS.URL_ROOT}{chebi_variant}.{CHEBI_SDF_DEFS.FILE_EXTENSION}"
    )

    logger.info(f"Downloading CHEBI SDF file from {chebi_sdf_url} to {target_uri}")
    download_wget(chebi_sdf_url, target_uri)

    return None


@require_rdkit
def extract_molecules_from_sdf(
    sdf_path: Union[str, Path],
    remove_invalid: bool = True,
    verbose: bool = False,
    suppress_rdkit_errors: bool = True,
) -> List[nMol]:
    """
    Extract all molecule objects from an SDF file using RDKit.

    Parameters
    ----------
    sdf_path : str
        Path to the SDF file. Supports both compressed (.sdf.gz) and
        uncompressed (.sdf) files.
    remove_invalid : bool, default=True
        If True, filter out molecules that fail validation checks (e.g., cannot
        be converted to SMILES, have coordinate bonds, etc.). If False, return
        all molecules including those that may be problematic.
    verbose : bool, default=False
        If True, log detailed information about the extraction process.
    suppress_rdkit_errors : bool, default=True
        If True, suppress RDKit warnings and errors during extraction (e.g.,
        sanitization failures, valence errors, etc.).

    Returns
    -------
    List[nMol]
        List of nMol objects extracted from the SDF file.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    FileNotFoundError
        If the SDF file does not exist.
    ValueError
        If the file cannot be parsed as an SDF file.
    """
    from rdkit import Chem

    sdf_path_obj = ensure_path(sdf_path, expand_user=True)
    if not sdf_path_obj.is_file():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    if verbose:
        print(f"Extracting molecules from SDF file: {sdf_path}")

    # Determine if file is compressed
    is_compressed = sdf_path.endswith(".gz") or sdf_path.endswith(".sdf.gz")

    # Suppress RDKit errors only during SDF file reading (supplier iteration)
    if suppress_rdkit_errors:
        suppress_rdkit_logging()

    try:
        if is_compressed:
            with gzip.open(sdf_path, "rb") as f:
                supplier = Chem.ForwardSDMolSupplier(f)
                molecules = [mol for mol in supplier if mol is not None]
        else:
            supplier = Chem.SDMolSupplier(str(sdf_path))
            molecules = [mol for mol in supplier if mol is not None]
    except Exception as e:
        error_msg = f"Failed to parse SDF file {sdf_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    finally:
        # Restore RDKit logging immediately after reading SDF
        if suppress_rdkit_errors:
            restore_rdkit_logging()

    initial_count = len(molecules)

    # Convert to nMol objects
    napistu_molecules = [nMol(mol) for mol in molecules]

    # Filter out invalid molecules if requested
    if remove_invalid:
        # Suppress RDKit errors during validation if requested
        # Pass suppress_rdkit_errors=False to is_valid() to avoid nested suppression
        if suppress_rdkit_errors:
            suppress_rdkit_logging()
        try:
            napistu_molecules = [
                mol
                for mol in napistu_molecules
                if mol.is_valid(verbose=verbose, suppress_rdkit_errors=False)
            ]
            removed_count = initial_count - len(napistu_molecules)
            if removed_count > 0 and verbose:
                print(
                    f"Filtered out {removed_count} invalid molecule(s) from {sdf_path}"
                )
        finally:
            if suppress_rdkit_errors:
                restore_rdkit_logging()

    if verbose:
        print(f"Extracted {len(napistu_molecules)} molecule(s) from {sdf_path}")
    return napistu_molecules
