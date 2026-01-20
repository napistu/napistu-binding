"""
Functions for loading .sdf files containing metabolite structures and metadata.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from napistu.utils import download_wget
from napistu_torch.utils.base_utils import ensure_path

from napistu_binding.load.constants import CHEBI_SDF_DEFS
from napistu_binding.utils.logging import restore_rdkit_logging, suppress_rdkit_logging
from napistu_binding.utils.optional import require_rdkit

if TYPE_CHECKING:
    from rdkit.Chem import Mol

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
) -> List[Mol]:
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
    List[Mol]
        List of RDKit Mol objects extracted from the SDF file.

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

    # Filter out invalid molecules if requested
    if remove_invalid:
        # Suppress RDKit errors during validation if requested
        # Pass suppress_rdkit_errors=False to is_valid_molecule to avoid nested suppression
        if suppress_rdkit_errors:
            suppress_rdkit_logging()
        try:
            molecules = [
                mol
                for mol in molecules
                if is_valid_molecule(mol, verbose=verbose, suppress_rdkit_errors=False)
            ]
            removed_count = initial_count - len(molecules)
            if removed_count > 0 and verbose:
                print(
                    f"Filtered out {removed_count} invalid molecule(s) from {sdf_path}"
                )
        finally:
            if suppress_rdkit_errors:
                restore_rdkit_logging()

    if verbose:
        print(f"Extracted {len(molecules)} molecule(s) from {sdf_path}")
    return molecules


@require_rdkit
def is_valid_molecule(
    mol: Mol, verbose: bool = False, suppress_rdkit_errors: bool = True
) -> bool:
    """
    Filter function to determine if a Mol object should be retained.

    Applies a set of validation checks to ensure the molecule can be properly
    processed. Filters out molecules that:
    - Are None or invalid
    - Cannot be sanitized
    - Have coordinate bonds that prevent SMILES conversion
    - Cannot be converted to a valid SMILES string

    Parameters
    ----------
    mol : Mol
        RDKit molecule object to validate.
    verbose : bool, default=False
        If True, log detailed information about why molecules are filtered out.
    suppress_rdkit_errors : bool, default=True
        If True, suppress RDKit warnings and errors during validation.

    Returns
    -------
    bool
        True if the molecule should be retained, False if it should be filtered out.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CC(=O)O")
    >>> is_valid_molecule(mol)
    True
    >>> is_valid_molecule(mol, verbose=True)
    True
    """
    from rdkit import Chem

    # Suppress RDKit errors during validation if requested
    if suppress_rdkit_errors:
        suppress_rdkit_logging()
    try:
        # Filter out None molecules
        if mol is None:
            if verbose:
                print("Molecule filtered: None molecule")
            return False

        # Try to sanitize the molecule (catches basic validity issues)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            if verbose:
                print(f"Molecule filtered: Failed sanitization - {str(e)}")
            return False

        # Try to convert to SMILES to ensure it's processable
        # This will catch issues like coordinate bonds that can't be represented
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            if not smiles or len(smiles) == 0:
                if verbose:
                    print("Molecule filtered: Empty SMILES string")
                return False

            # Try to parse it back to ensure round-trip works
            parsed_mol = Chem.MolFromSmiles(smiles)
            if parsed_mol is None:
                if verbose:
                    print(
                        f"Molecule filtered: Failed to parse SMILES round-trip - {smiles}"
                    )
                return False

        except Exception as e:
            if verbose:
                print(f"Molecule filtered: Failed SMILES conversion - {str(e)}")
            return False

        return True
    finally:
        if suppress_rdkit_errors:
            restore_rdkit_logging()


@require_rdkit
def mol_to_isomeric_smiles(
    mol: Mol, num_round_trips: int = 2, suppress_rdkit_errors: bool = True
) -> str:
    """
    Convert a Mol object to a canonical isomeric SMILES string.

    Performs multiple round-trips (SMILES -> Mol -> SMILES) to ensure
    a stable, canonical representation. This is particularly useful for
    molecules that may have been modified or come from sources that don't
    preserve canonical atom ordering.

    Parameters
    ----------
    mol : Mol
        RDKit molecule object to convert.
    num_round_trips : int, default=2
        Number of round-trips to perform for canonicalization. More round-trips
        ensure greater stability but are slower. Typical values are 1-3.
    suppress_rdkit_errors : bool, default=True
        If True, suppress RDKit warnings and errors during SMILES conversion.

    Returns
    -------
    str
        Canonical isomeric SMILES string.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    ValueError
        If the molecule cannot be converted to SMILES, or if num_round_trips
        is less than 1.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CC(=O)O")
    >>> smiles = mol_to_isomeric_smiles(mol)
    >>> print(smiles)
    'CC(=O)O'
    """
    from rdkit import Chem

    if num_round_trips < 1:
        raise ValueError(f"num_round_trips must be >= 1, got {num_round_trips}")

    if not isinstance(mol, Chem.Mol):
        raise ValueError(f"mol must be an RDKit Mol object, got {type(mol)}")

    # Suppress RDKit errors during SMILES conversion if requested
    if suppress_rdkit_errors:
        suppress_rdkit_logging()
    try:
        # Start with the original molecule
        current_mol = mol

        # Perform round-trips for canonicalization
        for _ in range(num_round_trips):
            try:
                # Convert to SMILES and back to Mol
                smiles = Chem.MolToSmiles(
                    current_mol, isomericSmiles=True, canonical=True
                )
                current_mol = Chem.MolFromSmiles(smiles)
                if current_mol is None:
                    raise ValueError(
                        f"Failed to parse SMILES after round-trip: {smiles}"
                    )
            except Exception as e:
                error_msg = f"Failed to convert molecule to canonical SMILES: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        # Final conversion to SMILES
        return Chem.MolToSmiles(current_mol, isomericSmiles=True, canonical=True)
    finally:
        if suppress_rdkit_errors:
            restore_rdkit_logging()
