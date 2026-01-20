"""
nMol class for enhanced molecule handling with RDKit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napistu_binding.utils.logging import restore_rdkit_logging, suppress_rdkit_logging
from napistu_binding.utils.optional import require_rdkit

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


@require_rdkit
def _get_mol_class():
    """Get the RDKit Mol class, ensuring RDKit is available with helpful error messages."""
    from rdkit import Chem

    return Chem.Mol


# Get the Mol class for subclassing - this will raise a helpful error if RDKit is not installed
_Mol = _get_mol_class()


class nMol(_Mol):
    """
    Enhanced molecule class that subclasses RDKit's Mol with additional functionality.

    This class extends RDKit's Mol object and adds convenience methods for
    validation and SMILES conversion. It can be used as a drop-in replacement
    for Mol objects in most contexts.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from napistu_binding.load.nmol import nMol
    >>>
    >>> # Create from SMILES string
    >>> mol = Chem.MolFromSmiles("CC(=O)O")
    >>> nmol = nMol(mol)
    >>>
    >>> # Use like a Mol object
    >>> num_atoms = nmol.GetNumAtoms()
    >>>
    >>> # Use enhanced methods
    >>> is_valid = nmol.is_valid()
    >>> smiles = nmol.to_isomeric_smiles()
    """

    def __init__(self, mol: Mol):
        """
        Initialize nMol from an RDKit Mol object.

        Similar to NapistuGraph, this class subclasses RDKit's Mol. However,
        RDKit's Mol is a C extension type, so direct initialization requires
        copying the molecule data.

        Parameters
        ----------
        mol : Mol
            RDKit molecule object to base this NapistuMol on.

        Raises
        ------
        ImportError
            If RDKit is not installed.
        ValueError
            If mol is not an RDKit Mol object.
        """
        from rdkit import Chem

        if not isinstance(mol, Chem.Mol):
            raise ValueError(f"mol must be an RDKit Mol object, got {type(mol)}")

        # RDKit Mol is a C extension, so we need to copy it properly
        # Try using RDKit's serialization for a deep copy
        try:
            # Use binary serialization for a true copy
            mol_binary = mol.ToBinary()
            mol_copy = Chem.Mol(mol_binary)
            # Initialize parent with the copy
            super().__init__(mol_copy)
        except (AttributeError, TypeError):
            # Fallback: if ToBinary doesn't work, try direct copy
            # This may or may not work depending on RDKit version
            super().__init__(mol)

    @classmethod
    def FromSmiles(cls, smiles: str, sanitize: bool = True):
        """
        Create an nMol from a SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string representation of the molecule.
        sanitize : bool, default=True
            Whether to sanitize the molecule upon creation.

        Returns
        -------
        nMol
            An nMol instance created from the SMILES string.
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")
        return cls(mol)

    @classmethod
    def from_mol(cls, mol: Mol):
        """
        Create an nMol from an existing RDKit Mol object.

        Similar to NapistuGraph.from_igraph(), this allows transparent conversion
        from RDKit Mol objects to nMol objects.

        Parameters
        ----------
        mol : Mol
            RDKit molecule object to convert.

        Returns
        -------
        nMol
            An nMol instance created from the Mol object.

        Examples
        --------
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CC(=O)O")
        >>> nmol = nMol.from_mol(mol)
        """
        return cls(mol)

    def __repr__(self) -> str:
        """Return a string representation of nMol."""
        try:
            mol_repr = super().__repr__()
        except Exception:
            mol_repr = str(self)
        return f"nMol({mol_repr})"

    def is_valid(
        self, verbose: bool = False, suppress_rdkit_errors: bool = True
    ) -> bool:
        """
        Check if this molecule is valid and can be properly processed.

        Applies a set of validation checks to ensure the molecule can be properly
        processed. Returns False for molecules that:
        - Are None or invalid
        - Cannot be sanitized
        - Have coordinate bonds that prevent SMILES conversion
        - Cannot be converted to a valid SMILES string

        Parameters
        ----------
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
        >>> from napistu_binding.load.nmol import nMol
        >>>
        >>> mol = Chem.MolFromSmiles("CC(=O)O")
        >>> nmol = nMol(mol)
        >>> nmol.is_valid()
        True
        """
        from rdkit import Chem

        # Suppress RDKit errors during validation if requested
        if suppress_rdkit_errors:
            suppress_rdkit_logging()
        try:
            # Filter out None molecules
            if self is None:
                if verbose:
                    print("Molecule filtered: None molecule")
                return False

            # Try to sanitize the molecule (catches basic validity issues)
            try:
                Chem.SanitizeMol(self)
            except Exception as e:
                if verbose:
                    print(f"Molecule filtered: Failed sanitization - {str(e)}")
                return False

            # Try to convert to SMILES to ensure it's processable
            # This will catch issues like coordinate bonds that can't be represented
            try:
                smiles = Chem.MolToSmiles(self, isomericSmiles=True, canonical=True)
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

    def to_isomeric_smiles(
        self, num_round_trips: int = 2, suppress_rdkit_errors: bool = True
    ) -> str:
        """
        Convert this molecule to a canonical isomeric SMILES string.

        Performs multiple round-trips (SMILES -> Mol -> SMILES) to ensure
        a stable, canonical representation. This is particularly useful for
        molecules that may have been modified or come from sources that don't
        preserve canonical atom ordering.

        Parameters
        ----------
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
        ValueError
            If the molecule cannot be converted to SMILES, or if num_round_trips
            is less than 1.

        Examples
        --------
        >>> from rdkit import Chem
        >>> from napistu_binding.load.nmol import nMol
        >>>
        >>> mol = Chem.MolFromSmiles("CC(=O)O")
        >>> nmol = nMol(mol)
        >>> smiles = nmol.to_isomeric_smiles()
        >>> print(smiles)
        'CC(=O)O'
        """
        from rdkit import Chem

        if num_round_trips < 1:
            raise ValueError(f"num_round_trips must be >= 1, got {num_round_trips}")

        if not isinstance(self, Chem.Mol):
            raise ValueError(f"mol must be an RDKit Mol object, got {type(self)}")

        # Suppress RDKit errors during SMILES conversion if requested
        if suppress_rdkit_errors:
            suppress_rdkit_logging()
        try:
            # Start with the original molecule
            current_mol = self

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
                    error_msg = (
                        f"Failed to convert molecule to canonical SMILES: {str(e)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e

            # Final conversion to SMILES
            return Chem.MolToSmiles(current_mol, isomericSmiles=True, canonical=True)
        finally:
            if suppress_rdkit_errors:
                restore_rdkit_logging()
