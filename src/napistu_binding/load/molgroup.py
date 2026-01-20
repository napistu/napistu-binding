"""
MolGroup class for organizing groups of related molecules. This is generally used to group conformers of the same molecule by their SMILES or InChI key.

Classes
--------
MolGroup
    A container for grouping related Mol objects by a common identifier.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from rdkit.Chem import Mol


class MolGroup:
    """
    A container for grouping related Mol objects by a common identifier.

    This class groups molecules that share the same identifier (e.g., SMILES string,
    InChI key, or any other grouping variable).

    Attributes
    ----------
    id : Any
        The identifier used to group the molecules (e.g., SMILES string).
    molecules : List[Mol]
        List of Mol objects that share this identifier.

    Examples
    --------
    >>> from napistu_binding.load.sdf import extract_molecules_from_sdf
    >>> from napistu_binding.load.groups import MolGroup
    >>>
    >>> # Load molecules from an SDF file
    >>> mols = extract_molecules_from_sdf("path/to/file.sdf")
    >>>
    >>> # Generate SMILES for grouping
    >>> smiles_list = [mol.to_isomeric_smiles() for mol in mols]
    >>>
    >>> # Create groups
    >>> groups = MolGroup.from_aligned_lists(smiles_list, mols)
    >>>
    >>> # Access a specific group
    >>> group = groups["CCO"]  # ethanol
    >>> print(group.id)  # "CCO"
    >>> print(len(group.molecules))  # number of molecules with this SMILES
    """

    def __init__(self, id: Any, molecules: List[Mol]):
        """
        Initialize a MolGroup.

        Parameters
        ----------
        id : Any
            The identifier used to group the molecules (e.g., SMILES string).
        molecules : List[Mol]
            List of Mol objects that share this identifier.
        """
        self.id = id
        self.molecules = molecules

    def __len__(self) -> int:
        """Return the number of molecules in this group."""
        return len(self.molecules)

    def __repr__(self) -> str:
        """Return a string representation of the MolGroup."""
        return f"MolGroup(id={self.id!r}, n_molecules={len(self.molecules)})"

    @classmethod
    def from_aligned_lists(
        cls, group_ids: List[Any], molecules: List[Mol]
    ) -> Dict[Any, MolGroup]:
        """
        Create MolGroup instances from aligned lists of group IDs and molecules.

        Groups molecules by their corresponding group ID. Molecules with the same
        group ID will be collected into a single MolGroup.

        Parameters
        ----------
        group_ids : List[Any]
            List of identifiers (e.g., SMILES strings) aligned with molecules.
            Must be the same length as molecules.
        molecules : List[Mol]
            List of Mol objects aligned with group_ids.
            Must be the same length as group_ids.

        Returns
        -------
        Dict[Any, MolGroup]
            Dictionary mapping group IDs to MolGroup instances. Each MolGroup
            contains all molecules that share that group ID.

        Examples
        --------
        >>> smiles = ["CCO", "CCO", "CCN", "CCO"]
        >>> mols = [mol1, mol2, mol3, mol4]  # aligned Mol objects
        >>> groups = MolGroup.from_aligned_lists(smiles, mols)
        >>> # groups["CCO"].molecules contains [mol1, mol2, mol4]
        >>> # groups["CCN"].molecules contains [mol3]
        """
        if len(group_ids) != len(molecules):
            raise ValueError(
                f"group_ids and molecules must have the same length. "
                f"Got {len(group_ids)} and {len(molecules)}"
            )

        # Group molecules by their ID
        grouped: Dict[Any, List[Mol]] = defaultdict(list)
        for group_id, mol in zip(group_ids, molecules):
            grouped[group_id].append(mol)

        # Create MolGroup instances
        groups = {}
        for group_id, mol_list in grouped.items():
            groups[group_id] = cls(id=group_id, molecules=mol_list)

        return groups
