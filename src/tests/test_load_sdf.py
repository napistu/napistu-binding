"""Test functions for loading SDF files."""

import pytest

try:
    from rdkit.Chem import Mol
except ImportError:
    Mol = None  # type: ignore
    pytest.skip("RDKit not available", allow_module_level=True)

from napistu_binding.load.sdf import is_valid_molecule, mol_to_isomeric_smiles


def test_mol_to_isomeric_smiles_on_fixture(mini_chebi_molecules):
    """Test that mol_to_isomeric_smiles works on all molecules in the fixture."""
    smiles_list = [mol_to_isomeric_smiles(mol) for mol in mini_chebi_molecules]

    assert len(smiles_list) == 10
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert all(len(smiles) > 0 for smiles in smiles_list)


def test_weird_chebi_sdf_coordinate_bonds_filtered(weird_chebi_molecules):
    """Test that weird_chebi.sdf with coordinate bonds is filtered out."""
    # Load the molecule - it has coordinate bonds (type 8) representing hydrogen bonds
    assert len(weird_chebi_molecules) == 1

    mol = weird_chebi_molecules[0]

    # The SDF has SMILES with tilde (~) for coordinate bonds:
    # Nc1nc2n3c(n1)N[H]~O=C1NC(=O)NC(=O~[H]N2)N1[H]~3
    # This molecule has coordinate bonds that cause issues with SMILES conversion

    # The filter should reject this molecule because it can't be converted to SMILES
    result = is_valid_molecule(mol, verbose=True)
    assert not result, "Molecule with coordinate bonds should be filtered out"

    # Verify it gets filtered from a list
    valid_molecules = [m for m in weird_chebi_molecules if is_valid_molecule(m)]
    assert (
        len(valid_molecules) == 0
    ), "All molecules with coordinate bonds should be filtered"
