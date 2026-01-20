"""Test functions for nMol class."""

import pytest

try:
    from rdkit import Chem
except ImportError:
    pytest.skip("RDKit not available", allow_module_level=True)

from napistu_binding.load.nmol import nMol


def test_nmol_standard_mol_methods():
    """Test that standard RDKit Mol methods work on nMol."""
    # Create a simple molecule
    mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
    nmol = nMol(mol)

    # Test standard Mol methods work
    assert nmol.GetNumAtoms() == 4  # C, C, O, O
    assert nmol.GetNumBonds() == 3
    assert nmol.GetNumConformers() == 0  # no 3D coordinates

    # Test we can get atom information
    atom = nmol.GetAtomWithIdx(0)
    assert atom.GetSymbol() == "C"

    # Test we can iterate over atoms
    atoms = [atom.GetSymbol() for atom in nmol.GetAtoms()]
    assert "C" in atoms
    assert "O" in atoms

    # Test we can get bonds
    bond = nmol.GetBondWithIdx(0)
    assert bond is not None

    # Test isinstance works correctly
    assert isinstance(nmol, Chem.Mol)
    assert isinstance(nmol, nMol)


def test_nmol_from_smiles(mini_chebi_molecules):
    """Test that nMol works on molecules from fixture."""
    # Molecules are already nMol objects
    assert len(mini_chebi_molecules) == 10
    assert all(isinstance(mol, nMol) for mol in mini_chebi_molecules)
    assert all(isinstance(mol, Chem.Mol) for mol in mini_chebi_molecules)


def test_nmol_to_isomeric_smiles_on_fixture(mini_chebi_molecules):
    """Test that to_isomeric_smiles works on all nMol molecules in the fixture."""
    smiles_list = [mol.to_isomeric_smiles() for mol in mini_chebi_molecules]

    assert len(smiles_list) == 10
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert all(len(smiles) > 0 for smiles in smiles_list)


def test_nmol_is_valid_on_fixture(mini_chebi_molecules):
    """Test that is_valid works on valid molecules."""
    valid_results = [mol.is_valid() for mol in mini_chebi_molecules]

    assert len(valid_results) == 10
    assert all(isinstance(result, bool) for result in valid_results)
    # All molecules in mini_chebi should be valid
    assert all(valid_results), "All molecules in mini_chebi should be valid"


def test_nmol_weird_chebi_coordinate_bonds_filtered(weird_chebi_molecules):
    """Test that weird_chebi.sdf with coordinate bonds is filtered out by nMol.is_valid()."""
    # Load the molecule - it has coordinate bonds (type 8) representing hydrogen bonds
    assert len(weird_chebi_molecules) == 1

    nmol = weird_chebi_molecules[0]

    # The SDF has SMILES with tilde (~) for coordinate bonds:
    # Nc1nc2n3c(n1)N[H]~O=C1NC(=O)NC(=O~[H]N2)N1[H]~3
    # This molecule has coordinate bonds that cause issues with SMILES conversion

    # The filter should reject this molecule because it can't be converted to SMILES
    result = nmol.is_valid(verbose=True)
    assert not result, "Molecule with coordinate bonds should be filtered out"

    # Verify it gets filtered from a list
    valid_molecules = [m for m in weird_chebi_molecules if m.is_valid()]
    assert (
        len(valid_molecules) == 0
    ), "All molecules with coordinate bonds should be filtered"


def test_nmol_from_smiles_classmethod():
    """Test nMol.FromSmiles classmethod."""
    nmol = nMol.FromSmiles("CC(=O)O")

    assert isinstance(nmol, nMol)
    assert isinstance(nmol, Chem.Mol)
    assert nmol.GetNumAtoms() == 4


def test_nmol_from_mol_classmethod(mini_chebi_molecules):
    """Test nMol.from_mol classmethod."""
    mol = mini_chebi_molecules[0]
    nmol = nMol.from_mol(mol)

    assert isinstance(nmol, nMol)
    assert isinstance(nmol, Chem.Mol)
    assert nmol.GetNumAtoms() == mol.GetNumAtoms()
