"""Test functions for NapistuMol class."""

import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import Mol
except ImportError:
    Chem = None
    Mol = None
    pytest.skip("RDKit not available", allow_module_level=True)

from napistu_binding.load.napistu_mol import NapistuMol


def test_napistumol_standard_mol_methods():
    """Test that standard RDKit Mol methods work on NapistuMol."""
    # Create a simple molecule
    mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
    napistu_mol = NapistuMol(mol)

    # Test standard Mol methods work
    assert napistu_mol.GetNumAtoms() == 4  # C, C, O, O
    assert napistu_mol.GetNumBonds() == 3
    assert napistu_mol.GetNumConformers() == 0  # no 3D coordinates

    # Test we can get atom information
    atom = napistu_mol.GetAtomWithIdx(0)
    assert atom.GetSymbol() == "C"

    # Test we can iterate over atoms
    atoms = [atom.GetSymbol() for atom in napistu_mol.GetAtoms()]
    assert "C" in atoms
    assert "O" in atoms

    # Test we can get bonds
    bond = napistu_mol.GetBondWithIdx(0)
    assert bond is not None

    # Test isinstance works correctly
    assert isinstance(napistu_mol, Chem.Mol)
    assert isinstance(napistu_mol, NapistuMol)


def test_napistumol_from_smiles(mini_chebi_molecules):
    """Test that NapistuMol.from_smiles works on molecules from fixture."""
    # Convert regular Mol objects to NapistuMol
    napistu_mols = [NapistuMol(mol) for mol in mini_chebi_molecules]

    assert len(napistu_mols) == 10
    assert all(isinstance(mol, NapistuMol) for mol in napistu_mols)
    assert all(isinstance(mol, Chem.Mol) for mol in napistu_mols)


def test_napistumol_to_isomeric_smiles_on_fixture(mini_chebi_molecules):
    """Test that to_isomeric_smiles works on all NapistuMol molecules in the fixture."""
    napistu_mols = [NapistuMol(mol) for mol in mini_chebi_molecules]
    smiles_list = [mol.to_isomeric_smiles() for mol in napistu_mols]

    assert len(smiles_list) == 10
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert all(len(smiles) > 0 for smiles in smiles_list)


def test_napistumol_is_valid_on_fixture(mini_chebi_molecules):
    """Test that is_valid works on valid molecules."""
    napistu_mols = [NapistuMol(mol) for mol in mini_chebi_molecules]
    valid_results = [mol.is_valid() for mol in napistu_mols]

    assert len(valid_results) == 10
    assert all(isinstance(result, bool) for result in valid_results)
    # All molecules in mini_chebi should be valid
    assert all(valid_results), "All molecules in mini_chebi should be valid"


def test_napistumol_weird_chebi_coordinate_bonds_filtered(weird_chebi_molecules):
    """Test that weird_chebi.sdf with coordinate bonds is filtered out by NapistuMol.is_valid()."""
    # Load the molecule - it has coordinate bonds (type 8) representing hydrogen bonds
    assert len(weird_chebi_molecules) == 1

    mol = weird_chebi_molecules[0]
    napistu_mol = NapistuMol(mol)

    # The SDF has SMILES with tilde (~) for coordinate bonds:
    # Nc1nc2n3c(n1)N[H]~O=C1NC(=O)NC(=O~[H]N2)N1[H]~3
    # This molecule has coordinate bonds that cause issues with SMILES conversion

    # The filter should reject this molecule because it can't be converted to SMILES
    result = napistu_mol.is_valid(verbose=True)
    assert not result, "Molecule with coordinate bonds should be filtered out"

    # Verify it gets filtered from a list
    napistu_mols = [NapistuMol(m) for m in weird_chebi_molecules]
    valid_molecules = [m for m in napistu_mols if m.is_valid()]
    assert (
        len(valid_molecules) == 0
    ), "All molecules with coordinate bonds should be filtered"


def test_napistumol_from_smiles_classmethod():
    """Test NapistuMol.FromSmiles classmethod."""
    napistu_mol = NapistuMol.FromSmiles("CC(=O)O")

    assert isinstance(napistu_mol, NapistuMol)
    assert isinstance(napistu_mol, Chem.Mol)
    assert napistu_mol.GetNumAtoms() == 4


def test_napistumol_from_mol_classmethod(mini_chebi_molecules):
    """Test NapistuMol.from_mol classmethod."""
    mol = mini_chebi_molecules[0]
    napistu_mol = NapistuMol.from_mol(mol)

    assert isinstance(napistu_mol, NapistuMol)
    assert isinstance(napistu_mol, Chem.Mol)
    assert napistu_mol.GetNumAtoms() == mol.GetNumAtoms()
