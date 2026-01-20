"""Test that fixtures work correctly."""

import pytest

try:
    from rdkit.Chem import Mol
except ImportError:
    pytest.skip("RDKit not available", allow_module_level=True)


def test_mini_chebi_molecules_fixture(mini_chebi_molecules):
    """Test that mini_chebi_molecules fixture works and returns expected structure."""
    assert isinstance(mini_chebi_molecules, list)
    assert len(mini_chebi_molecules) == 10
    assert all(isinstance(mol, Mol) for mol in mini_chebi_molecules)
