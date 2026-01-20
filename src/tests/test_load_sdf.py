"""Test functions for loading SDF files."""

import os

import pytest

try:
    from rdkit import Chem
except ImportError:
    pytest.skip("RDKit not available", allow_module_level=True)

from napistu_binding.load.nmol import nMol
from napistu_binding.load.sdf import extract_molecules_from_sdf


def test_extract_molecules_from_sdf_returns_nmol(mini_chebi_molecules):
    """Test that extract_molecules_from_sdf returns nMol objects."""
    assert len(mini_chebi_molecules) == 10
    assert all(isinstance(mol, nMol) for mol in mini_chebi_molecules)
    assert all(isinstance(mol, Chem.Mol) for mol in mini_chebi_molecules)


def test_extract_molecules_from_sdf_filters_invalid(weird_chebi_molecules):
    """Test that extract_molecules_from_sdf with remove_invalid=True filters invalid molecules."""
    # weird_chebi_molecules is loaded with remove_invalid=False
    # So it should contain the invalid molecule
    assert len(weird_chebi_molecules) == 1

    # Now test with remove_invalid=True - should filter it out
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
    sdf_path = os.path.join(test_data_path, "weird_chebi.sdf")

    filtered_molecules = extract_molecules_from_sdf(sdf_path, remove_invalid=True)
    assert len(filtered_molecules) == 0, "Invalid molecules should be filtered out"
