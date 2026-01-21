"""Test functions for loading PDB files with nStructure."""

import os
import tempfile

import pytest

try:
    from Bio.PDB import Structure
except ImportError:
    pytest.skip("BioPython not available", allow_module_level=True)

from napistu_binding.load.nstructure import nStructure


def test_from_pdb_file(pdb_file_path):
    """Test that from_pdb_file loads a structure from a PDB file."""
    structure = nStructure.from_pdb_file(pdb_file_path)

    # Verify it's an nStructure instance
    assert isinstance(structure, nStructure)
    assert isinstance(structure, Structure.Structure)

    # Verify it has content (at least one model)
    assert len(list(structure.get_models())) > 0


def test_nstructure_fixture(nstructure):
    """Test that nstructure fixture loads correctly."""
    assert isinstance(nstructure, nStructure)
    assert isinstance(nstructure, Structure.Structure)
    assert len(list(nstructure.get_models())) > 0
    assert nstructure.id == "1CRN"
    assert len(list(nstructure.get_residues())) == 46


def test_to_pdb_file(nstructure):
    """Test that nStructure can be written to a PDB file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        nstructure.to_pdb_file(output_path)

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify file has content
        assert os.path.getsize(output_path) > 0

        # Verify we can read it back
        reloaded = nStructure.from_pdb_file(output_path)
        assert isinstance(reloaded, nStructure)
        assert len(list(reloaded.get_models())) > 0
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


def test_structure_attributes_and_methods(nstructure):
    """Test that normal Structure attributes and methods work on nStructure."""
    # Test id attribute
    assert hasattr(nstructure, "id")
    assert nstructure.id is not None

    # Test get_models() method
    models = list(nstructure.get_models())
    assert len(models) > 0

    # Test get_chains() method
    for model in models:
        chains = list(model.get_chains())
        assert len(chains) > 0

        # Test get_residues() method
        for chain in chains:
            residues = list(chain.get_residues())
            assert len(residues) > 0

            # Test get_atoms() method
            for residue in residues:
                atoms = list(residue.get_atoms())
                assert len(atoms) > 0
