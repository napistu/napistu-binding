"""Shared fixtures for napistu_binding tests."""

import os

import pytest

from napistu_binding.load.sdf import extract_molecules_from_sdf


@pytest.fixture
def test_data_path():
    """Path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def mini_chebi_molecules(test_data_path):
    """Load molecules from mini_chebi.sdf test file."""
    sdf_path = os.path.join(test_data_path, "mini_chebi.sdf")
    return extract_molecules_from_sdf(sdf_path, remove_invalid=False)


@pytest.fixture
def weird_chebi_molecules(test_data_path):
    """Load molecules from weird_chebi.sdf test file (contains coordinate bonds)."""
    sdf_path = os.path.join(test_data_path, "weird_chebi.sdf")
    # Load with remove_invalid=False so we can test invalid molecules
    return extract_molecules_from_sdf(sdf_path, remove_invalid=False)
