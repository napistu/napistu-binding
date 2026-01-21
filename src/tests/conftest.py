"""Shared fixtures for napistu_binding tests."""

import os
import sys

from pytest import fixture, skip

from napistu_binding.load.sdf import extract_molecules_from_sdf

try:
    from napistu_binding.load.nstructure import nStructure
except ImportError:
    # nStructure may not be available if BioPython is not installed
    nStructure = None


@fixture
def test_data_path():
    """Path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@fixture
def mini_chebi_molecules(test_data_path):
    """Load molecules from mini_chebi.sdf test file."""
    sdf_path = os.path.join(test_data_path, "mini_chebi.sdf")
    return extract_molecules_from_sdf(sdf_path, remove_invalid=False)


@fixture
def weird_chebi_molecules(test_data_path):
    """Load molecules from weird_chebi.sdf test file (contains coordinate bonds)."""
    sdf_path = os.path.join(test_data_path, "weird_chebi.sdf")
    # Load with remove_invalid=False so we can test invalid molecules
    return extract_molecules_from_sdf(sdf_path, remove_invalid=False)


@fixture
def pdb_file_path(test_data_path):
    """Path to 1CRN.pdb test file."""
    return os.path.join(test_data_path, "1CRN.pdb")


@fixture
def nstructure(pdb_file_path):
    """Load nStructure from 1CRN.pdb test file."""
    if nStructure is None:
        skip("BioPython not available")
    return nStructure.from_pdb_file(pdb_file_path)


# Define custom markers for platforms
def pytest_configure(config):
    config.addinivalue_line("markers", "skip_on_windows: mark test to skip on Windows")
    config.addinivalue_line("markers", "skip_on_macos: mark test to skip on macOS")
    config.addinivalue_line(
        "markers", "unix_only: mark test to run only on Unix/Linux systems"
    )


# Define platform conditions
is_windows = sys.platform == "win32"
is_macos = sys.platform == "darwin"
is_unix = not (is_windows or is_macos)


# Apply skipping based on platform
def pytest_runtest_setup(item):
    # Skip tests marked to be skipped on Windows
    if is_windows and any(
        mark.name == "skip_on_windows" for mark in item.iter_markers()
    ):
        skip("Test skipped on Windows")

    # Skip tests marked to be skipped on macOS
    if is_macos and any(mark.name == "skip_on_macos" for mark in item.iter_markers()):
        skip("Test skipped on macOS")

    # Skip tests that should run only on Unix
    if not is_unix and any(mark.name == "unix_only" for mark in item.iter_markers()):
        skip("Test runs only on Unix systems")
