"""Test functions for MolGroup class."""

import pytest

try:
    from napistu_binding.load.molgroup import MolGroup
except ImportError:
    pytest.skip("RDKit not available", allow_module_level=True)


def test_molgroup_from_fixture_groups_into_2_sets(mini_chebi_molecules):
    """Test that MolGroup can group the 10 molecules from the fixture into 2 sets."""
    # The fixture contains 10 molecules which are already nMol objects
    molecules = mini_chebi_molecules
    assert len(molecules) == 10

    # Create artificial group labels: first 5 are "A", next 5 are "B"
    group_labels = ["A"] * 5 + ["B"] * 5
    assert len(group_labels) == 10

    # Group molecules by labels
    groups = MolGroup.from_aligned_lists(group_labels, molecules)

    # Verify we have exactly 2 groups
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)} groups"
    assert "A" in groups
    assert "B" in groups

    # Verify all groups are MolGroup instances
    assert all(isinstance(group, MolGroup) for group in groups.values())

    # Verify each group has an id and molecules
    for group_id, group in groups.items():
        assert group.id == group_id
        assert isinstance(group.id, str)
        assert isinstance(group.molecules, list)
        assert len(group.molecules) == 5  # Each group should have 5 molecules

    # Verify total molecules match
    total_molecules = sum(len(group.molecules) for group in groups.values())
    assert total_molecules == 10, f"Total molecules should be 10, got {total_molecules}"

    # Verify each molecule appears exactly once across all groups
    all_grouped_molecules = []
    for group in groups.values():
        all_grouped_molecules.extend(group.molecules)
    assert len(all_grouped_molecules) == 10
    assert len(set(all_grouped_molecules)) == 10  # All unique

    # Verify the groups contain the correct molecules
    assert len(groups["A"].molecules) == 5
    assert len(groups["B"].molecules) == 5
