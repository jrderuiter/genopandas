"""Tests for tree-related classes/functions."""

import pytest

from genopandas.core.tree import Interval, GenomicIntervalTree

# pylint: disable=redefined-outer-name,no-self-use


@pytest.fixture()
def simple_tuples():
    """Tuples for simple tree."""
    return [('2', 10, 20), ('2', 30, 40), ('3', 15, 25), ('3', 20, 25)]


@pytest.fixture()
def simple_tree(simple_tuples):
    """Simple tree containing a few intervals."""
    return GenomicIntervalTree.from_tuples(simple_tuples)


class TestGenomicIntervalTree(object):
    """Unit tests for the GenomicIntervalTree class."""

    def test_from_tuples(self, simple_tuples):
        """Build example tree from a set of tuples."""

        tree = GenomicIntervalTree.from_tuples(simple_tuples)

        assert len(tree) == 2
        assert Interval(10, 20) in tree['2']
        assert Interval(15, 25) in tree['3']

    def test_search(self, simple_tree):
        """Test search function."""

        # Check overlapping region.
        assert len(simple_tree.search('2', 5, 15)) == 1

        # Check region just outside interval (end is not inclusive).
        assert len(simple_tree.search('2', 5, 10)) == 0

        # Check regions on other chromosome.
        assert len(simple_tree.search('3', 15, 20)) == 1
        assert len(simple_tree.search('3', 20, 30)) == 2

        # Check region on non-existant chromosome.
        with pytest.raises(KeyError):
            simple_tree.search('4', 30, 40)

    def test_search_via_index(self, simple_tree):
        """Test search via index."""

        # Check overlapping region.
        assert len(simple_tree['2'].search(5, 15)) == 1

        # Check region just outside interval (end is not inclusive).
        assert len(simple_tree['2'].search(5, 10)) == 0

        # Check regions on other chromosome.
        assert len(simple_tree['3'].search(15, 20)) == 1
        assert len(simple_tree['3'].search(20, 30)) == 2

        # Check region on non-existant chromosome.
        with pytest.raises(KeyError):
            simple_tree['4'].search(30, 40)

    def test_intersection(self, simple_tree):
        """Test intersection."""

        tree_b = GenomicIntervalTree.from_tuples([('2', 10, 20)])
        intersection = simple_tree.intersection(tree_b)

        # Should drop unshared chromosome 3.
        assert len(intersection) == 1
        assert len(intersection['2']) == 1

        # Test operator.
        assert intersection == (tree_b & simple_tree)

    def test_union(self, simple_tree):
        """Test union."""

        tree_b = GenomicIntervalTree.from_tuples(
            [('3', 30, 40), ('4', 20, 30)])  # yapf: disable

        merged = simple_tree.union(tree_b)
        assert len(merged) == 3

        # CHeck presence of intervals.
        assert Interval(30, 40) in merged['3']
        assert Interval(20, 30) in merged['4']

        # Check original trees are unchanged.
        assert len(simple_tree) == 2
        assert len(tree_b) == 2

        # Test operator.
        assert simple_tree.union(tree_b) == (simple_tree | tree_b)

    def test_difference(self, simple_tree):
        """Test difference."""

        tree_b = GenomicIntervalTree.from_tuples(
            [('2', 10, 20), ('2', 30, 40), ('3', 5, 10)])  # yapf: disable
        diff = simple_tree.difference(tree_b)

        # Should not have dropped chromosome 2, but it should be empty..
        assert len(diff) == 2
        assert diff['2'].is_empty()

        # Chromosome 3 should be unchanged.
        assert diff['3'] == simple_tree['3']

        # Test operator.
        assert simple_tree.difference(tree_b) == (simple_tree - tree_b)
