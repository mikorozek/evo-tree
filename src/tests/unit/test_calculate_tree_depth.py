from test_common import (balanced_tree, single_node_tree,
                         two_decision_node_tree, very_unbalanced_tree)

from tree import DecisionTree


def test_single_node_depth(single_node_tree):
    assert single_node_tree.calculate_depth() == 0


def test_two_decision_node_depth(two_decision_node_tree):
    assert two_decision_node_tree.calculate_depth() == 2


def test_balanced_tree_depth(balanced_tree):
    assert balanced_tree.calculate_depth() == 2


def test_very_unbalanced_tree_depth(very_unbalanced_tree):
    assert very_unbalanced_tree.calculate_depth() == 3


def test_empty_tree_depth():
    tree = DecisionTree(attributes=[], thresholds=[])
    assert tree.calculate_depth() == 0
