from unittest.mock import patch

from test_common import (are_subtrees_equal, assert_subtree_unchanged,
                         balanced_tree, single_node_tree,
                         two_decision_node_tree, very_unbalanced_tree)

from population import Population


def print_crossover_visualization(parent1, parent2, child1, child2, mock_indices):
    """
    Displays trees before and after crossover operation
    """
    print("\n" + "=" * 50)
    print(f"CROSSOVER with indices: {mock_indices}")
    print("=" * 50)

    print("\nParent trees:")
    print("\nParent 1:")
    parent1.print_tree()
    print("\nParent 2:")
    parent2.print_tree()

    print("\nAfter crossover:")
    print("\nChild 1:")
    child1.print_tree()
    print("\nChild 2:")
    child2.print_tree()
    print("\n" + "=" * 50 + "\n")


def test_crossover_with_single_node(single_node_tree, two_decision_node_tree):
    with patch("random.choice") as mock_choice:
        mock_indices = [0, 0]
        mock_choice.side_effect = mock_indices

        child1, child2 = Population.crossover(single_node_tree, two_decision_node_tree)
        print_crossover_visualization(
            single_node_tree, two_decision_node_tree, child1, child2, mock_indices
        )

        assert are_subtrees_equal(child1, two_decision_node_tree, 0, 0)
        assert are_subtrees_equal(child2, single_node_tree, 0, 0)


def test_crossover_replaces_subtree(two_decision_node_tree, balanced_tree):
    with patch("random.choice") as mock_choice:
        mock_indices = [2, 0]
        mock_choice.side_effect = mock_indices

        child1, child2 = Population.crossover(two_decision_node_tree, balanced_tree)
        print_crossover_visualization(
            two_decision_node_tree, balanced_tree, child1, child2, mock_indices
        )

        assert are_subtrees_equal(child1, balanced_tree, 2, 0)
        assert are_subtrees_equal(child2, two_decision_node_tree, 0, 2)
        assert assert_subtree_unchanged(child1, two_decision_node_tree, 0, 2)
        assert assert_subtree_unchanged(child2, balanced_tree, 0, 0)


def test_crossover_entire_tree_replaced_with_subtree(
    two_decision_node_tree, balanced_tree
):
    with patch("random.choice") as mock_choice:
        mock_indices = [0, 1]
        mock_choice.side_effect = mock_indices

        child1, child2 = Population.crossover(two_decision_node_tree, balanced_tree)
        print_crossover_visualization(
            two_decision_node_tree, balanced_tree, child1, child2, mock_indices
        )

        assert are_subtrees_equal(child1, balanced_tree, 0, 1)
        assert are_subtrees_equal(child2, two_decision_node_tree, 1, 0)


def test_crossover_with_leaf_node(two_decision_node_tree, balanced_tree):
    with patch("random.choice") as mock_choice:
        mock_indices = [1, 1]
        mock_choice.side_effect = mock_indices

        child1, child2 = Population.crossover(two_decision_node_tree, balanced_tree)
        print_crossover_visualization(
            two_decision_node_tree, balanced_tree, child1, child2, mock_indices
        )

        assert are_subtrees_equal(child2, two_decision_node_tree, 1, 1)
        assert are_subtrees_equal(child1, balanced_tree, 1, 1)
        assert assert_subtree_unchanged(child1, two_decision_node_tree, 0, 1)
        assert assert_subtree_unchanged(child2, balanced_tree, 0, 1)


def test_crossover_with_very_unbalanced_trees(very_unbalanced_tree, balanced_tree):
    """
    Tests crossover when an unbalanced tree requires placeholder nodes (None, None) in its array representation.
    The very_unbalanced_tree needs several placeholder nodes to maintain proper parent-child relationships,
    which is necessary because children of node i are stored at positions 2i+1 and 2i+2.
    For example, node 5 (x4 ≤ 1) has children at positions 11 and 12, requiring placeholder nodes in between.
    """
    with patch("random.choice") as mock_choice:
        mock_indices = [2, 1]
        mock_choice.side_effect = mock_indices

        child1, child2 = Population.crossover(very_unbalanced_tree, balanced_tree)
        print_crossover_visualization(
            very_unbalanced_tree, balanced_tree, child1, child2, mock_indices
        )

        assert are_subtrees_equal(child1, balanced_tree, 2, 1)
        assert are_subtrees_equal(child2, very_unbalanced_tree, 1, 2)
        assert assert_subtree_unchanged(child1, very_unbalanced_tree, 0, 2)
        assert assert_subtree_unchanged(child2, balanced_tree, 0, 1)
