import pytest
from population import Population
from tree import DecisionTree
from unittest.mock import patch

@pytest.fixture
def single_node_tree():
   """
   indices (1-based):
   1: [leaf, class 0]
   
   structure:
   [1: leaf (class 0)]
   
   representation:
   [1: leaf class:0]
   """
   return DecisionTree(
       attributes=[None],
       thresholds=[0],
       max_depth=0
   )

@pytest.fixture
def two_decision_node_tree():
   """
   indices (1-based):
   1: [x1 ≤ 4]
   2: [leaf, class 1]  
   3: [x5 ≤ 7]
   4: [leaf, class 0]
   5: [leaf, class 1]
   
   representation:
        x1 ≤ 4
          |
    +-----+-----+
    |           |
[leaf class:1]  x5 ≤ 7
                |
          +-----+-----+
          |           |
   [leaf class:0] [leaf class:1]
   """
   return DecisionTree(
       attributes=[1, None, 5, None, None, None, None],
       thresholds=[4, 1, 7, None, None, 2, 1],
       max_depth=2
   )

@pytest.fixture
def balanced_tree():
   """
   indices (1-based):
   1: [x1 ≤ 5]
   2: [x2 ≤ 3]
   3: [x3 ≤ 7]
   4: [leaf, class 0]
   5: [leaf, class 1]
   6: [leaf, class 1]
   7: [leaf, class 0]
   
   representation:
           x1 ≤ 5
              |
       +------+------+
       |             |
    x2 ≤ 3        x3 ≤ 7
       |             |
   +---+---+     +---+---+
   |       |     |       |
[class:0] [class:1] [class:1] [class:0]
   """
   return DecisionTree(
       attributes=[1, 2, 3, None, None, None, None],
       thresholds=[5, 3, 7, 0, 1, 1, 0],
       max_depth=2
   )


def are_subtrees_equal(tree1: DecisionTree, tree2: DecisionTree, idx1: int, idx2: int) -> bool:
    """
    recursively checks if two subtrees are identical.

    args:
        tree1: first decision tree
        tree2: second decision tree
        idx1: root index of the first subtree
        idx2: root index of the second subtree
    """
    if idx1 >= len(tree1.attributes) or idx2 >= len(tree2.attributes):
        return False
        
    if tree1.attributes[idx1] != tree2.attributes[idx2]:
        return False
    if tree1.thresholds[idx1] != tree2.thresholds[idx2]:
        return False
        
    if tree1.attributes[idx1] is None:
        return True
        
    left_equal = are_subtrees_equal(tree1, tree2, 2*idx1 + 1, 2*idx2 + 1)
    right_equal = are_subtrees_equal(tree1, tree2, 2*idx1 + 2, 2*idx2 + 2)
    
    return left_equal and right_equal

def assert_subtree_unchanged(result: DecisionTree, original: DecisionTree, start_idx: int, excluded_idx: int) -> bool:
    """
    verifies that all nodes except the excluded subtree remain unchanged after crossover.

    args:
        result: tree after crossover operation
        original: original tree before crossover
        start_idx: starting index for comparison
        excluded_idx: root index of the subtree that was replaced
    """
    if start_idx >= len(original.attributes):
        return True
        
    if start_idx == excluded_idx:
        return True
        
    if result.attributes[start_idx] != original.attributes[start_idx]:
        return False
    if result.thresholds[start_idx] != original.thresholds[start_idx]:
        return False
        
    if original.attributes[start_idx] is None:
        return True
        
    return (assert_subtree_unchanged(result, original, 2*start_idx + 1, excluded_idx) and 
            assert_subtree_unchanged(result, original, 2*start_idx + 2, excluded_idx))

def print_crossover_visualization(parent1, parent2, child1, child2, mock_indices):
    """
    Displays trees before and after crossover operation
    """
    print("\n" + "="*50)
    print(f"CROSSOVER with indices: {mock_indices}")
    print("="*50)
    
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
    print("\n" + "="*50 + "\n")

def test_crossover_with_single_node(single_node_tree, two_decision_node_tree):
    with patch('random.choice') as mock_choice:
        mock_indices = [0, 0]
        mock_choice.side_effect = mock_indices
        
        child1, child2 = Population.crossover(single_node_tree, two_decision_node_tree)
        print_crossover_visualization(
            single_node_tree, 
            two_decision_node_tree, 
            child1, 
            child2, 
            mock_indices
        )
        
        assert are_subtrees_equal(child1, two_decision_node_tree, 0, 0)
        assert are_subtrees_equal(child2, single_node_tree, 0, 0)

def test_crossover_replaces_subtree(two_decision_node_tree, balanced_tree):
    with patch('random.choice') as mock_choice:
        mock_indices = [2, 0]
        mock_choice.side_effect = mock_indices
        
        child1, child2 = Population.crossover(two_decision_node_tree, balanced_tree)
        print_crossover_visualization(
            two_decision_node_tree, 
            balanced_tree, 
            child1, 
            child2, 
            mock_indices
        )
        
        assert are_subtrees_equal(child1, balanced_tree, 2, 0)
        assert are_subtrees_equal(child2, two_decision_node_tree, 0, 2)
        assert assert_subtree_unchanged(child1, two_decision_node_tree, 0, 2)
        assert assert_subtree_unchanged(child2, balanced_tree, 0, 0)

def test_crossover_entire_tree_replaced_with_subtree(two_decision_node_tree, balanced_tree):
    with patch('random.choice') as mock_choice:
        mock_indices = [0, 1]
        mock_choice.side_effect = mock_indices
        
        child1, child2 = Population.crossover(two_decision_node_tree, balanced_tree)
        print_crossover_visualization(
            two_decision_node_tree, 
            balanced_tree, 
            child1, 
            child2, 
            mock_indices
        )
        
        assert are_subtrees_equal(child1, balanced_tree, 0, 1)
        assert are_subtrees_equal(child2, two_decision_node_tree, 1, 0)

def test_crossover_with_leaf_node(two_decision_node_tree, balanced_tree):
    with patch('random.choice') as mock_choice:
        mock_indices = [1, 1]
        mock_choice.side_effect = mock_indices
        
        child1, child2 = Population.crossover(two_decision_node_tree, balanced_tree)
        print_crossover_visualization(
            two_decision_node_tree, 
            balanced_tree, 
            child1, 
            child2, 
            mock_indices
        )
        
        assert are_subtrees_equal(child2, two_decision_node_tree, 1, 1)
        assert are_subtrees_equal(child1, balanced_tree, 1, 1)
        assert assert_subtree_unchanged(child1, two_decision_node_tree, 0, 1)
        assert assert_subtree_unchanged(child2, balanced_tree, 0, 1)
