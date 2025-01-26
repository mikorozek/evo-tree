import pytest
import numpy as np
from population import Population
from tree import DecisionTree

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
   )

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    y = np.array([0, 1, 1])
    return X, y

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
   )

@pytest.fixture
def very_unbalanced_tree():
    """
    indices (0-based):
    0: [x1 ≤ 5]
    1: [leaf, class 0]
    2: [x3 ≤ 2]
    3: [null, null]  # placeholder
    4: [null, null]  # placeholder
    5: [x4 ≤ 1]     # left child of node 2
    6: [leaf, class 1] # right child of node 2
    7-10: [null, null] # placeholders
    11: [leaf, class 0] # left child of node 5
    12: [leaf, class 1] # right child of node 5
    
    representation:
           x1 ≤ 5
              |
       +------+------+
       |             |
  [class:0]       x3 ≤ 2
                     |
                +----+----+
                |         |
              x4 ≤ 1  [class:1]
                |
           +----+----+
           |         |
     [class:0]  [class:1]
    """
    return DecisionTree(
        attributes=[1, None, 3, None, None, 4, None, None, None, None, None, None, None],
        thresholds=[5, 0, 2, None, None, 1, 1, None, None, None, None, 0, 1]
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
