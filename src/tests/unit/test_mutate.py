from unittest.mock import patch

import numpy as np
import pytest
from test_common import (are_subtrees_equal, single_node_tree,
                         two_decision_node_tree)

from population import Population
from tree import DecisionTree


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 1])
    return X, y


@pytest.fixture
def simple_population(sample_data):
    """Create a simple population instance for testing"""
    X, y = sample_data
    attributes = [0, 1, 2]
    possible_thresholds = {0: [2.0, 3.0], 1: [4.0, 5.0], 2: [6.0, 7.0]}
    return Population(
        X,
        y,
        population_size=1,
        attributes=attributes,
        possible_thresholds=possible_thresholds,
        max_depth=3,
        p_split=0.5,
    )


def test_mutate_decision_node(simple_population, two_decision_node_tree):
    with patch("random.choice") as mock_choice:
        mock_choice.side_effect = [(0, 1), 2, 7.0]

        original_tree = DecisionTree(
            attributes=two_decision_node_tree.attributes.copy(),
            thresholds=two_decision_node_tree.thresholds.copy(),
        )

        simple_population.mutate(
            two_decision_node_tree,
            attributes=[0, 1, 2],
            possible_thresholds={0: [2.0], 1: [4.0], 2: [7.0]},
        )

        assert are_subtrees_equal(two_decision_node_tree, original_tree, 1, 1)
        assert are_subtrees_equal(two_decision_node_tree, original_tree, 2, 2)
        assert two_decision_node_tree.attributes[0] == 2
        assert two_decision_node_tree.thresholds[0] == 7.0


def test_mutate_leaf_node(simple_population, single_node_tree):
    simple_population.mutate(
        single_node_tree,
        attributes=[0, 1, 2],
        possible_thresholds={0: [2.0], 1: [4.0], 2: [7.0]},
    )

    assert single_node_tree.attributes[0] is None
    assert single_node_tree.thresholds[0] == 1
