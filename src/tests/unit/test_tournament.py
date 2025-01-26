from unittest.mock import patch

import pytest
from test_common import sample_data

from population import Population
from tree import DecisionTree


@pytest.fixture
def sample_trees():
    """Create sample trees with different fitness values"""
    trees = []
    for i in range(4):
        tree = DecisionTree([None], [i])
        tree.fitness = i
        trees.append(tree)
    return trees


@pytest.fixture
def population_with_trees(sample_data, sample_trees):
    """Create population with predefined trees"""
    X, y = sample_data
    pop = Population(
        X,
        y,
        population_size=len(sample_trees),
        attributes=[0],
        possible_thresholds={0: [1.0]},
        max_depth=1,
        p_split=0.5,
    )
    pop.individuals = sample_trees
    return pop


def test_tournament_selection_size(population_with_trees):
    selected = population_with_trees.tournament_selection(tournament_size=2)
    assert len(selected) == len(population_with_trees.individuals)


def test_tournament_selects_better_fitness(population_with_trees):
    with patch("random.sample") as mock_sample:
        mock_sample.return_value = population_with_trees.individuals[:2]

        selected = population_with_trees.tournament_selection(tournament_size=2)
        assert selected[0].fitness == 0


def test_tournament_entire_population(population_with_trees):
    with patch("random.sample") as mock_sample:
        mock_sample.return_value = population_with_trees.individuals

        selected = population_with_trees.tournament_selection(tournament_size=4)
        assert selected[0].fitness == 0


def test_tournament_multiple_selections(population_with_trees):
    with patch("random.sample") as mock_sample:
        mock_sample.return_value = population_with_trees.individuals[:2]

        selected = population_with_trees.tournament_selection(tournament_size=2)
        assert all(tree.fitness == 0 for tree in selected)
        assert len(selected) == len(population_with_trees.individuals)
