import numpy as np
import random
from typing import List, Dict
from tree import DecisionTree

class Population:
    def __init__(self, X: np.ndarray, y: np.ndarray, population_size: int, 
                 attributes: List[int], possible_thresholds: Dict[int, List[float]], 
                 max_depth: int):
        self.X = X
        self.y = y
        self.individuals = [
            self._create_random_tree(X, y, attributes, possible_thresholds, max_depth)
            for _ in range(population_size)
        ]
        self.generation = 0
        print(self.X.shape)
    
    @staticmethod
    def _create_random_tree(X: np.ndarray, y: np.ndarray, attributes: List[int], 
                            possible_thresholds: Dict[int, List[float]], max_depth: int, 
                            p_split: float = 0.7) -> DecisionTree:
        size = 2**(max_depth + 1) - 1
        attributes_arr = []
        thresholds_arr = []
        queue = [(0, 0, np.arange(len(X)))]
        
        while queue:
            print(f'attr: {attributes_arr}')
            print(f'trhesh: {thresholds_arr}')
            idx, depth, data_indices = queue.pop(0)
            if len(data_indices) == 0 or idx >= size:
                continue

            if idx >= len(attributes_arr):
                elements_to_add = idx - len(attributes_arr) + 1
                attributes_arr.extend([None] * elements_to_add)
                thresholds_arr.extend([None] * elements_to_add)

            X_subset = X[data_indices]
            y_subset = y[data_indices]
            
            if depth >= max_depth or random.random() > p_split:
                if len(y_subset) > 0:
                    majority_class = np.argmax(np.bincount(y_subset))
                else:
                    majority_class = np.argmax(np.bincount(y))
                attributes_arr[idx] = None
                thresholds_arr[idx] = majority_class
            else:
                attr = random.choice(attributes)
                threshold = random.choice(possible_thresholds[attr])
                attributes_arr[idx] = attr
                thresholds_arr[idx] = threshold
                
                left_indices = data_indices[X_subset[:, attr] <= threshold]
                right_indices = data_indices[X_subset[:, attr] > threshold] 
                
                queue.append((2*idx + 1, depth + 1, left_indices))
                queue.append((2*idx + 2, depth + 1, right_indices))
    
        print(f'attr:{attributes_arr}')
        print(f'thresh:{thresholds_arr}')
        return DecisionTree(attributes_arr, thresholds_arr, max_depth)

    def evaluate_population(self, alpha1: float = 0.99, alpha2: float = 0.01):
        for tree in self.individuals:
            correct = 0
            for xi, yi in zip(self.X, self.y):
                node_idx = 0
                while True:
                    attr = tree.attributes[node_idx]
                    if attr is None:
                        pred = tree.thresholds[node_idx]
                        correct += (pred == yi)
                        break
                    if xi[attr] <= tree.thresholds[node_idx]:
                        node_idx = 2*node_idx + 1
                    else:
                        node_idx = 2*node_idx + 2
            
            accuracy = correct / len(self.y)
            current_depth = tree.calculate_depth()
            tree.fitness = alpha1 * (1 - accuracy) + alpha2 * current_depth

    def tournament_selection(self, tournament_size: int = 2) -> List[DecisionTree]:
        selected = []
        for _ in range(len(self.individuals)):
            contestants = random.sample(self.individuals, tournament_size)
            winner = min(contestants, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    @staticmethod
    def crossover(parent1: DecisionTree, parent2: DecisionTree) -> tuple[DecisionTree, DecisionTree]:
        def get_nodes(parent: DecisionTree) -> list[int]:
            return list(range(len(parent.attributes)))
        
        p1_nodes, p2_nodes = get_nodes(parent1), get_nodes(parent2)
        
        if not p1_nodes or not p2_nodes:
            return parent1, parent2
        
        p1_idx = random.choice(p1_nodes)
        p2_idx = random.choice(p2_nodes)
        
        def copy_subtree(src_idx: int, dest_idx: int, src_tree: DecisionTree, dest_attrs: list, dest_thresholds: list) -> None:
            if dest_idx >= len(dest_attrs):
                dest_attrs.extend([None] * (dest_idx - len(dest_attrs) + 1))
                dest_thresholds.extend([None] * (dest_idx - len(dest_thresholds) + 1))
            
            dest_attrs[dest_idx] = src_tree.attributes[src_idx]
            dest_thresholds[dest_idx] = src_tree.thresholds[src_idx]
            
            if src_tree.attributes[src_idx] is not None:
                copy_subtree(2*src_idx + 1, 2*dest_idx + 1, src_tree, dest_attrs, dest_thresholds)
                copy_subtree(2*src_idx + 2, 2*dest_idx + 2, src_tree, dest_attrs, dest_thresholds)
        
        def create_child(base_parent: DecisionTree, donor_parent: DecisionTree, base_idx: int, donor_idx: int) -> tuple[list, list]:
            new_attrs = base_parent.attributes.copy()
            new_thresholds = base_parent.thresholds.copy()
            copy_subtree(donor_idx, base_idx, donor_parent, new_attrs, new_thresholds)
            return new_attrs, new_thresholds
        
        child1_attrs, child1_thresholds = create_child(parent1, parent2, p1_idx, p2_idx)
        child2_attrs, child2_thresholds = create_child(parent2, parent1, p2_idx, p1_idx)
        
        return (
            DecisionTree(child1_attrs, child1_thresholds, parent1.max_depth),
            DecisionTree(child2_attrs, child2_thresholds, parent2.max_depth)
        )

    def mutate(self, tree: DecisionTree, 
               attributes: List[int], possible_thresholds: Dict[int, List[float]], 
               mutation_rate: float = 0.1) -> None:
        for index, attribute in enumerate(tree.attributes):
            if random.random() < mutation_rate:
                if attribute is None:
                    label = self.get_majority_label_for_leaf(tree, index)
                    tree.thresholds[index] = label
                else:
                    tree.attributes[index] = random.choice(attributes)
                    tree.thresholds[index] = random.choice(possible_thresholds[attribute])

    def get_majority_label_for_leaf(self, tree: DecisionTree, leaf_index: int) -> int:
        matching_indices = []
        for index, xi in enumerate(self.X):
            current_index = 0
            while True:
                if current_index == leaf_index:
                    matching_indices.append(index)
                    break
                attr = tree.attributes[current_index]
                threshold = tree.thresholds[current_index]
                if attr is None:
                    break
                if xi[attr] <= threshold:
                    current_index = 2 * current_index + 1
                else:
                    current_index = 2 * current_index + 2

        if not matching_indices:
            return int(np.argmax(np.bincount(self.y)))

        return int(np.argmax(np.bincount(self.y[matching_indices])))
        

    def create_new_generation(self, attributes: List[int], possible_thresholds: Dict[int, List[float]], 
                              crossover_rate: float = 0.7, mutation_rate: float = 0.2, 
                              elitism: int = 1) -> None:
        elites = sorted(self.individuals, key=lambda x: x.fitness)[:elitism]
        
        selected = self.tournament_selection()
        
        children = []
        for i in range(0, len(selected), 2):
            if i == len(selected) - 1:
                continue

            parent1 = selected[i]
            parent2 = selected[i + 1]
            
            if random.random() < crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
                children.extend([child1, child2])
            else:
                children.extend([parent1, parent2])

        for child in children:
            if random.random() < mutation_rate:
                self.mutate(child, attributes, possible_thresholds)
        
        self.individuals = elites + children[:len(self.individuals)-elitism]
        self.generation += 1

    def get_best(self) -> DecisionTree:
        return min(self.individuals, key=lambda x: x.fitness)

