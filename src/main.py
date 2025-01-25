import numpy as np
import random
from typing import List, Dict

class DecisionTree:
    def __init__(self, attributes: List[int], thresholds: List[float], max_depth: int):
        self.attributes = attributes
        self.thresholds = thresholds
        self.max_depth = max_depth
        self.fitness = float('inf')
    
    def __repr__(self):
        return f"Attributes: {self.attributes}\nThresholds: {self.thresholds}\nFitness: {self.fitness:.4f}"

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
    
    @staticmethod
    def _create_random_tree(X: np.ndarray, y: np.ndarray, attributes: List[int], 
                            possible_thresholds: Dict[int, List[float]], max_depth: int, 
                            p_split: float = 0.7) -> DecisionTree:
        attributes_arr = []
        thresholds_arr = []
        queue = [(0, 0, list(range(len(X))))]
        
        while queue:
            idx, depth, data_indices = queue.pop(0)
            X_subset = X[data_indices]
            y_subset = y[data_indices]
            
            if depth >= max_depth or random.random() > p_split:
                attributes_arr.append(None)
                majority_class = np.argmax(np.bincount(y_subset)) if len(y_subset) > 0 else 0
                thresholds_arr.append(majority_class)
            else:
                attr = random.choice(attributes)
                threshold = random.choice(possible_thresholds[attr])
                attributes_arr.append(attr)
                thresholds_arr.append(threshold)
                
                left_mask = X_subset[:, attr] <= threshold
                left_indices = [data_indices[i] for i in np.where(left_mask)[0]]
                right_indices = [data_indices[i] for i in np.where(~left_mask)[0]]
                
                queue.append((2*idx + 1, depth + 1, left_indices))
                queue.append((2*idx + 2, depth + 1, right_indices))
    
        return DecisionTree(attributes_arr, thresholds_arr, max_depth)

    def evaluate_population(self, alpha1: float = 0.99, alpha2: float = 0.01):
        for tree in self.individuals:
            correct = 0
            for xi, yi in zip(self.X, self.y):
                node_idx = 0
                while True:
                    attr = tree.attributes[node_idx] if node_idx < len(tree.attributes) else None
                    if attr is None:
                        pred = tree.thresholds[node_idx]
                        correct += (pred == yi)
                        break
                    if xi[attr] <= tree.thresholds[node_idx]:
                        node_idx = 2*node_idx + 1
                    else:
                        node_idx = 2*node_idx + 2
            
            accuracy = correct / len(y)
            current_depth = int(np.log2(len(tree.attributes))) if tree.attributes else 0
            tree.fitness = alpha1 * (1 - accuracy) + alpha2 * current_depth

    def tournament_selection(self, tournament_size: int = 3) -> List[DecisionTree]:
        selected = []
        for _ in range(len(self.individuals)):
            contestants = random.sample(self.individuals, tournament_size)
            winner = min(contestants, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def crossover(self, parent1: DecisionTree, parent2: DecisionTree) -> DecisionTree:
        min_len = min(len(parent1.attributes), len(parent2.attributes))
        if min_len == 0:
            return parent1
        
        crossover_point = random.randint(1, min_len - 1)
        child_attrs = parent1.attributes[:crossover_point] + parent2.attributes[crossover_point:]
        child_thresholds = parent1.thresholds[:crossover_point] + parent2.thresholds[crossover_point:]
        return DecisionTree(child_attrs, child_thresholds, parent1.max_depth)

    def mutate(self, tree: DecisionTree, 
               attributes: List[int], possible_thresholds: Dict[int, List[float]], 
               mutation_rate: float = 0.1) -> None:
        for i in range(len(tree.attributes)):
            if random.random() < mutation_rate:
                if tree.attributes[i] is None:
                    data_in_leaf = self._get_data_for_leaf(self.X, tree, i)
                    y_subset = self.y[data_in_leaf]
                    majority_class = np.argmax(np.bincount(y_subset)) if len(y_subset) > 0 else 0
                    tree.thresholds[i] = majority_class
                else:
                    tree.attributes[i] = random.choice(attributes)
                    tree.thresholds[i] = random.choice(possible_thresholds[tree.attributes[i]])

    def _get_data_for_leaf(self, X: np.ndarray, tree: DecisionTree, leaf_idx: int) -> List[int]:
        data_indices = []
        for i, xi in enumerate(X):
            current_idx = 0
            while True:
                if current_idx == leaf_idx:
                    data_indices.append(i)
                    break
                attr = tree.attributes[current_idx]
                threshold = tree.thresholds[current_idx]
                if xi[attr] <= threshold:
                    current_idx = 2 * current_idx + 1
                else:
                    current_idx = 2 * current_idx + 2
        return data_indices

    def create_new_generation(self, attributes: List[int], possible_thresholds: Dict[int, List[float]], 
                              crossover_rate: float = 0.7, mutation_rate: float = 0.2, 
                              elitism: int = 1) -> None:
        elites = sorted(self.individuals, key=lambda x: x.fitness)[:elitism]
        
        selected = self.tournament_selection()
        
        children = []
        for i in range(0, len(selected)-1, 2):
            parent1 = selected[i]
            parent2 = selected[i+1]
            if random.random() < crossover_rate:
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
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

