from typing import List
import numpy as np
from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self, attributes: List[int], thresholds: List[float], max_depth: int):
        self.attributes = attributes
        self.thresholds = thresholds
        self.max_depth = max_depth
        self.fitness = float('inf')
    
    def __repr__(self):
        return f"Attributes: {self.attributes}\nThresholds: {self.thresholds}\nFitness: {self.fitness:.4f}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for xi in X:
            node_idx = 0
            while True:
                attr = self.attributes[node_idx] if node_idx < len(self.attributes) else None
                if attr is None:
                    pred = self.thresholds[node_idx]
                    predictions.append(pred)
                    break
                if xi[attr] <= self.thresholds[node_idx]:
                    node_idx = 2 * node_idx + 1
                else:
                    node_idx = 2 * node_idx + 2
        return np.array(predictions)

    def calculate_depth(self) -> int:
        max_depth = 0
        for idx, attribute in enumerate(self.attributes):
            if attribute is not None:
                depth = int(np.floor(np.log2(idx + 1)))
                max_depth = max(max_depth, depth)
        return max_depth

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
