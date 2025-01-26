from ast import _Attributes
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

    def print_tree(self, feature_names=None, node_idx=0, prefix=""):
        if node_idx >= len(self.attributes):
            return

        attr = self.attributes[node_idx]
        threshold = self.thresholds[node_idx]
        
        if attr is None:
            print(f"{prefix}[{node_idx}] Class: {threshold}")
            return
        else:
            feat_name = feature_names[attr] if feature_names and attr < len(feature_names) else f"Feature {attr}"
            print(f"{prefix}[{node_idx}] {feat_name} <= {threshold}")
        
        left_idx = 2 * node_idx + 1
        right_idx = 2 * node_idx + 2
        
        if left_idx < len(self.attributes):
            print(f"{prefix}  ├── Left (True):")
            self.print_tree(feature_names, left_idx, prefix + "  │   ")
            
        if right_idx < len(self.attributes):
            print(f"{prefix}  └── Right (False):")
            self.print_tree(feature_names, right_idx, prefix + "      ")

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
