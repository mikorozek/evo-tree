import pandas as pd
import numpy as np
from typing import List, Dict
from population import Population
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder


def calculate_possible_thresholds(X: np.ndarray) -> Dict[int, List[float]]:
    """Oblicza możliwe progi dla każdego atrybutu."""
    possible_thresholds = {}
    for attr in range(X.shape[1]):
        unique_values = np.unique(X[:, attr])
        sorted_values = np.sort(unique_values)
        thresholds = [(sorted_values[i] + sorted_values[i+1]) / 2 for i in range(len(sorted_values) - 1)]
        possible_thresholds[attr] = thresholds
    return possible_thresholds

df = pd.read_csv('../data/letter-recognition.data', header=None)


le = LabelEncoder()
df[0] = le.fit_transform(df[0])


X = df.drop(0, axis=1).values
y = df[0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

possible_thresholds = calculate_possible_thresholds(np.array(X_train))

pop = Population(
    X=np.array(X_train),
    y=np.array(y_train),
    population_size=5,
    attributes=list(range(16)),
    possible_thresholds=possible_thresholds,
    max_depth=3
)


for gen in range(100):
    best = pop.get_best()
    print(f"Generation {gen+1}, Best Fitness: {best.fitness:.4f}")
    pop.create_new_generation(
        attributes=list(range(16)),
        possible_thresholds=possible_thresholds,
        elitism=2
    )
    pop.evaluate_population()


final_best = pop.get_best()
print("\nNajlepsze drzewo:")
print(final_best)


scores = cross_val_score(final_best, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")
