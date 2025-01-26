import pandas as pd
import numpy as np
import random
from typing import List, Dict
from population import Population
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

param_ranges = {
    "max_depth": ("int", 2, 5),
    "population_size": ("int", 50, 100),
    "elites_amount": ("int", 1, 10),
    "p_split": ("float", 0.5, 0.8),
    "crossover_rate": ("float", 0.4, 0.8),
    "mutation_rate": ("float", 0.05, 0.5),
    "alpha1": ("float", 0.95, 1.0),
    "alpha2": ("float", 0.01, 0.2),
    "num_gen": ("int", 20, 50),
}


def random_search_params(param_ranges):
    params = {}
    for param_name, (param_type, min_val, max_val) in param_ranges.items():
        if param_type == "int":
            params[param_name] = random.randint(min_val, max_val)
        elif param_type == "float":
            params[param_name] = random.uniform(min_val, max_val)
    return params


def calculate_possible_thresholds(X: np.ndarray) -> Dict[int, List[float]]:
    possible_thresholds = {}
    for attr in range(X.shape[1]):
        unique_values = np.unique(X[:, attr])
        sorted_values = np.sort(unique_values)
        thresholds = [
            (sorted_values[i] + sorted_values[i + 1]) / 2
            for i in range(len(sorted_values) - 1)
        ]
        possible_thresholds[attr] = thresholds
    return possible_thresholds


df = pd.read_csv("../data/letter-recognition.data", header=None)


le = LabelEncoder()
df[0] = le.fit_transform(df[0])


X = df.drop(0, axis=1).values
y = df[0].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

possible_thresholds = calculate_possible_thresholds(np.array(X_train))
parameters = random_search_params(param_ranges)

pop = Population(
    X=np.array(X_train),
    y=np.array(y_train),
    population_size=parameters["population_size"],
    attributes=list(range(16)),
    possible_thresholds=possible_thresholds,
    max_depth=parameters["max_depth"],
    p_split=parameters["p_split"],
)


for gen in range(parameters["num_gen"]):
    best = pop.get_best()
    print(f"Generation {gen+1}, Best Fitness: {best.fitness:.4f}")
    pop.create_new_generation(
        attributes=list(range(16)),
        possible_thresholds=possible_thresholds,
        elites_amount=parameters["elites_amount"],
        crossover_rate=parameters["crossover_rate"],
        mutation_rate=parameters["mutation_rate"],
    )
    pop.evaluate_population(parameters["alpha1"], parameters["alpha2"])


final_best = pop.get_best()
print("\nBest tree:")
print(final_best)
