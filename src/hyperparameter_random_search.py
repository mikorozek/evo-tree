import random
import os
import datetime
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from population import Population

param_ranges = {
    "max_depth": ("int", 3, 10),
    "population_size": ("int", 100, 500),
    "elites_amount": ("int", 5, 20),
    "p_split": ("float", 0.4, 0.9),
    "crossover_rate": ("float", 0.3, 0.9),
    "mutation_rate": ("float", 0.01, 0.7),
    "alpha1": ("float", 0.8, 1.0),
    "alpha2": ("float", 0.001, 0.3),
    "num_gen": ("int", 50, 100),
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

def setup_logging(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "fitness.log")

df = pd.read_csv("../data/letter-recognition.data", header=None)
le = LabelEncoder()
df[0] = le.fit_transform(df[0])
X = df.drop(0, axis=1).values
y = df[0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
possible_thresholds = calculate_possible_thresholds(np.array(X_train))

num_random_searches = 10

for search_idx in range(num_random_searches):
    parameters = random_search_params(param_ranges)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = f"runs/run_{timestamp}_search_{search_idx}"
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        for k, v in parameters.items():
            f.write(f"{k}: {v}\n")
    
    pop = Population(
        X=np.array(X_train),
        y=np.array(y_train),
        population_size=parameters["population_size"],
        attributes=list(range(16)),
        possible_thresholds=possible_thresholds,
        max_depth=parameters["max_depth"],
        p_split=parameters["p_split"],
    )
    
    best_fitness_history = []
    best_model = None
    best_fitness = -np.inf
    
    for gen in range(parameters["num_gen"]):
        current_best = pop.get_best()
        current_fitness = current_best.fitness
        best_fitness_history.append(current_fitness)
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_model = current_best
        
        print(f"Search {search_idx+1}/{num_random_searches} | Generation {gen+1}/{parameters['num_gen']} | Best Fitness: {current_fitness:.4f}")
        with open(os.path.join(run_dir, "fitness_log.csv"), "a") as f:
            f.write(f"{gen+1},{current_fitness:.4f}\n")
        
        pop.create_new_generation(
            attributes=list(range(16)),
            possible_thresholds=possible_thresholds,
            elites_amount=parameters["elites_amount"],
            crossover_rate=parameters["crossover_rate"],
            mutation_rate=parameters["mutation_rate"],
        )
        pop.evaluate_population(parameters["alpha1"], parameters["alpha2"])
    
    with open(os.path.join(run_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    
    np.savetxt(os.path.join(run_dir, "fitness_history.csv"), 
               np.array(best_fitness_history), 
               delimiter=",", 
               header="Generation,Fitness", 
               comments="")
    
    print(f"Search {search_idx+1} completed. Best fitness: {best_fitness:.4f}")
