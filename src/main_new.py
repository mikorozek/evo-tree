import pandas as pd
import numpy as np
import random
import os
import csv
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from population import Population

def calculate_possible_thresholds(X: np.ndarray) -> Dict[int, List[float]]:
    possible_thresholds = {}
    for attr in range(X.shape[1]):
        unique_values = np.unique(X[:, attr])
        sorted_values = np.sort(unique_values)
        thresholds = [(sorted_values[i] + sorted_values[i+1])/2 for i in range(len(sorted_values)-1)]
        possible_thresholds[attr] = thresholds
    return possible_thresholds

# Wczytanie i przygotowanie danych
df = pd.read_csv('../data/letter-recognition.data', header=None)

le = LabelEncoder()
df[0] = le.fit_transform(df[0])

X = df.drop(0, axis=1).values
y = df[0].values

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przestrzeń hiperparametrów
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'population_size': [30, 40, 50, 60, 70, 150, 300],
    'elites_amount': [1, 2, 3, 4],
    'p_split': [0.6, 0.7, 0.8, 0.9],
    'crossover_rate': [0.4, 0.5, 0.6, 0.7],
    'mutation_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.4],
    'alpha1': [0.95, 0.97, 0.99, 1.0],
    'alpha2': [0.01, 0.03, 0.05, 0.07, 0.15, 0.3, 0.6, 0.9],
    'num_gen': [80, 100, 120, 150, 500, 1000, 2000]
}

# Inicjalizacja pliku wynikowego
filename = 'hyperparameter_results.csv'
if not os.path.isfile(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Params', 'Test Accuracy'])

best_score = 0
best_params = None

# Główna pętla przeszukiwania
trial_counter = 1
while True:
    # Losowa próbka hiperparametrów
    params = {key: random.choice(values) for key, values in param_dist.items()}
    
    try:
        # Trening i ewaluacja
        possible_thresholds = calculate_possible_thresholds(X_train)
        
        pop = Population(
            X=X_train,
            y=y_train,
            population_size=params['population_size'],
            attributes=list(range(16)),
            possible_thresholds=possible_thresholds,
            max_depth=params['max_depth'],
            p_split=params['p_split']
        )

        # Przebieg treningowy
        for gen in range(params['num_gen']):
            pop.create_new_generation(
                attributes=list(range(16)),
                possible_thresholds=possible_thresholds,
                elites_amount=params['elites_amount'],
                crossover_rate=params['crossover_rate'],
                mutation_rate=params['mutation_rate']
            )
            pop.evaluate_population(params['alpha1'], params['alpha2'])

        # Ocena na zbiorze testowym
        best_tree = pop.get_best()
        y_pred = best_tree.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        # Aktualizacja najlepszego wyniku
        if test_acc > best_score:
            best_score = test_acc
            best_params = params
            print(f"\nNowy najlepszy wynik: {best_score:.4f}")
            print("Parametry:", best_params)

        # Zapis do pliku
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([params, test_acc])

        print(f"Próba {trial_counter} | Dokładność: {test_acc:.4f}")
        trial_counter += 1

    except Exception as e:
        print(f"Błąd w próbie {trial_counter}: {str(e)}")
        continue
