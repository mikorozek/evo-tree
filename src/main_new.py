import pandas as pd
import numpy as np
import random
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

# Podstawowy podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dalszy podział na zbiór treningowy i walidacyjny
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Przestrzen hiperparametrów do przeszukania
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'population_size': [30, 40, 50, 60, 70],
    'elites_amount': [1, 2, 3, 4],
    'p_split': [0.6, 0.7, 0.8, 0.9],
    'crossover_rate': [0.4, 0.5, 0.6, 0.7],
    'mutation_rate': [0.1, 0.15, 0.2, 0.25],
    'alpha1': [0.95, 0.97, 0.99, 1.0],
    'alpha2': [0.01, 0.03, 0.05, 0.07],
    'num_gen': [80, 100, 120, 150]
}

num_trials = 30
best_params = None
best_score = 0
history = []

for trial in range(num_trials):
    # Losowa próbka hiperparametrów
    params = {
        'max_depth': random.choice(param_dist['max_depth']),
        'population_size': random.choice(param_dist['population_size']),
        'elites_amount': random.choice(param_dist['elites_amount']),
        'p_split': random.choice(param_dist['p_split']),
        'crossover_rate': random.choice(param_dist['crossover_rate']),
        'mutation_rate': random.choice(param_dist['mutation_rate']),
        'alpha1': random.choice(param_dist['alpha1']),
        'alpha2': random.choice(param_dist['alpha2']),
        'num_gen': random.choice(param_dist['num_gen'])
    }

    # Obliczanie progów dla aktualnego podzbioru treningowego
    possible_thresholds = calculate_possible_thresholds(X_train_sub)
    
    try:
        # Inicjalizacja populacji
        pop = Population(
            X=X_train_sub,
            y=y_train_sub,
            population_size=params['population_size'],
            attributes=list(range(16)),
            possible_thresholds=possible_thresholds,
            max_depth=params['max_depth'],
            p_split=params['p_split']
        )

        # Trening populacji
        for gen in range(params['num_gen']):
            pop.create_new_generation(
                attributes=list(range(16)),
                possible_thresholds=possible_thresholds,
                elites_amount=params['elites_amount'],
                crossover_rate=params['crossover_rate'],
                mutation_rate=params['mutation_rate']
            )
            pop.evaluate_population(params['alpha1'], params['alpha2'])

        # Ocena na zbiorze walidacyjnym
        best_tree = pop.get_best()
        y_val_pred = best_tree.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        history.append((params, val_acc))
        
        print(f"Próba {trial+1}/{num_trials} | Dokładność: {val_acc:.4f}")
        
        if val_acc > best_score:
            best_score = val_acc
            best_params = params
            print(f"Nowy najlepszy wynik: {best_score:.4f}")
            print("Parametry:", best_params)
            
    except Exception as e:
        print(f"Błąd w próbie {trial+1}: {str(e)}")
        continue

# Trening końcowy na pełnym zbiorze treningowym z najlepszymi parametrami
if best_params is not None:
    print("\nTrening końcowy z najlepszymi parametrami...")
    possible_thresholds_full = calculate_possible_thresholds(X_train)
    
    pop_final = Population(
        X=X_train,
        y=y_train,
        population_size=best_params['population_size'],
        attributes=list(range(16)),
        possible_thresholds=possible_thresholds_full,
        max_depth=best_params['max_depth'],
        p_split=best_params['p_split']
    )

    for gen in range(best_params['num_gen']):
        pop_final.create_new_generation(
            attributes=list(range(16)),
            possible_thresholds=possible_thresholds_full,
            elites_amount=best_params['elites_amount'],
            crossover_rate=best_params['crossover_rate'],
            mutation_rate=best_params['mutation_rate']
        )
        pop_final.evaluate_population(best_params['alpha1'], best_params['alpha2'])

    # Ocena na zbiorze testowym
    best_tree_final = pop_final.get_best()
    y_test_pred = best_tree_final.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("\nPodsumowanie:")
    print("Najlepsze parametry:", best_params)
    print(f"Dokładność walidacyjna: {best_score:.4f}")
    print(f"Dokładność testowa: {test_acc:.4f}")
    
    # Zapis historii do pliku
    import csv
    with open('hyperparameter_history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Params', 'Validation Accuracy'])
        for entry in history:
            writer.writerow([entry[0], entry[1]])
            
else:
    print("Nie udało się znaleźć żadnych działających parametrów.")
