import pandas as pd
import numpy as np
import random
import os
import json
import argparse
import csv
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from population import Population


parser = argparse.ArgumentParser()
parser.add_argument("--num_trials", type=int, default=1, help="Choose number of trials")
parser.add_argument(
    "--num_splits",
    type=int,
    default=5,
    help="Choose number of split for cross-validation",
)
args = parser.parse_args()
num_trials = int(args.num_trials)
num_splits = int(args.num_splits)

datasets = {
    "letter": "../data/letter-recognition.data",
    "weather": "../data/weather_forecast_data.csv",
    "wine": "../data/Wine_QT.csv",
    "mobile": "../data/mobile_price.csv",
    "cancer": "../data/Cancer_Data.csv",
}


param_ranges = {
    "max_depth": ("int", 2, 10),
    "population_size": ("int", 50, 500),
    "elites_amount": ("int", 1, 10),
    "p_split": ("float", 0.5, 0.8),
    "crossover_rate": ("float", 0.4, 0.8),
    "mutation_rate": ("float", 0.05, 0.5),
    "alpha1": ("float", 0.95, 1.0),
    "alpha2": ("float", 0.01, 0.2),
    "num_gen": ("int", 80, 500),
}


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


def random_search_params(param_ranges):
    params = {}
    for param_name, (param_type, min_val, max_val) in param_ranges.items():
        if param_type == "int":
            params[param_name] = random.randint(min_val, max_val)
        elif param_type == "float":
            params[param_name] = random.uniform(min_val, max_val)
    return params


for dataset, dataset_path in datasets.items():
    df = pd.read_csv(dataset_path, header=None)
    if dataset is "wine":
        df = df.drop(-1, axis=1).values
    elif dataset is "cancer":
        df = df.drop(0, axis=1).values

    le = LabelEncoder()

    if dataset is "letter" or "cancer":
        df[0] = le.fit_transform(df[0])
        X = df.drop(0, axis=1).values
        y = df[0].values
    elif dataset is "weather" or "mobile" or "wine":
        df[-1] = le.fit_transform(df[-1])
        X = df.drop(-1, axis=1).values
        y = df[-1].values

    results = []
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)


    for trial in range(num_trials):
        params = random_search_params(param_ranges)
        cv_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                possible_thresholds = calculate_possible_thresholds(X_train)
                pop = Population(
                    X=X_train,
                    y=y_train,
                    population_size=params["population_size"],
                    attributes=list(range(X_train.shape[1])),
                    possible_thresholds=possible_thresholds,
                    max_depth=params["max_depth"],
                    p_split=params["p_split"],
                )
                for gen in range(params["num_gen"]):
                    pop.create_new_generation(
                        attributes=list(range(X_train.shape[1])),
                        possible_thresholds=possible_thresholds,
                        elites_amount=params["elites_amount"],
                        crossover_rate=params["crossover_rate"],
                        mutation_rate=params["mutation_rate"]
                    )
                    pop.evaluate_population(params["alpha1"], params["alpha2"])

                best_tree = pop.get_best()
                y_pred = best_tree.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                cv_scores.append(score)

            except Exception as e:
                print(f"Error in trial {trial + 1}, fold {fold_idx + 1}: {str(e)}")

        if cv_scores:
            result = {
                "params": params,
                "cv_scores": cv_scores,
                "mean_cv_score": np.mean(cv_scores),
                "std_dev_score": np.std(cv_scores)
            }
            results.append(result)

    with open(f"../results/{dataset}.json", "a") as file_handle:
        json.dump(results, file_handle, indent=4)


filename = "hyperparameter_results.csv"
if not os.path.isfile(filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Params", "Test Accuracy"])

best_score = 0
best_params = None

# Główna pętla przeszukiwania
trial_counter = 1
while True:
    # Losowa próbka hiperparametrów
    params = random_search_params(param_ranges)
    try:
        # Trening i ewaluacja
        possible_thresholds = calculate_possible_thresholds(X_train)

        print(params)
        pop = Population(
            X=X_train,
            y=y_train,
            population_size=params["population_size"],
            attributes=list(range(16)),
            possible_thresholds=possible_thresholds,
            max_depth=params["max_depth"],
            p_split=params["p_split"],
        )

        # Przebieg treningowy
        for gen in range(params["num_gen"]):
            best = pop.get_best()
            print(f"Generation {gen+1}, Best Fitness: {best.fitness:.4f}")
            pop.create_new_generation(
                attributes=list(range(16)),
                possible_thresholds=possible_thresholds,
                elites_amount=params["elites_amount"],
                crossover_rate=params["crossover_rate"],
                mutation_rate=params["mutation_rate"],
            )
            pop.evaluate_population(params["alpha1"], params["alpha2"])

        best_tree = pop.get_best()
        y_pred = best_tree.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        if test_acc > best_score:
            best_score = test_acc
            best_params = params
            print(f"\nNowy najlepszy wynik: {best_score:.4f}")
            print("Parametry:", best_params)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([params, test_acc])

        print(f"Próba {trial_counter} | Dokładność: {test_acc:.4f}")
        trial_counter += 1

    except Exception as e:
        print(f"Błąd w próbie {trial_counter}: {str(e)}")
        continue
