import pandas as pd
import numpy as np
import random
import json
import argparse
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
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
    "wine": "../data/WineQT.csv",
    "mobile": "../data/mobile_price.csv",
    "cancer": "../data/Cancer_Data.csv",
}


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


param_dist = {
    "max_depth": [8, 10, 12, 14, 16],
    "population_size": [150, 300, 500, 1000, 2000],
    "elites_amount": [10, 15, 21, 31],
    "p_split": [0.7, 0.8, 0.9],
    "crossover_rate": [0.4, 0.5, 0.6],
    "mutation_rate": [0.1, 0.15, 0.2, 0.25],
    "alpha1": [0.95, 0.97, 0.99, 1.0],
    "alpha2": [0.01, 0.03, 0.05, 0.07, 0.15, 0.3, 0.6, 0.9],
    "num_gen": [100, 300, 600, 1000],
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


init_params = {
    "max_depth": 9,
    "population_size": 494,
    "elites_amount": 10,
    "p_split": 0.6939588167822919,
    "crossover_rate": 0.4158614032572565,
    "mutation_rate": 0.3468609511181633,
    "alpha1": 0.9653208422298942,
    "alpha2": 0.012685866059831859,
    "num_gen": 294,
}

for dataset, dataset_path in datasets.items():
    print(f"Cross-Validation for {dataset}")
    le = LabelEncoder()
    if dataset == "letter":
        df = pd.read_csv(dataset_path, header=None)
        df[0] = le.fit_transform(df[0])
        X = df.drop(0, axis=1).values
        y = df[0].values
    elif dataset == "cancer":
        df = pd.read_csv(dataset_path, header=None, skiprows=1)
        df = df.drop(0, axis=1)
        df[1] = le.fit_transform(df[1])
        X = df.drop(1, axis=1).values
        y = df[1].values
    elif dataset == "wine":
        df = pd.read_csv(dataset_path, header=None, skiprows=1)
        df = df.iloc[:, :-1]
        df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif dataset == "weather":
        df = pd.read_csv(dataset_path, header=None, skiprows=1)
        df.iloc[:, -1] = df.iloc[:, -1].str.lower()
        df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
        df = df.apply(pd.to_numeric)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif dataset == "mobile":
        df = pd.read_csv(dataset_path, header=None, skiprows=1)
        df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        print("Unsupported dataset")
        exit(-1)

    results = []
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    for trial in range(num_trials):
        # params = random_search_params(param_ranges)
        cv_train_acc_scores = []
        cv_test_acc_scores = []
        cv_test_prec_scores = []
        cv_test_rec_scores = []
        cv_test_f1_scores = []

        cv_id3_train_acc_scores = []
        cv_id3_test_acc_scores = []
        cv_id3_test_prec_scores = []
        cv_id3_test_rec_scores = []
        cv_id3_test_f1_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

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

            generations_with_no_progress_count = 0
            min_fitness_score_all_generations = 1000
            for gen in range(params["num_gen"]):
                if generations_with_no_progress_count > 20:
                    break
                pop.create_new_generation(
                    attributes=list(range(X_train.shape[1])),
                    possible_thresholds=possible_thresholds,
                    elites_amount=params["elites_amount"],
                    crossover_rate=params["crossover_rate"],
                    mutation_rate=params["mutation_rate"],
                )
                min_fitness_score_this_generation = pop.evaluate_population(
                    params["alpha1"], params["alpha2"]
                )
                print(min_fitness_score_this_generation)
                generations_with_no_progress_count += 1
                if (
                    min_fitness_score_this_generation
                    < min_fitness_score_all_generations
                ):
                    min_fitness_score_all_generations = (
                        min_fitness_score_this_generation
                    )
                    generations_with_no_progress_count = 0

            best_tree = pop.get_best()

            y_train_pred = best_tree.predict(X_train)
            y_test_pred = best_tree.predict(X_test)
            cv_train_acc_scores.append(accuracy_score(y_train, y_train_pred))
            cv_test_acc_scores.append(accuracy_score(y_test, y_test_pred))
            cv_test_prec_scores.append(precision_score(y_test, y_test_pred))
            cv_test_rec_scores.append(precision_score(y_test, y_test_pred))
            cv_test_f1_scores.append(f1_score(y_test, y_test_pred))

            id3 = DecisionTreeClassifier(criterion="entropy")
            id3.fit(X_train, y_train)
            id3_y_train_pred = id3.predict(X_train)
            id3_y_test_pred = id3.predict(X_test)

            cv_id3_train_acc_scores.append(accuracy_score(y_train, id3_y_train_pred))
            cv_id3_test_acc_scores.append(accuracy_score(y_test, id3_y_test_pred))
            cv_id3_test_prec_scores.append(precision_score(y_test, id3_y_test_pred))
            cv_id3_test_rec_scores.append(precision_score(y_test, id3_y_test_pred))
            cv_id3_test_f1_scores.append(f1_score(y_test, id3_y_test_pred))

        result = {
            "params": params,
            "mean_cv_train_acc": np.mean(cv_train_acc_scores),
            "std_dev_train_acc": np.std(cv_train_acc_scores),
            "mean_cv_test_acc": np.mean(cv_test_acc_scores),
            "std_dev_test_acc": np.std(cv_test_acc_scores),
            "mean_cv_test_prec": np.mean(cv_test_prec_scores),
            "std_dev_test_prec": np.std(cv_test_prec_scores),
            "mean_cv_test_rec": np.mean(cv_test_rec_scores),
            "std_dev_test_rec": np.std(cv_test_rec_scores),
            "mean_cv_test_f1": np.mean(cv_test_f1_scores),
            "std_dev_test_f1": np.std(cv_test_f1_scores),
            "id3_mean_cv_train_acc": np.mean(cv_id3_train_acc_scores),
            "id3_std_dev_train_acc": np.std(cv_id3_train_acc_scores),
            "id3_mean_cv_test_acc": np.mean(cv_id3_test_acc_scores),
            "id3_std_dev_test_acc": np.std(cv_id3_test_acc_scores),
            "id3_mean_cv_test_prec": np.mean(cv_id3_test_prec_scores),
            "id3_std_dev_test_prec": np.std(cv_id3_test_prec_scores),
            "id3_mean_cv_test_rec": np.mean(cv_id3_test_rec_scores),
            "id3_std_dev_test_rec": np.std(cv_id3_test_rec_scores),
            "id3_mean_cv_test_f1": np.mean(cv_id3_test_f1_scores),
            "id3_std_dev_test_f1": np.std(cv_id3_test_f1_scores),
        }
        results.append(result)

    with open(f"../results/{dataset}.json", "a") as file_handle:
        json.dump(results, file_handle, indent=4)
