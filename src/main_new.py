import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import argparse
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from population import Population


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_splits",
    type=int,
    default=5,
    help="Choose number of split for cross-validation",
)
args = parser.parse_args()
num_splits = int(args.num_splits)


datasets = {
    "letter": "../data/letter-recognition.data",
    "weather": "../data/weather_forecast_data.csv",
    "wine": "../data/WineQT.csv",
    "mobile": "../data/mobile_price.csv",
    "cancer": "../data/Cancer_Data.csv",
}


param_dist = {
    "max_depth": [4],
    "population_size": [10],
    "elites_amount": [10],
    "p_split": [0.7],
    "crossover_rate": [0.4],
    "mutation_rate": [0.1],
    "alpha1": [0.95],
    "alpha2": [0.01],
    "num_gen": [50],
}


init_params = {
    "max_depth": 9,
    "population_size": 10,
    "elites_amount": 10,
    "p_split": 0.6939588167822919,
    "crossover_rate": 0.4158614032572565,
    "mutation_rate": 0.3468609511181633,
    "alpha1": 0.9653208422298942,
    "alpha2": 0.012685866059831859,
    "num_gen": 20,
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

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    params = init_params
    for parameter, values in param_dist.items():
        results = []
        for value in values:
            print(
                f"Changing parameter: {parameter}; Old value: {params[parameter]}; New Value: {value}"
            )
            old_value = params[parameter]
            params[parameter] = value

            tree_depth_levels = []

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

                tree_depth_levels.append(best_tree.calculate_depth())

                y_train_pred = best_tree.predict(X_train)
                y_test_pred = best_tree.predict(X_test)

                if fold_idx == 0:
                    y_test_decoded = le.inverse_transform(y_test)
                    y_test_pred_decoded = le.inverse_transform(y_test_pred)
                    cm = confusion_matrix(y_test, y_test_pred)
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt="d")
                    plt.title(
                        f"Confusion Matrix - {dataset} - {parameter} - {params[parameter]} - {old_value}"
                    )
                    plt.ylabel("True label")
                    plt.xlabel("Predicted label")
                    plt.savefig(
                        f"../cf_matrix/{dataset}_{parameter}_{params[parameter]}_{old_value}.png"
                    )
                    plt.close()

                cv_train_acc_scores.append(accuracy_score(y_train, y_train_pred))
                cv_test_acc_scores.append(accuracy_score(y_test, y_test_pred))
                cv_test_prec_scores.append(
                    precision_score(y_test, y_test_pred, average="weighted")
                )
                cv_test_rec_scores.append(
                    recall_score(y_test, y_test_pred, average="weighted")
                )
                cv_test_f1_scores.append(
                    f1_score(y_test, y_test_pred, average="weighted")
                )

                id3 = DecisionTreeClassifier(criterion="entropy")
                id3.fit(X_train, y_train)
                id3_y_train_pred = id3.predict(X_train)
                id3_y_test_pred = id3.predict(X_test)

                cv_id3_train_acc_scores.append(
                    accuracy_score(y_train, id3_y_train_pred)
                )
                cv_id3_test_acc_scores.append(accuracy_score(y_test, id3_y_test_pred))
                cv_id3_test_prec_scores.append(
                    precision_score(y_test, id3_y_test_pred, average="weighted")
                )
                cv_id3_test_rec_scores.append(
                    recall_score(y_test, id3_y_test_pred, average="weighted")
                )
                cv_id3_test_f1_scores.append(
                    f1_score(y_test, id3_y_test_pred, average="weighted")
                )

            result = {
                "old_value": old_value,
                "new_value": params[parameter],
                "params": params,
                "mean_tree_depth": np.mean(tree_depth_levels),
                "std_dev_tree_depth": np.std(tree_depth_levels),
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

        with open(f"../results/{dataset}.{parameter}.json", "a") as file_handle:
            json.dump(results, file_handle, indent=4)
