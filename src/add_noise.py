import numpy as np

def add_noise(
    X: np.ndarray,
    y: np.ndarray,
    attribute_noise: float = 0.0,
    label_noise: float = 0.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds noise to input data with separate controls for attributes and labels.

    Parameters:
    X : np.ndarray - Feature matrix
    y : np.ndarray - Label vector
    attribute_noise : float - Noise level for attributes (0-1)
    label_noise : float - Noise level for labels (0-1)
    seed : int - Random seed (optional)

    Returns:
    tuple[np.ndarray, np.ndarray] - Noisy feature matrix and label vector
    """
    if seed is not None:
        np.random.seed(seed)

    X_noisy = X.copy()
    y_noisy = y.copy()

    if attribute_noise > 0:
        n_elements = X.size
        n_noisy = int(n_elements * attribute_noise)

        indices = np.random.choice(n_elements, n_noisy, replace=False)
        rows, cols = np.unravel_index(indices, X.shape)

        for col in np.unique(cols):
            col_mask = cols == col
            selected_rows = rows[col_mask]
            original_values = X[selected_rows, col]
            unique_values = np.unique(X[:, col])

            new_values = []
            for val in original_values:
                possible = unique_values[unique_values != val]
                new_val = np.random.choice(possible) if len(possible) > 0 else val
                new_values.append(new_val)

            X_noisy[selected_rows, col] = new_values

    if label_noise > 0:
        n_samples = y.size
        n_noisy = int(n_samples * label_noise)

        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        unique_classes = np.unique(y)

        for idx in noisy_indices:
            original = y[idx]
            possible_classes = unique_classes[unique_classes != original]
            if len(possible_classes) > 0:
                y_noisy[idx] = np.random.choice(possible_classes)

    return X_noisy, y_noisy


def calculate_actual_noise(original, noisy):
    """Calculates actual noise percentage between original and noisy data"""
    if original.ndim == 1:
        return np.mean(original != noisy)
    return np.mean(original != noisy)


def main():
    X = np.array([[0, 1], [1, 0], [0, 0]])
    y = np.array([0, 1, 0])
    print("Example 1 - Attribute noise (50%):")
    X_noisy, y_noisy = add_noise(X, y, attribute_noise=0.5, seed=10001)
    print("Original X:\n", X)
    print("Noisy X:\n", X_noisy)
    print(f"Actual attribute noise: {calculate_actual_noise(X, X_noisy)*100:.1f}%")
    print("Labels remain unchanged:", np.array_equal(y, y_noisy))

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    print("\nExample 2 - Label noise (50%):")
    _, y_noisy = add_noise(X, y, label_noise=0.5, seed=10001)
    print("Original y:", y)
    print("Noisy y:   ", y_noisy)
    print(f"Actual label noise: {calculate_actual_noise(y, y_noisy)*100:.1f}%")

    np.random.seed(10001)
    X_large = np.random.choice([0, 1, 2, 3], size=(10, 5))
    y_large = np.random.choice([0, 1], size=10)

    print("\nExample 3 - Combined noise (20% attributes + 30% labels):")
    X_noisy, y_noisy = add_noise(
        X_large, y_large, attribute_noise=0.2, label_noise=0.3, seed=10001
    )

    print("Original X (first 3 rows):\n", X_large[:3])
    print("Noisy X (first 3 rows):\n", X_noisy[:3])
    print(f"Attribute noise level: {calculate_actual_noise(X_large, X_noisy)*100:.1f}%")

    print("\nOriginal y:", y_large)
    print("Noisy y:   ", y_noisy)
    print(f"Label noise level: {calculate_actual_noise(y_large, y_noisy)*100:.1f}%")


if __name__ == "__main__":
    main()
