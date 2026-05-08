import numpy as np
from collections import Counter


class CustomKNN:
    def __init__(self, k=5, weights="uniform"):
        """
        k: Number of nearest neighbors to use.
        weights: "uniform" for majority vote or "distance" for inverse-distance voting.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if weights not in {"uniform", "distance"}:
            raise ValueError('weights must be either "uniform" or "distance".')

        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        KNN is a lazy learner; training memorizes the feature matrix and labels.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if self.k > len(X):
            raise ValueError("k cannot be larger than the number of training samples.")

        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        """
        Predict labels for a 2D array of samples.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("KNN has not been trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D sample or 2D array of samples.")
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Expected {self.X_train.shape[1]} features, got {X.shape[1]}."
            )

        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argpartition(distances, self.k - 1)[: self.k]
        k_labels = self.y_train[k_indices]

        if self.weights == "distance":
            k_distances = distances[k_indices]
            zero_distance_mask = k_distances == 0
            if np.any(zero_distance_mask):
                return Counter(k_labels[zero_distance_mask]).most_common(1)[0][0]

            vote_totals = {}
            for label, distance in zip(k_labels, k_distances):
                vote_totals[label] = vote_totals.get(label, 0.0) + (1.0 / distance)
            return max(vote_totals, key=vote_totals.get)

        return Counter(k_labels).most_common(1)[0][0]


def find_optimal_k(X_train, y_train, X_val, y_val, max_k=15, weights="distance"):
    """
    Evaluate odd k values from 1 to max_k and return the k with the lowest
    validation error.
    """
    if max_k < 1:
        raise ValueError("max_k must be at least 1.")

    max_k = min(max_k, len(X_train))
    k_values = list(range(1, max_k + 1, 2))
    errors = []

    print(f"{'K Value':<10} | {'Validation Error':<20} | {'Accuracy':<10}")
    print("-" * 45)

    for k in k_values:
        knn = CustomKNN(k=k, weights=weights).fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        error = 1.0 - accuracy
        errors.append(error)

        print(f"{k:<10} | {error:<20.4f} | {accuracy:<10.4f}")

    best_idx = int(np.argmin(errors))
    best_k = k_values[best_idx]

    print("-" * 45)
    print(f"Optimal K found at K = {best_k} with Error = {errors[best_idx]:.4f}")

    return best_k


def make_knn_trainer(k, weights="distance"):
    """
    Build a train_fn compatible with run_kfold.
    """
    def train_knn(X, y):
        return CustomKNN(k=k, weights=weights).fit(X, y)

    return train_knn


def predict_knn(model, X):
    """
    predict_fn compatible with run_kfold.
    """
    return model.predict(X)
