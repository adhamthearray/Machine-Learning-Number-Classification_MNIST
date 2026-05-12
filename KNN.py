import numpy as np
from collections import Counter


class CustomKNN:
    """K-Nearest Neighbors classifier (binary or multi-class)."""

    def __init__(self, k=5, weights="uniform"):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if weights not in {"uniform", "distance"}:
            raise ValueError('weights must be "uniform" or "distance".')

        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """KNN is a lazy learner: training just memorizes the data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D: (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if self.k > len(X):
            raise ValueError("k cannot be larger than the number of training samples.")

        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        """Predict labels for a 2D array of samples."""
        if self.X_train is None:
            raise ValueError("KNN has not been trained. Call fit() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Expected {self.X_train.shape[1]} features, got {X.shape[1]}."
            )

        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        # 1) Euclidean distance from x to every training point.
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # 2) Indices of the k smallest distances.
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]

        # 3) Vote.
        if self.weights == "distance":
            # Closer neighbors get more vote weight. +1e-5 avoids divide-by-zero.
            k_distances = distances[k_indices]
            vote_weights = 1.0 / (k_distances + 1e-5)
            tally = {}
            for label, w in zip(k_labels, vote_weights):
                tally[label] = tally.get(label, 0.0) + w
            return max(tally, key=tally.get)

        # Uniform: simple majority vote.
        return Counter(k_labels).most_common(1)[0][0]


def find_optimal_k(X_train, y_train, X_val, y_val, max_k=15, weights="distance"):
    """Try odd k values from 1 to max_k and return the one with lowest validation error."""
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
    print(f"Optimal K = {best_k} with Error = {errors[best_idx]:.4f}")
    return best_k


def make_knn_trainer(k, weights="distance"):
    """Build a train_fn compatible with run_kfold."""
    def train_knn(X, y):
        return CustomKNN(k=k, weights=weights).fit(X, y)
    return train_knn


def predict_knn(model, X):
    """predict_fn compatible with run_kfold."""
    return model.predict(X)
