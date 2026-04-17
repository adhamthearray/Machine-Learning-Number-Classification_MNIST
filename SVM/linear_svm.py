import numpy as np


class LinearSVMScartch:
    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=50,
                 batch_size=64, use_class_weights=True, random_state=42):
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_class_weights = use_class_weights
        self.random_state = random_state
        self.history = []

    def __compute_loss(self, x, y, sample_weights):
        scores = x @ self.w + self.b
        margin = y * scores
        hinge_loss = np.maximum(0, 1 - margin)
        weighted_hinge = (sample_weights * hinge_loss).mean()
        reg = 0.5 * np.sum(self.w ** 2)
        return reg + self.C * weighted_hinge

    def fit(self, X, y):
        self.history = []
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        if self.use_class_weights:
            n_pos = (y == 1).sum()
            n_neg = (y == -1).sum()
            weight_pos = n_samples / (2.0 * n_pos)
            weight_neg = n_samples / (2.0 * n_neg)
            sample_weights = np.where(y == 1, weight_pos, weight_neg)
        else:
            sample_weights = np.ones(n_samples)
        for epoch in range(self.n_epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                sw_batch = sample_weights[batch_indices]

                scores = X_batch @ self.w + self.b
                margins = y_batch * scores
                mask = margins < 1
                if mask.any():
                    wighted_y = y_batch[mask] * sw_batch[mask]
                    dw = self.w - self.C * (X_batch[mask].T @ wighted_y)
                    db = -self.C * wighted_y.sum()
                else:
                    dw = self.w.copy()
                    db = 0.0

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            epoch_loss = self.__compute_loss(X, y, sample_weights)
            self.history.append(epoch_loss)

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
