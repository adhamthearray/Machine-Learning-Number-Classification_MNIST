import numpy as np
from tensorflow import keras
class GNB:

    def gaussian_naive_train(self, X, y):
        classes = np.unique(y)

        model = {}
        priors = {}

        for c in classes:
            X_c = X[y == c]
            priors[c] = len(X_c) / len(X)

            mean = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0) + 1e-9

            model[c] = (mean, var)

        self.model = model
        self.priors = priors

    def predict(self, X, class_weights=None):
        predictions = []

        for x in X:
            best_class = None
            best_score = -float("inf")

            for c in self.model:
                mean, var = self.model[c]

                score = np.log(self.priors[c])

                if class_weights is not None:
                    score += np.log(class_weights[c])

                log_prob = -((x - mean) ** 2) / (2 * var) - 0.5 * np.log(2 * np.pi * var)
                score += np.sum(log_prob)

                if score > best_score:
                    best_score = score
                    best_class = c

            predictions.append(best_class)

        return predictions
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()