import numpy as np

def gaussian(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def gaussian_naive_train(X, y):
    classes = np.unique(y)

    model = {}
    priors = {}

    for c in classes:
        X_c = X[y == c]  # filter rows of this class

        priors[c] = len(X_c) / len(X)

        mean = np.mean(X_c, axis=0)
        var = np.var(X_c, axis=0) + 1e-9  # avoid zero

        model[c] = (mean, var)

    return model, priors

def predict(X, model, priors):
    predictions = []

    for x in X:  # for each test image

        best_class = None
        best_score = -float("inf")  # very small number

        for c in model:  # for each class (0, 1, ...)

            mean, var = model[c]

            score = np.log(priors[c])  # start with prior



            log_prob = -((x - mean) ** 2) / (2 * var) - 0.5 * np.log(2 * np.pi * var)
            score += np.sum(log_prob)

            # keep the best class
            if score > best_score:
                best_score = score
                best_class = c

        predictions.append(best_class)

    return predictions
# Training data

model, priors = gaussian_naive_train(X_train, y_train)

predictions = predict(X_test, model, priors)

print("Predictions:", predictions)
