import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ======================
# 1. LOAD DATA
# ======================
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Flatten
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0


# ======================
# 2. VARIANCE FILTER
# ======================
def compute_pixel_variance(X):
    mean = np.mean(X, axis=0)
    return np.mean((X - mean) ** 2, axis=0)

def variance_threshold(X, threshold=1e-4):
    variances = compute_pixel_variance(X)
    mask = variances > threshold
    return X[:, mask], mask

X_train_clean, mask = variance_threshold(X_train)
X_test_clean = X_test[:, mask]

print("After variance filtering:", X_train_clean.shape)


# ======================
# 3. PCA (FULL)
# ======================
def pca_full(X):
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    return eigenvalues, eigenvectors, mean


def choose_k(eigenvalues, threshold=0.95):
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained_variance_ratio)

    k = np.argmax(cumulative >= threshold) + 1
    return k, cumulative


# Run PCA
eigenvalues, eigenvectors, mean = pca_full(X_train_clean)
k, cumulative = choose_k(eigenvalues, threshold=0.95)

print(f"\nOptimal number of components: {k}")


# ======================
# 4. PROJECT DATA
# ======================
components = eigenvectors[:, :k]

X_train_pca = (X_train_clean - mean) @ components
X_test_pca  = (X_test_clean - mean) @ components

print("Shape after PCA:", X_train_pca.shape)


# ======================
# 5. GAUSSIAN NAIVE BAYES
# ======================
def gaussian_naive_train(X, y):
    classes = np.unique(y)
    model = {}
    priors = {}

    for c in classes:
        X_c = X[y == c]

        priors[c] = len(X_c) / len(X)

        mean = np.mean(X_c, axis=0)
        var = np.var(X_c, axis=0) + 1e-9

        model[c] = (mean, var)

    return model, priors


def predict(X, model, priors):
    predictions = []

    for x in X:
        best_class = None
        best_score = -float("inf")

        for c in model:
            mean, var = model[c]

            log_prob = -((x - mean) ** 2) / (2 * var) - 0.5 * np.log(2 * np.pi * var)
            score = np.log(priors[c]) + np.sum(log_prob)

            if score > best_score:
                best_score = score
                best_class = c

        predictions.append(best_class)

    return np.array(predictions)


# Train on PCA data
model, priors = gaussian_naive_train(X_train_pca, y_train)

# Predict
predictions = predict(X_test_pca, model, priors)


# ======================
# 6. EVALUATION
# ======================
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)


# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# ======================
# 7. VISUALIZE PCA SPACE
# ======================
plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
            c=y_train, cmap='tab10', s=5)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("MNIST in PCA Space")
plt.colorbar()
plt.show()


# ======================
# 8. EXPLAINED VARIANCE
# ======================
plt.plot(cumulative)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("Explained Variance Curve")
plt.show()
