import numpy as np


# Load dataset
import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")


# Flatten images: (60000, 28, 28) → (60000, 784)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize (important for variance stability)
X_train = X_train / 255.0
X_test = X_test / 255.0
def compute_pixel_variance(X):
    mean = np.mean(X, axis=0)
    variance = np.mean((X - mean) ** 2, axis=0)
    return variance


def variance_threshold(X, threshold=1e-4):
    variances = compute_pixel_variance(X)
    mask = variances > threshold
    X_reduced = X[:, mask]
    return X_reduced, mask, variances
# Apply your method
X_reduced, mask, variances = variance_threshold(X_train, threshold=1e-4)

# Print results
original_features = X_train.shape[1]
remaining_features = X_reduced.shape[1]
removed_features = original_features - remaining_features

print(f"Original pixels: {original_features}")
print(f"Remaining pixels: {remaining_features}")
print(f"Removed pixels: {removed_features}")
print(f"Reduction: {removed_features/original_features * 100:.2f}%")
