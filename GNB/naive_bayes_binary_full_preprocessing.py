import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from DT.DecisionTree import DecisionTree

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
# 2. BINARY LABELS (6 vs NOT 6)
# ======================
y_train_bin = (y_train == 6).astype(int)
y_test_bin  = (y_test == 6).astype(int)

# ======================
# 3. VARIANCE FILTER
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
# 4. BALANCE DATA (UNDERSAMPLING)
# ======================
def balance_data(X, y):
    idx_6 = np.where(y == 1)[0]
    idx_not6 = np.where(y == 0)[0]

    np.random.shuffle(idx_not6)
    idx_not6 = idx_not6[:len(idx_6)]

    idx_balanced = np.concatenate([idx_6, idx_not6])
    np.random.shuffle(idx_balanced)

    return X[idx_balanced], y[idx_balanced]

X_train_bal, y_train_bal = balance_data(X_train_clean, y_train_bin)

print("Balanced shape:", X_train_bal.shape)
print("Class counts:", np.bincount(y_train_bal))

# ======================
# 5. PCA (FULL)
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

# IMPORTANT: PCA ON BALANCED TRAINING DATA
eigenvalues, eigenvectors, mean = pca_full(X_train_bal)
k, cumulative = choose_k(eigenvalues, threshold=0.95)

print(f"\nOptimal number of components: {k}")

# ======================
# 6. PROJECT DATA
# ======================
components = eigenvectors[:, :k]

X_train_pca = (X_train_bal - mean) @ components
X_test_pca  = (X_test_clean - mean) @ components

print("Shape after PCA (train):", X_train_pca.shape)
print("Shape after PCA (test):", X_test_pca.shape)

# ======================
# 7. DECISION TREE
# ======================
dt = DecisionTree(maxDepth=8, minSamplesSplit=10)
X_train_small = X_train_bal[:5000]
y_train_small = y_train_bal[:5000]
dt.fit(X_train_small, y_train_small)

# Predict
predictions = dt.predict(X_test_pca)

# ======================
# 8. EVALUATION
# ======================
accuracy = accuracy_score(y_test_bin, predictions)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test_bin, predictions)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (1 vs Not 1)")
plt.show()

print("\nClassification Report:\n")
print(classification_report(
    y_test_bin, predictions,
    target_names=["Not 1", "Is 1"]
))

# ======================
# 9. PCA SPACE VISUALIZATION
# ======================
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
            c=y_train_bal, cmap='coolwarm', s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Space (1 vs Not 1)")
plt.colorbar()
plt.show()

# ======================
# 10. EXPLAINED VARIANCE
# ======================
plt.plot(cumulative)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("Explained Variance Curve")
plt.show()

# Optional: train accuracy on balanced train set
train_preds = dt.predict(X_train_pca)
print("Train Accuracy:", accuracy_score(y_train_bal, train_preds))

# Count classes in test set
num_6 = np.sum(y_test_bin == 1)
num_not6 = np.sum(y_test_bin == 0)

print("\nTest set distribution:")
print("Is 6:", num_6)
print("Not 6:", num_not6)

ratio = num_not6 / num_6
print("\nRatio (Not 6 : 6) =", ratio)
print("Ratio (6 : Not 6) = 1 :", ratio)