import numpy as np
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# ======================
# YOUR GNB (unchanged except safety fix)
# ======================
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

                # ✅ FIX: safe weight access
                if class_weights is not None:
                    score += np.log(class_weights.get(c, 1.0))

                log_prob = -((x - mean) ** 2) / (2 * var) - 0.5 * np.log(2 * np.pi * var)
                score += np.sum(log_prob)

                if score > best_score:
                    best_score = score
                    best_class = c

            predictions.append(best_class)

        return predictions


# ======================
# LOAD DATA
# ======================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# flatten
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# ======================
# 🔥 BINARY CONVERSION (6 vs not 6)
# ======================
y_train_bin = (y_train == 6).astype(int)
y_test_bin  = (y_test == 6).astype(int)


# ======================
# HOG (your original)
# ======================
from skimage.feature import hog

def extract_hog_features(X):
    features = []

    for img in X:
        img_2d = img.reshape(28, 28)

        hog_features = hog(
            img_2d,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )

        features.append(hog_features)

    return np.array(features)


# ======================
# RUN FUNCTION (UPDATED)
# ======================
def run_gnb(X_train, X_test, title):
    print(f"\n===== {title} =====")

    gnb = GNB()
    gnb.gaussian_naive_train(X_train, y_train_bin)

    y_pred = gnb.predict(X_test, class_weights={0: 1.0, 1: 10})

    print("Accuracy:", accuracy_score(y_test_bin, y_pred))
    print(classification_report(y_test_bin, y_pred,target_names=["Not 6", "Is 6"]))


# ======================
# CASE 1 — RAW
# ======================
run_gnb(X_train_flat, X_test_flat, "RAW")


# ======================
# CASE 2 — PCA ONLY
# ======================
pca = PCA(n_components=50)

X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

run_gnb(X_train_pca, X_test_pca, "PCA ONLY")


# ======================
# CASE 3 — HOG ONLY
# ======================
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

run_gnb(X_train_hog, X_test_hog, "HOG ONLY")


# ======================
# CASE 4 — PCA + HOG
# ======================
pca = PCA(n_components=50)

X_train_hog_pca = pca.fit_transform(X_train_hog)
X_test_hog_pca = pca.transform(X_test_hog)

run_gnb(X_train_hog_pca, X_test_hog_pca, "PCA + HOG")