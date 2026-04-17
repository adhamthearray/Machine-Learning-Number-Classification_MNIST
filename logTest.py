import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  #Tried to supress some warnings but it did not but warnings are only because of loading dataset from tensorflow so not a big problem
from tensorflow.keras.datasets import mnist

from LogisticRegression import LogReg
from PCA import PCA


def balance_dataset(X, y):
    ones = np.where(y == 1)[0]
    zeros = np.where(y == 0)[0]

    np.random.shuffle(zeros)

    n = len(ones)
    zeros = zeros[:n]

    idx = np.concatenate([ones, zeros])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def main():
    # 🔴 1. Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 🔴 2. Convert labels (6 vs not 6)
    y_train = (y_train == 6).astype(int)
    y_test = (y_test == 6).astype(int)

    # 🔴 3. Balance dataset
    X_train, y_train = balance_dataset(X_train, y_train)
    X_test, y_test = balance_dataset(X_test, y_test)

    # 🔴 4. Flatten images
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # 🔴 5. Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 🔴 6. Apply PCA
    pca = PCA(n_components=50)
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # 🔴 7. Train Logistic Regression
    model = LogReg(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)

    # 🔴 8. Predict
    preds = model.predict(X_test)

    # 🔴 9. Accuracy
    accuracy = np.mean(preds == y_test)
    print("Test Accuracy:", accuracy)


if __name__ == "__main__":
    main()
