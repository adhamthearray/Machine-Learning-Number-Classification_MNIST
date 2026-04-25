import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, accuracy_score

from LogisticRegresssion.LogisticRegression import LogReg, MultiLogReg


# =========================
# Helper Functions
# =========================

def preprocess(X):
    X = X.reshape(-1, 28 * 28)
    X = X / 255.0
    return X


def balance_dataset(X, y):
    ones = np.where(y == 1)[0]
    zeros = np.where(y == 0)[0]

    np.random.shuffle(zeros)
    zeros = zeros[:len(ones)]

    idx = np.concatenate([ones, zeros])
    np.random.shuffle(idx)

    return X[idx], y[idx]


# =========================
# Binary Experiments
# =========================

def run_binary_experiments(X_train, y_train, X_test, y_test):

    print("\n==============================")
    print("BINARY: 6 vs NOT 6")
    print("==============================")

    # Convert labels
    y_train_bin = (y_train == 6).astype(int)
    y_test_bin = (y_test == 6).astype(int)

    # -------------------------
    # 1. Balanced dataset
    # -------------------------
    print("\n--- Balanced Dataset ---")
    Xb, yb = balance_dataset(X_train, y_train_bin)

    model = LogReg(learning_rate=0.1, max_iterations=500)
    model.fit(Xb, yb)

    preds = model.predict(X_test)
    print(classification_report(y_test_bin, preds))
    print("Accuracy:", accuracy_score(y_test_bin, preds))


    # -------------------------
    # 2. Imbalanced dataset
    # -------------------------
    print("\n--- Imbalanced Dataset ---")

    model = LogReg(learning_rate=0.1, max_iterations=500)
    model.fit(X_train, y_train_bin)

    preds = model.predict(X_test)
    print(classification_report(y_test_bin, preds))
    print("Accuracy:", accuracy_score(y_test_bin, preds))


    # -------------------------
    # 3. Imbalanced + class_weight='balanced'
    # -------------------------
    print("\n--- Imbalanced + class_weight='balanced' ---")

    model = LogReg(learning_rate=0.1, max_iterations=500, class_weight='balanced')
    model.fit(X_train, y_train_bin)

    preds = model.predict(X_test)
    print(classification_report(y_test_bin, preds))
    print("Accuracy:", accuracy_score(y_test_bin, preds))


    # -------------------------
    # 4. Imbalanced + manual weights
    # -------------------------
    print("\n--- Imbalanced + manual class weights ---")

    model = LogReg(
        learning_rate=0.1,
        max_iterations=500,
        class_weight={0: 1.0, 1: 5.0}  # boost class 6
    )
    model.fit(X_train, y_train_bin)

    preds = model.predict(X_test)
    print(classification_report(y_test_bin, preds))
    print("Accuracy:", accuracy_score(y_test_bin, preds))


    # -------------------------
    # 5. Regularization (L2)
    # -------------------------
    print("\n--- Imbalanced + L2 Regularization ---")

    model = LogReg(
        learning_rate=0.1,
        max_iterations=500,
        reg_eqn='L2',
        reg_param=0.1
    )
    model.fit(X_train, y_train_bin)

    preds = model.predict(X_test)
    print(classification_report(y_test_bin, preds))
    print("Accuracy:", accuracy_score(y_test_bin, preds))


# =========================
# Multiclass Experiments
# =========================

def run_multiclass_experiments(X_train, y_train, X_test, y_test):

    print("\n==============================")
    print("MULTICLASS (0–9)")
    print("==============================")

    # -------------------------
    # 1. Basic multiclass
    # -------------------------
    print("\n--- Basic Multiclass ---")

    model = MultiLogReg(learning_rate=0.1, max_iterations=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))


    # -------------------------
    # 2. With L2 Regularization
    # -------------------------
    print("\n--- Multiclass + L2 ---")

    model = MultiLogReg(
        learning_rate=0.1,
        max_iterations=500,
        reg_eqn='L2',
        reg_param=0.1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))


    # -------------------------
    # 3. With class_weight='balanced'
    # -------------------------
    print("\n--- Multiclass + class_weight='balanced' ---")

    model = MultiLogReg(
        learning_rate=0.1,
        max_iterations=500,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))


# =========================
# MAIN
# =========================

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Basic preprocessing ONLY
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    run_binary_experiments(X_train, y_train, X_test, y_test)
    run_multiclass_experiments(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
