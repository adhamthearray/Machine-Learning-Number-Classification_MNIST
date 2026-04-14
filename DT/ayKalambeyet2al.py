import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_openml
from DT.DecisionTree import DecisionTree


def load_mnist_multiclass():
    mnist = fetch_openml('mnist_784', version=1)

    X = mnist.data.to_numpy(dtype=np.float32) / 255.0
    y = mnist.target.to_numpy().astype(int)   # labels 0 to 9

    # You can keep 10k for training if your tree is slow
    X_train = X[:10000]
    X_test = X[10000:12000]   # keep test smaller too, otherwise prediction may take long
    y_train = y[:10000]
    y_test = y[10000:12000]

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    dt = DecisionTree(maxDepth=8, minSamplesSplit=50)
    dt.fit(X_train, y_train)
    return dt


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(10),
            target_names=[str(i) for i in range(10)]
        )
    )


def main():
    X_train, y_train, X_test, y_test = load_mnist_multiclass()

    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"Digit {cls}: {cnt}")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()