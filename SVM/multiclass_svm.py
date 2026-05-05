import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

from linear_svm import LinearSVMScartch

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# One-vs-Rest multiclass wrapper around LinearSVMScartch
# ---------------------------------------------------------------------------

class MultiClassSVMScratch:
    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=50,
                 batch_size=64, random_state=42):
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.classifiers = {}   # maps class label -> trained LinearSVMScartch

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for k in self.classes_:
            # Relabel: +1 if sample belongs to class k, -1 otherwise
            y_binary = np.where(y == k, 1, -1)

            # use_class_weights=True handles the ~10% / ~90% OvR imbalance
            clf = LinearSVMScartch(
                C=self.C,
                learning_rate=self.learning_rate,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                use_class_weights=True,
                random_state=self.random_state,
            )
            clf.fit(X, y_binary)
            self.classifiers[k] = clf
        return self

    def decision_scores(self, X):
        # Returns (n_samples, n_classes) matrix of raw w·x+b scores
        scores = np.column_stack([
            self.classifiers[k].decision_function(X)
            for k in self.classes_
        ])
        return scores

    def predict(self, X):
        # Pick the class whose binary SVM is most confident (+highest score)
        scores = self.decision_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(n_dev=5000):
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False,
                         parser='auto', data_home='d:/sklearn_data')
    X, y = mnist.data, mnist.target.astype(int)

    # Normalize pixels to [0, 1]
    X = X / 255.0

    # 60 / 20 / 20 stratified split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval,
        random_state=RANDOM_STATE)

    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    # Standardize — fit on train only to avoid leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Small dev subset for fast hyperparameter search
    rng = np.random.RandomState(RANDOM_STATE)
    dev_idx = rng.choice(len(X_train_s), size=n_dev, replace=False)
    X_dev, y_dev = X_train_s[dev_idx], y_train[dev_idx]
    print(f"Dev subset: {len(y_dev)} samples")

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, X_dev, y_dev, scaler


# ---------------------------------------------------------------------------
# Manual grid search
# ---------------------------------------------------------------------------

def manual_grid_search(X_dev, y_dev, X_val, y_val):
    C_values  = [0.1, 1.0, 10.0]
    lr_values = [0.001, 0.0005, 0.0001]

    results = []
    print("\n--- Hyperparameter Grid Search ---")
    print(f"{'C':>6}  {'LR':>8}  {'Val Acc':>9}")
    print("-" * 30)

    for C in C_values:
        for lr in lr_values:
            model = MultiClassSVMScratch(C=C, learning_rate=lr,
                                         n_epochs=30, random_state=RANDOM_STATE)
            model.fit(X_dev, y_dev)
            acc = accuracy_score(y_val, model.predict(X_val))
            results.append((C, lr, acc))
            print(f"{C:>6}  {lr:>8}  {acc:>9.4f}")

    best = max(results, key=lambda x: x[2])
    print(f"\nBest: C={best[0]}, lr={best[1]}, Val Acc={best[2]:.4f}")
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(model):
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in model.classes_:
        ax.plot(model.classifiers[k].history, label=f"Digit {k}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss per Binary SVM (OvR)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Multiclass SVM (Scratch)")
    plt.tight_layout()
    plt.show()


def plot_decision_scores(model, X_sample, true_label):
    scores = model.decision_scores(X_sample.reshape(1, -1))[0]
    colors = ["steelblue"] * 10
    colors[true_label] = "tomato"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(model.classes_, scores, color=colors)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Decision score (w·x + b)")
    ax.set_title(f"Decision scores for one sample  (true label = {true_label})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_per_class_metrics(y_true, y_pred):
    """Bar chart + table showing accuracy, precision, recall, F1 per digit class."""
    classes = np.arange(10)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    precision = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    recall    = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1        = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    print("\n=== PER-CLASS METRICS ===")
    print(f"{'Digit':>6}  {'Accuracy':>9}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 50)
    for c in classes:
        print(f"{c:>6}  {per_class_acc[c]:>9.4f}  {precision[c]:>10.4f}  {recall[c]:>8.4f}  {f1[c]:>8.4f}")

    x = np.arange(10)
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5*width, per_class_acc, width, label="Accuracy")
    ax.bar(x - 0.5*width, precision,     width, label="Precision")
    ax.bar(x + 0.5*width, recall,         width, label="Recall")
    ax.bar(x + 1.5*width, f1,             width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Digit {c}" for c in classes], rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics — Multiclass SVM (Scratch)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def test_single_digit(model, X_test, y_test, digit=7):
    """Find one test image of `digit`, run it through the model, print prediction."""
    idx = np.where(y_test == digit)[0][0]
    X_sample = X_test[idx]
    predicted = model.predict(X_sample.reshape(1, -1))[0]
    print(f"Input: {digit}  →  Predicted: {predicted}  ({'CORRECT' if predicted == digit else 'WRONG'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test, X_dev, y_dev, scaler = load_data()

    # 2. Hyperparameter tuning
    best_C, best_lr = manual_grid_search(X_dev, y_dev, X_val, y_val)

    # 3. Train final model on full training set
    print("\nTraining final model on full training set...")
    final_model = MultiClassSVMScratch(C=best_C, learning_rate=best_lr,
                                        n_epochs=50, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)
    print("Done.")

    # 4. Evaluate on test set
    y_pred = final_model.predict(X_test)

    print("\n=== TEST SET RESULTS (Scratch SVM) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro)   : {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall    (macro)   : {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1        (macro)   : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1        (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # 6. Plots
    plot_loss_curves(final_model)
    plot_confusion_matrix(y_test, y_pred)
    plot_per_class_metrics(y_test, y_pred)

    # Decision score bar chart for one sample (pick first test sample)
    sample_idx = 0
    plot_decision_scores(final_model, X_test[sample_idx], y_test[sample_idx])

    # 7. Single image test — change digit= to any digit 0-9
    test_single_digit(final_model, X_test, y_test, digit=4)
