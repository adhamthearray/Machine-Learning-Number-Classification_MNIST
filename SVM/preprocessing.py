import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_data(n_dev=10000):
    print("Loading MNIST (this may take a minute the first time)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto',
                         data_home='d:/sklearn_data')
    X, y_digits = mnist.data, mnist.target.astype(int)

    # Binary relabeling: +1 if digit is 6, -1 otherwise
    y = np.where(y_digits == 6, 1, -1)

    print(f"Total samples: {X.shape[0]}")
    print(f"Class +1 (is 6):     {(y == 1).sum():5d}  ({(y == 1).mean()*100:.1f}%)")
    print(f"Class -1 (not 6):    {(y == -1).sum():5d}  ({(y == -1).mean()*100:.1f}%)")

    # Normalize pixels to [0, 1]
    X = X / 255.0

    # Train / Validation / Test split (60 / 20 / 20)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval,
        random_state=RANDOM_STATE)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Standardize — fit on train only to avoid data leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Dev subsample for faster training during development
    rng = np.random.RandomState(RANDOM_STATE)
    dev_idx = rng.choice(len(X_train_s), size=n_dev, replace=False)
    X_dev, y_dev = X_train_s[dev_idx], y_train[dev_idx]
    print(f"Dev set: {len(y_dev)} samples "
          f"({(y_dev == 1).sum()} sixes, {(y_dev == -1).sum()} non-sixes)")

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, X_dev, y_dev, scaler
