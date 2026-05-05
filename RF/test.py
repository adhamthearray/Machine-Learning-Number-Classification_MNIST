import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from RF.grappaBroskiSkibiddiLewandoski import HobaTitoMambo

def test_mnist_random_forest():
    print("Loading MNIST dataset...")
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

    # 1. Subsample the dataset for custom model performance
    # Using 5,000 for training and 1,000 for testing to keep execution time reasonable
    train_samples = 60000
    test_samples = 10000

    X_train_sub = X_train_full[:train_samples]
    y_train = y_train_full[:train_samples]
    
    X_test_sub = X_test_full[:test_samples]
    y_test = y_test_full[:test_samples]

    # 2. Feature Extraction: Flatten the 28x28 images into 1D arrays of 784 features
    print("Flattening images...")
    X_train_flat = X_train_sub.reshape(X_train_sub.shape[0], -1)
    X_test_flat = X_test_sub.reshape(X_test_sub.shape[0], -1)

    # 3. Data Processing: Scaling performed strictly AFTER splitting to prevent data leakage
    print("Normalizing features...")
    X_train = X_train_flat / 255.0
    X_test = X_test_flat / 255.0

    # 4. Initialize and Train the Model
    # max_depth=None allows trees to grow until leaves are pure (or min_samples_split is hit)
    print("Initializing HobaTitoMambo Random Forest...")
    rf = HobaTitoMambo(
        n_trees=10, 
        max_depth=15, 
        max_features="sqrt", 
        random_state=42
    )

    print("Training model (this may take a few minutes)...")
    rf.fit(X_train, y_train)

    # 5. Evaluate the Model
    print("Predicting on test set...")
    predictions = rf.predict(X_test)

    # Calculate required project metrics
    accuracy = accuracy_score(y_test, predictions)
    # Using 'macro' average for multi-class F1-score
    f1 = f1_score(y_test, predictions, average='macro') 
    conf_matrix = confusion_matrix(y_test, predictions)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    test_mnist_random_forest()