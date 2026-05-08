import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from RF.grappaBroskiSkibiddiLewandoski import HobaTitoMambo

#Download Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0


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


X_train_HOG = extract_hog_features(X_train)
X_test_HOG = extract_hog_features(X_test)

print("Initializing HobaTitoMambo Random Forest...")
rf = HobaTitoMambo(
        n_trees=20, 
        max_depth=15,
        min_samples_split=20,
        min_sample_leafs=5,
        max_features="sqrt",
        criterion="gini", 
        random_state=42
    )

print("Training model (this may take a few minutes)...")
rf.fit(X_train_HOG, y_train)

print("Predicting on test set...")
predictions = rf.predict(X_test_HOG)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("\n--- Evaluation Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
    
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions))

model_filename = 'rf_hog_model.pkl'
print(f"\nSaving model to {model_filename}...")
with open(model_filename, 'wb') as file:
    pickle.dump(rf, file)
print("Model saved successfully!")