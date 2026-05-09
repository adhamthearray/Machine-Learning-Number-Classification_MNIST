# ================================
# 1) Setup
# ================================
import numpy as np
import cv2
import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# ================================
# 2) Load dataset
# ================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ================================
# 3) Feature extraction function
# ================================
def extract_features(model, X, batch_size=64):
    features_list = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]

        processed = []
        for img in batch:
            img = cv2.resize(img, (224, 224))
            img = np.stack([img]*3, axis=-1)  # grayscale → 3 channels
            processed.append(img)

        processed = np.array(processed)
        processed = preprocess_input(processed)

        features = model.predict(processed, batch_size=batch_size, verbose=0)
        features_list.append(features)

        if i % (batch_size * 50) == 0:
            print(f"Processed {i} samples...")

    return np.vstack(features_list)

# ================================
# 4) Save function
# ================================
def save_features(train_features, test_features):
    save_path = "/content/drive/MyDrive/cnn_features/"
    os.makedirs(save_path, exist_ok=True)

    np.savez(save_path + "vgg16_gap_train.npz", X=train_features, y=y_train)
    np.savez(save_path + "vgg16_gap_test.npz", X=test_features, y=y_test)

    print("✅ Saved vgg16_gap_train.npz and vgg16_gap_test.npz")

# ================================
# 5) Load VGG16 with GAP
# ================================
vgg_model = VGG16(
    weights='imagenet',
    include_top=False,
    pooling='avg'   # 🔥 this is the key
)

print("\n🔹 Extracting VGG16 (GAP) features...")

# ================================
# 6) Extract features
# ================================
vgg_train = extract_features(vgg_model, X_train)
vgg_test = extract_features(vgg_model, X_test)

# ================================
# 7) Save
# ================================
save_features(vgg_train, vgg_test)

print("\n🎉 DONE — VGG16 features ready!")
