from sklearn.datasets import fetch_openml
from PIL import Image
import numpy as np
import os

NUM_SIX = 5
NUM_NOT_SIX = 5

OUTPUT_DIR = "mnist_sample_jpeg"

os.makedirs(f"{OUTPUT_DIR}/is6", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/not6", exist_ok=True)

print("Downloading MNIST dataset (one-time)...")

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

six_count = 0
not_six_count = 0

for i in range(len(X)):

    image = X[i].reshape(28, 28).astype(np.uint8)

    img = Image.fromarray(image)

    if y[i] == "6" and six_count < NUM_SIX:
        img.save(f"{OUTPUT_DIR}/is6/sample_6_{six_count}.jpg")
        six_count += 1

    elif y[i] != "6" and not_six_count < NUM_NOT_SIX:
        img.save(f"{OUTPUT_DIR}/not6/sample_not6_{not_six_count}.jpg")
        not_six_count += 1

    if six_count >= NUM_SIX and not_six_count >= NUM_NOT_SIX:
        break

print("Done. Images saved in:", OUTPUT_DIR)