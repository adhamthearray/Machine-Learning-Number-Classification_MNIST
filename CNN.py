import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model


def extract_vgg_features(X, batch_size=32):
   
    processed = []

    for img in X:
        img = cv2.resize(img, (224, 224))   # resize
        img = np.stack([img]*3, axis=-1)    # grayscale → RGB
        processed.append(img)

    processed = np.array(processed)
    processed = preprocess_input(processed)

    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    features = model.predict(processed, batch_size=batch_size)
    features = features.reshape(features.shape[0], -1)

    return features