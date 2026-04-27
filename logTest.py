import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


from Kfolds import run_kfold
from LogisticRegresssion.LogisticRegression import LogReg

def extract_hog_features(X):
    """Extracts HOG features using the exact parameters provided."""
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

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("📥 Loading MNIST dataset via scikit-learn...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X_all = mnist.data
y_all = mnist.target.astype(int)

# Standard MNIST split: 60k train, 10k test
X_train_raw, X_test_raw = X_all[:60000], X_all[60000:]
y_train_raw, y_test_raw = y_all[:60000], y_all[60000:]

print("⚙️ Converting labels to binary (6 vs Not 6)...")
y_train = (y_train_raw == 6).astype(int)
y_test = (y_test_raw == 6).astype(int)

print("⚖️ Normalizing pixel values (0-1)...")
X_train_norm = X_train_raw / 255.0
X_test_norm = X_test_raw / 255.0

print("🧠 Extracting HOG features (pixels_per_cell=(4,4) - this will take a moment)...")
X_train_HOG = extract_hog_features(X_train_norm)
X_test_HOG = extract_hog_features(X_test_norm)

print("📉 Applying PCA to reduce dimensions to 50...")
pca = PCA(n_components=50, random_state=42)
X_train_hog_pca = pca.fit_transform(X_train_HOG)
X_test_hog_pca = pca.transform(X_test_HOG)


# ==========================================
# 2. EXPERIMENTAL CONFIGURATIONS
# ==========================================
configs = [
    {'weight': None, 'reg_eqn': None, 'reg_param': 0},
    {'weight': 'balanced', 'reg_eqn': None, 'reg_param': 0},
    {'weight': 'balanced', 'reg_eqn': 'L2', 'reg_param': 0.1},
    {'weight': {0: 1.0, 1: 5.0}, 'reg_eqn': 'L2', 'reg_param': 0.05},
    {'weight': {0: 1.0, 1: 10.0}, 'reg_eqn': 'L2', 'reg_param': 0.05}
]

# ==========================================
# 3. EXPERIMENT A: WITHOUT K-FOLD (Train/Val Split)
# ==========================================
print("\n" + "="*50)
print("🔥 EXPERIMENT A: Tuning WITHOUT K-Fold (80/20 Split)")
print("="*50)

# Create a single validation split
X_t, X_v, y_t, y_v = train_test_split(X_train_hog_pca, y_train, test_size=0.2, random_state=42)

best_f1_no_kfold = 0
best_config_no_kfold = None

for config in configs:
    print(f"\nTesting Config: Weight={config['weight']}, Reg={config['reg_eqn']}, Param={config['reg_param']}")
    
    model = LogReg(max_iterations=1000, learning_rate=0.1, 
                   class_weight=config['weight'], 
                   reg_eqn=config['reg_eqn'], 
                   reg_param=config['reg_param'])
    
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    score = f1_score(y_v, preds, pos_label=1)
    
    print(f"Validation F1 Score: {score:.4f}")
    
    if score > best_f1_no_kfold:
        best_f1_no_kfold = score
        best_config_no_kfold = config

print(f"\n👉 Best Config (No K-Fold): {best_config_no_kfold} with F1: {best_f1_no_kfold:.4f}")


# ==========================================
# 4. EXPERIMENT B: WITH K-FOLD
# ==========================================
print("\n" + "="*50)
print("🔁 EXPERIMENT B: Tuning WITH 5-Fold Cross Validation")
print("="*50)

best_f1_kfold = 0
best_config_kfold = None

for config in configs:
    print(f"\nTesting Config: Weight={config['weight']}, Reg={config['reg_eqn']}, Param={config['reg_param']}")
    
    def train_lr(X_train_fold, y_train_fold):
        model = LogReg(max_iterations=1000, learning_rate=0.1, 
                       class_weight=config['weight'], 
                       reg_eqn=config['reg_eqn'], 
                       reg_param=config['reg_param'])
        model.fit(X_train_fold, y_train_fold)
        return model

    def predict_lr(model, X_val_fold):
        return model.predict(X_val_fold)
    
    avg_f1, _ = run_kfold(X_train_hog_pca, y_train, train_lr, predict_lr, k=5)
    
    if avg_f1 > best_f1_kfold:
        best_f1_kfold = avg_f1
        best_config_kfold = config

print(f"\n👉 Best Config (K-Fold): {best_config_kfold} with Avg F1: {best_f1_kfold:.4f}")


# ==========================================
# 5. FINAL EVALUATION ON UNSEEN TEST SET
# ==========================================
print("\n" + "="*50)
print("🧪 FINAL EVALUATION ON TEST SET")
print("="*50)

# We will use the best configuration found via K-Fold (usually the most robust)
print(f"Training final model on full training set using best K-Fold config...")
final_model = LogReg(max_iterations=1000, learning_rate=0.1, 
                     class_weight=best_config_kfold['weight'], 
                     reg_eqn=best_config_kfold['reg_eqn'], 
                     reg_param=best_config_kfold['reg_param'])

final_model.fit(X_train_hog_pca, y_train)
y_pred = final_model.predict(X_test_hog_pca)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not 6", "Is 6"], digits=4))