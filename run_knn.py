import sys, os
sys.path.insert(0, r'c:\Users\basem\OneDrive\سطح المكتب\Machine_project')
os.chdir(r'c:\Users\basem\OneDrive\سطح المكتب\Machine_project')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from skimage.feature import hog
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from KNN import CustomKNN

def extract_hog_features(X):
    features = []
    for img in X:
        f = hog(img.reshape(28, 28), orientations=9, pixels_per_cell=(4,4),
                cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
        features.append(f)
    return np.array(features)

def eval_config(X_tr, y_tr, X_te, y_te, name, k, subset_size=None):
    if subset_size:
        idx = np.random.RandomState(42).choice(len(X_tr), subset_size, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    knn = CustomKNN(k=k, weights='distance')
    knn.fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    acc = accuracy_score(y_te, preds)
    report = classification_report(y_te, preds, target_names=["Not 6","Is 6"], output_dict=True)
    prec  = report["Is 6"]["precision"]
    rec   = report["Is 6"]["recall"]
    f1    = report["Is 6"]["f1-score"]
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_te, preds, target_names=["Not 6","Is 6"]))
    return prec, rec, f1, acc

# ── Load MNIST ──────────────────────────────────────────────────────────────
print("Loading MNIST...")
(X_tr_raw, y_tr_raw), (X_te_raw, y_te_raw) = keras.datasets.mnist.load_data()
y_tr_bin = (y_tr_raw == 6).astype(int)
y_te_bin = (y_te_raw == 6).astype(int)

X_tr_norm = X_tr_raw / 255.0
X_te_norm = X_te_raw / 255.0

# ── HOG ─────────────────────────────────────────────────────────────────────
print("Extracting HOG (60k train)...")
X_tr_HOG = extract_hog_features(X_tr_norm)
print("Extracting HOG (10k test)...")
X_te_HOG = extract_hog_features(X_te_norm)
print(f"HOG shape: {X_tr_HOG.shape}")

# ── HOG + PCA ────────────────────────────────────────────────────────────────
print("PCA on HOG (50 components)...")
pca = PCA(n_components=50)
X_tr_hog_pca = pca.fit_transform(X_tr_HOG)
X_te_hog_pca = pca.transform(X_te_HOG)

# ── Raw pixels & PCA-only ────────────────────────────────────────────────────
print("Preparing raw pixels & pixel PCA...")
X_flat_tr = X_tr_raw.reshape(60000, -1) / 255.0
X_flat_te = X_te_raw.reshape(10000, -1) / 255.0
pca_pix = PCA(n_components=50)
X_pca_tr = pca_pix.fit_transform(X_flat_tr)
X_pca_te = pca_pix.transform(X_flat_te)

# ── Train / Val split (same as notebook) ─────────────────────────────────────
X_tr48, X_val12, y_tr48, y_val12 = train_test_split(
    X_tr_hog_pca, y_tr_bin, test_size=0.2, stratify=y_tr_bin, random_state=42)

# ── K hyperparameter search ──────────────────────────────────────────────────
print("\n=== KNN: Finding Optimal K (validation split) ===")
print(f"{'K Value':<10} | {'Val Error':<15} | {'Accuracy':<10}")
print("-" * 40)

k_values = list(range(1, 18, 2))
errors = []
for k in k_values:
    knn = CustomKNN(k=k, weights='distance')
    knn.fit(X_tr48, y_tr48)
    preds = knn.predict(X_val12)
    acc = np.mean(preds == y_val12)
    err = 1.0 - acc
    errors.append(err)
    print(f"{k:<10} | {err:<15.4f} | {acc:<10.4f}")

best_k = k_values[np.argmin(errors)]
print("-" * 40)
print(f"Optimal K = {best_k}  (error = {min(errors):.4f})")

# ── K-Fold ────────────────────────────────────────────────────────────────────
print(f"\n=== KNN K-Fold (k={best_k}, 5-fold on 10k subset) ===")
sub_idx = np.random.RandomState(42).choice(len(X_tr48), 10000, replace=False)
X_sub, y_sub = X_tr48[sub_idx], y_tr48[sub_idx]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_f1s = []
for i, (ti, vi) in enumerate(kf.split(X_sub)):
    knn_f = CustomKNN(k=best_k, weights='distance')
    knn_f.fit(X_sub[ti], y_sub[ti])
    pf = knn_f.predict(X_sub[vi])
    s = f1_score(y_sub[vi], pf, pos_label=1)
    fold_f1s.append(s)
    print(f"Fold {i+1} -> F1: {s:.4f}")
kfold_avg = np.mean(fold_f1s)
print(f"\nK-Fold Avg F1: {kfold_avg:.4f}")

# ── Final model (HOG+PCA, full 48k train) ─────────────────────────────────────
print(f"\n=== Final KNN (k={best_k}, distance weighting, HOG+PCA) ===")
knn_final = CustomKNN(k=best_k, weights='distance')
knn_final.fit(X_tr48, y_tr48)
final_preds = knn_final.predict(X_te_hog_pca)
print(classification_report(y_te_bin, final_preds, target_names=["Not 6","Is 6"]))

# ── Feature configuration comparison ─────────────────────────────────────────
print("\n\n" + "="*55)
print("  FEATURE CONFIGURATION COMPARISON")
print("="*55)

r_raw  = eval_config(X_flat_tr,   y_tr_bin, X_flat_te,   y_te_bin, "Raw Pixels   (10k subset, k=best_k)", best_k, subset_size=10000)
r_pca  = eval_config(X_pca_tr,    y_tr_bin, X_pca_te,    y_te_bin, "PCA only     (50 comp, full 60k)",    best_k)
r_hog  = eval_config(X_tr_HOG,    y_tr_bin, X_te_HOG,    y_te_bin, "HOG only     (10k subset, k=best_k)", best_k, subset_size=10000)
r_both = eval_config(X_tr_hog_pca,y_tr_bin, X_te_hog_pca,y_te_bin, "HOG + PCA   (50 comp, full 60k)",     best_k)

print("\n\n=== SUMMARY TABLE ===")
print(f"{'Config':<20} | {'Prec(6)':<9} | {'Rec(6)':<9} | {'F1(6)':<9} | {'Accuracy':<9}")
print("-" * 65)
for name, r in [("Raw Pixels",r_raw),("PCA only",r_pca),("HOG only",r_hog),("HOG+PCA",r_both)]:
    print(f"{name:<20} | {r[0]:<9.4f} | {r[1]:<9.4f} | {r[2]:<9.4f} | {r[3]:<9.4f}")

print(f"\nbest_k = {best_k}")
print(f"kfold_avg_f1 = {kfold_avg:.4f}")
