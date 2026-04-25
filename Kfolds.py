def run_kfold(X, y, train_fn, predict_fn, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        model = train_fn(X_train_fold, y_train_fold)
        preds = predict_fn(model, X_val_fold)

        score = f1_score(y_val_fold, preds, pos_label=1)
        scores.append(score)

        print(f"Fold {i+1} → F1: {score:.4f}")

    avg = np.mean(scores)
    print("\nK-Fold Avg F1:", avg)

    return avg, scores
