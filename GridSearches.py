import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import itertools
from DT.DecisionTree import DecisionTree
def manual_grid_search_DT(X, y, param_grid, k=5):

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score = -1
    best_params = None
    results = []

    for combo in itertools.product(*values):

        params = dict(zip(keys, combo))

        print("\nTesting:", params)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        fold_scores = []

        for train_idx, val_idx in kf.split(X):

            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]

            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            model = DecisionTree(
                maxDepth=params["maxDepth"],
                minSamplesSplit=params["minSamplesSplit"],
                minSampleLeafs=params["minSampleLeafs"],
                criterion=params["criterion"],
                maxFeatures=params["maxFeatures"]
            )

            model.fit(X_train_fold, y_train_fold)

            preds = model.predict(X_val_fold)

            score = f1_score(y_val_fold, preds, average="macro")

            fold_scores.append(score)

        avg_score = sum(fold_scores) / len(fold_scores)

        print("Average score:", avg_score)

        results.append((params, avg_score))

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print("\nBest params:", best_params)
    print("Best score:", best_score)

    return best_params, best_score, results