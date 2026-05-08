import numpy as np
import matplotlib.pyplot as plt
def analysis_curves(X_train, y_train, X_val, y_val, train_fn, predict_fn, metric_fn, num_steps=10, title="Learning Curve"):
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    min_size = 20
    subset_sizes = np.linspace(min_size, len(X_train), num_steps, dtype=int)
    
    train_trends = []
    val_trends = []
    
    for size in subset_sizes:
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        
        model = train_fn(X_train_subset, y_train_subset)
        
        train_preds = predict_fn(model, X_train_subset)
        val_preds = predict_fn(model, X_val)
        
        train_score = metric_fn(y_train_subset, train_preds)
        val_score = metric_fn(y_val, val_preds)
        
        train_trends.append(train_score)
        val_trends.append(val_score)
        
        print(f"Size {size}/{len(X_train)} -> Train: {train_score:.4f} | Val: {val_score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes, train_trends, marker='o', label='Training Score', color='blue', linewidth=2)
    plt.plot(subset_sizes, val_trends, marker='o', label='Validation Score', color='orange', linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()

    return subset_sizes, train_trends, val_trends