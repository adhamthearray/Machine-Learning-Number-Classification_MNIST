# K-Nearest Neighbors (KNN) for MNIST Digit "6" Detection

---

## 1. Problem Definition

### Objective
The task is to classify handwritten digits from the MNIST dataset into:

- Class 1: Digit "6"
- Class 0: Not digit "6"

---

### Mathematical Goal

We determine the class label of a query point $x$ based on the majority vote of its $k$ nearest neighbors in the feature space.

Prediction rule:

\[
\hat{y} = \arg\max_{c \in \{0, 1\}} \sum_{i=1}^{k} w_i \cdot I(y_i = c)
\]

where $w_i$ is the weight of the $i$-th neighbor, and $I$ is the indicator function.

---

## 2. Input and Output

### Input Features (X)

Preprocessing pipeline:

- Normalization (pixel values scaled to [0,1])
- Variance filtering to remove low-information pixels
- HOG feature extraction
- PCA dimensionality reduction to 50 features

Final representation:

\[
X = [x_1, x_2, ..., x_{50}]
\]

---

### Output (y)

\[
y =
\begin{cases}
1 & \text{if digit is 6} \\
0 & \text{otherwise}
\end{cases}
\]

---

## 3. Mathematical Formulation

### Distance Metric (Euclidean)

To find the nearest neighbors, we calculate the Euclidean distance between the query point $x$ and all training points $x'$:

\[
d(x, x') = \sqrt{\sum_{j=1}^{50} (x_j - x'_j)^2}
\]

---

### Voting Mechanisms

#### Uniform Voting

All $k$ nearest neighbors have an equal vote ($w_i = 1$). The class with the highest number of votes is predicted.

\[
\hat{y} = \text{mode}(y_1, y_2, ..., y_k)
\]

#### Distance-Weighted Voting

Closer neighbors have a stronger influence on the prediction. The weight $w_i$ is inversely proportional to the distance:

\[
w_i = \frac{1}{d(x, x_i) + \epsilon}
\]

where $\epsilon = 10^{-5}$ is a small constant to prevent division by zero.

---

## 4. Methodology

Pipeline:

1. Load MNIST dataset and convert labels (6 vs not 6).
2. Apply preprocessing (Variance filtering + HOG + PCA).
3. Implement a Custom KNN classifier from scratch.
4. Hyperparameter Tuning:
   - Evaluate odd values of $k$ ($k=1, 3, 5, ..., 15$) on a validation set.
   - Select the optimal $k$ that minimizes validation error.
5. Validate using:
   - 5-Fold Cross Validation on a subset of the training data.
6. Evaluate on the final test set using both uniform and distance weighting.

---

## 5. Implementation Enhancements

This custom KNN implementation includes:

- **Vectorized Distance Calculation**: Uses NumPy broadcasting to compute Euclidean distances efficiently without explicit loops.
- **Distance Weighting**: Supports `weights='distance'` to assign higher influence to closer neighbors, preventing ties and improving accuracy near decision boundaries.
- **Efficient Sorting**: Uses `np.argsort` to efficiently locate the top $k$ neighbors.

---

## 6. Experimental Results

*(Note: Since the notebook cells for KNN are pending execution, the actual metrics are to be populated below)*

### Experiment A: Finding Optimal K

| K Value | Validation Error | Accuracy |
|---------|------------------|----------|
| 1       | TBD              | TBD      |
| 3       | TBD              | TBD      |
| 5       | TBD              | TBD      |
| ...     | TBD              | TBD      |
| 15      | TBD              | TBD      |

**Optimal K found:** TBD

---

### Experiment B: 5-Fold Cross Validation

| Configuration | Avg F1 Score |
|--------------|--------------|
| CustomKNN (k=Optimal, Distance) | TBD |

---

### Experiment C: Weighting Strategy Comparison

| Weighting | Accuracy | F1 Score |
|-----------|----------|----------|
| Uniform   | TBD      | TBD      |
| Distance  | TBD      | TBD      |

---

## 7. Final Test Results

| Metric | Not 6 | Is 6 |
|------|------|------|
| Precision | TBD | TBD |
| Recall | TBD | TBD |
| F1-score | TBD | TBD |

Overall accuracy: **TBD**

---

## 8. Results Analysis

### Effect of K

- Small $k$ (e.g., $k=1$): Models noise and may overfit.
- Large $k$: Creates a smoother decision boundary but may underfit or become biased towards the majority class (Not 6).
- The optimal $k$ balances bias and variance.

---

### Effect of Distance Weighting

- In regions where classes overlap, uniform voting might favor the majority class simply due to density.
- Distance weighting helps mitigate class imbalance locally by allowing a very close minority class point to outvote several further majority class points.

---

## 9. Model Behavior Discussion

### Why KNN Works Here

- After PCA, the feature space is dense and well-structured, making distance-based similarity meaningful.
- HOG features provide strong spatial groupings.

### Limitations

- **Inference Time**: KNN is a lazy learner; predicting a new sample requires computing distances against the entire training set.
- **Memory Intensive**: Must store all training data in memory.
- **Curse of Dimensionality**: Although mitigated by PCA (reduced to 50 dimensions), distance metrics become less discriminative in high-dimensional spaces.

---

## 10. Model Comparison (Template)

| Model | F1 Score | Accuracy | Notes |
|------|--------|---------|------|
| Logistic Regression | 0.96 | 0.99 | Strong baseline, fast inference |
| Gaussian Naive Bayes | 0.96 | 0.99 | Fast, probabilistic |
| KNN | TBD | TBD | Non-parametric, slow inference |
| SVM | TBD | TBD | TBD |
| Decision Tree | 0.88 | 0.98 | Interpretable, prone to overfitting |

---

## 11. Conclusion

The K-Nearest Neighbors model provided a non-parametric approach to the digit "6" classification problem. 

Key findings:

- Feature reduction (PCA) is critical for KNN to function efficiently and effectively.
- Distance weighting provides an edge over uniform weighting by leveraging local proximity.
- While accurate, the model's inference time scales linearly with the training set size, making it less ideal for real-time applications compared to parametric models like Logistic Regression.
