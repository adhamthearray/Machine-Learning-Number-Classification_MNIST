# 🧠 Linear SVM (From Scratch) for MNIST Digit "6" Detection

---

# 1. Problem Definition

## 🎯 Objective

The task is to classify handwritten digits from the MNIST dataset into:

- **Class +1:** Digit "6"
- **Class -1:** Not digit "6"

---

## 🧠 Mathematical Goal

We want to find a hyperplane that best separates the two classes:

## ⭐ Decision Rule

$$
\boxed{
\hat{y} = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)
}
$$

👉 If the score is ≥ 0 → predict **+1 (is 6)**, else predict **-1 (not 6)**.

---

# 2. Input and Output

## 📥 Input Features (X)

Each image is processed as follows:

1. Flattening (28×28 → 784)
2. Pixel normalization (values scaled 0→1 by dividing by 255)
3. Standardization (zero mean, unit variance via StandardScaler — fit on training data only)

---

### ✅ Final Representation

$$
X = [x_1, x_2, ..., x_{784}]
$$

---

## 🤖 Output (y)

$$
y =
\begin{cases}
+1 & \text{if digit is 6} \\
-1 & \text{otherwise}
\end{cases}
$$

---

# 3. Mathematical Formulation

## 🧠 SVM Objective

Minimize the regularized hinge loss:

$$
\boxed{
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i + b))
}
$$

- $\frac{1}{2} \|\mathbf{w}\|^2$ — regularization term (penalizes large weights)
- $C$ — controls trade-off between margin size and classification error
- $\max(0, 1 - y_i f(\mathbf{x}_i))$ — hinge loss (zero if correctly classified with margin)

---

## 🧠 Decision Function

$$
f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b
$$

---

## ⭐ Hinge Loss

$$
\boxed{
\mathcal{L}_{\text{hinge}} = \max(0,\ 1 - y \cdot f(\mathbf{x}))
}
$$

---

## ⚠️ SGD Gradient Update

For each mini-batch, compute gradients and update:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} =
\begin{cases}
\mathbf{w} - C \cdot y_i \mathbf{x}_i & \text{if } y_i f(\mathbf{x}_i) < 1 \\
\mathbf{w} & \text{otherwise}
\end{cases}
$$

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{w}}
\qquad
b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
$$

---

## ⚖️ Class Weight Extension

To handle class imbalance (only 9.8% are "6"):

$$
\boxed{
w_{\text{pos}} = \frac{n}{2 \cdot n_{\text{pos}}}, \qquad w_{\text{neg}} = \frac{n}{2 \cdot n_{\text{neg}}}
}
$$

Weighted hinge loss:

$$
\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} s_i \cdot \max(0, 1 - y_i f(\mathbf{x}_i))
$$

where $s_i$ is the sample weight for each example.

---

# 4. Methodology

## 📋 Pipeline

1. Preprocess images (normalize + standardize)
2. Split data into train / validation / test (60 / 20 / 20)
3. Use a dev subset (10,000 samples) for fast hyperparameter search
4. Tune C and learning rate using validation F1 score
5. Train final model on full training set with best hyperparameters
6. Evaluate on test set

---

# 5. Proposed Enhancements (Our Contribution)

This work extends standard Linear SVM by:

- ✅ **From-Scratch Implementation**  
  Full SGD-based SVM built using only NumPy — no sklearn

- ✅ **Mini-Batch SGD**  
  Faster and more stable training than full-batch gradient descent

- ✅ **Class Weighting**  
  Handles severe imbalance between "6" (9.8%) and "not 6" (90.2%)

- ✅ **Loss History Tracking**  
  Monitors training loss per epoch to visualize convergence

- ✅ **F1-score Optimization**  
  Hyperparameter search focuses on F1, not just accuracy

---

# 6. Experimental Results

## 🔥 Hyperparameter Tuning

Grid search over C and learning rate, evaluated on validation F1:

| C | Learning Rate | Validation F1 |
|---|--------------|--------------|
| 0.01 | 0.0001 | 0.5371 |
| 0.01 | 0.0005 | 0.7852 |
| 0.01 | 0.001  | 0.8354 |
| 0.1  | 0.0001 | 0.8214 |
| 0.1  | 0.0005 | 0.8631 |
| 0.1  | 0.001  | 0.8685 |
| 1.0  | 0.0001 | 0.8647 |
| **1.0**  | **0.0005** | **0.8704** |
| 1.0  | 0.001  | 0.8501 |
| 10.0 | 0.0001 | 0.8595 |
| 10.0 | 0.0005 | 0.8379 |
| 10.0 | 0.001  | 0.8137 |

🏆 Best: **C = 1.0, learning rate = 0.0005, F1 = 0.8704**

---

## 🧪 Final Test Results

| Metric | Not 6 | Is 6 |
|--------|-------|------|
| Precision | TBD | TBD |
| Recall    | TBD | TBD |
| F1-score  | TBD | TBD |

Accuracy: **TBD**

---

# 7. Results Analysis

## 🧠 Effect of C (Regularization)

- Low C (0.01) → strong regularization → underfitting at low learning rates
- C = 1.0 → best balance between margin maximization and error tolerance
- High C (10.0) → less regularization → sensitive to learning rate choice

---

## ⚖️ Effect of Learning Rate

- Too low (0.0001) → slow convergence, underfits in 50 epochs
- Too high (0.001 with C=10) → overshoots, degrades performance
- Best at 0.0005 → stable and converges well

---

## 📉 Precision vs Recall Trade-off

- Class weights push the model to recall more "6"s
- Slightly more false positives as a trade-off
- F1 score balances both

---

# 8. Model Behavior Discussion

## 🧠 Why Linear SVM Works Here

- MNIST digit pixels have strong linear separability for binary tasks
- Standardization ensures all 784 features contribute equally
- Class weighting compensates for the 9:1 class imbalance
- The learned weight vector $\mathbf{w}$ can be reshaped to 28×28 to visualize which pixels the model focuses on

---

## ⚠️ Limitations

- Linear kernel only — cannot capture non-linear digit patterns
- Raw pixels used — no feature engineering (unlike GNB which uses HOG + PCA)
- Performance depends on learning rate and C tuning
- More epochs may be needed for full convergence

---

# 9. Model Comparison

| Model | F1 Score | Accuracy | Notes |
|-------|----------|----------|-------|
| GNB | 0.96 | 0.99 | Fast, uses HOG + PCA |
| Linear SVM (Scratch) | 0.87 (val) | TBD | Raw pixels, no feature engineering |
| Decision Tree | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD |

---

# 10. Conclusion

Linear SVM from scratch achieved:

- Strong validation F1 score (~0.87)
- Stable training with mini-batch SGD
- Effective handling of class imbalance via sample weighting

Raw pixel features with standardization proved sufficient for a linear classifier, though the lack of feature engineering (HOG/PCA) likely explains the gap compared to GNB.

---

# 💬 Final Takeaways

- The decision boundary is a hyperplane in 784-dimensional space
- Hinge loss penalizes only misclassified or margin-violating points
- C controls the bias-variance trade-off
- Class weighting is essential for imbalanced binary classification
- The weight vector can be visualized as a 28×28 "mental image" of digit 6
