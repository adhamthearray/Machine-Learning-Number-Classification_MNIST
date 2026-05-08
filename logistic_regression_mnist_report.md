# Logistic Regression for MNIST Digit “6” Detection (Updated Implementation)

---

## 1. Problem Definition

### Objective
The task is to classify handwritten digits from the MNIST dataset into:

- Class 1: Digit “6”
- Class 0: Not digit “6”

---

### Mathematical Goal

We model the posterior probability:

\[
P(y=1 \mid x) = \sigma(w^T x + b)
\]

Prediction rule:

\[
\hat{y} = \begin{cases}
1 & \text{if } P(y=1 \mid x) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

---

## 2. Input and Output

### Input Features (X)

Preprocessing pipeline:

- Flattening (28×28 → 784)
- Normalization (pixel values scaled to [0,1])
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

### Linear Model

\[
z = w^T x + b
\]

### Sigmoid Function

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

---

### Loss Function (Binary Cross-Entropy)

\[
L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]
\]

---

### Class-Weighted Gradient

\[
dw = \frac{1}{\sum w_i} X^T (w_i (p - y))
\]

\[
db = \frac{1}{\sum w_i} \sum (w_i (p - y))
\]

---

### Regularization (Implementation-Accurate)

#### L2 Regularization

Loss term:

\[
L = BCE + \lambda ||w||^2
\]

Gradient:

\[
dw = dw + \frac{\lambda}{\sum w_i} w
\]

---

#### L1 Regularization

Loss term:

\[
L = BCE + \lambda ||w||_1
\]

Gradient:

\[
dw = dw + \frac{\lambda}{\sum w_i} \cdot sign(w)
\]

---

## 4. Methodology

Pipeline:

1. Load MNIST dataset
2. Convert labels (6 vs not 6)
3. Apply preprocessing (HOG + PCA)
4. Train logistic regression
5. Apply:
   - Class weighting
   - Regularization (L1 / L2)
6. Validate using:
   - Train/Validation split
   - K-Fold Cross Validation
7. Evaluate on test set

---

## 5. Implementation Enhancements

This implementation includes:

- Class weighting (balanced and manual)
- L1 and L2 regularization
- Proper gradient scaling using weighted samples
- Random initialization with fixed seed
- Numerically stable sigmoid (clipping)

---

## 6. Experimental Results

### Experiment A: Without K-Fold (80/20 Split)

| Configuration | F1 Score |
|--------------|--------|
| No weights, no regularization | 0.9600 |
| Balanced weights | 0.9304 |
| Balanced + L2 | 0.9304 |
| Manual weights (1:5) + L2 | 0.9488 |
| Manual weights (1:10) + L2 | 0.9265 |

Best configuration:

- No class weighting
- No regularization

---

### Experiment B: 5-Fold Cross Validation

| Configuration | Avg F1 |
|--------------|--------|
| No weights | 0.9592 |
| Balanced | 0.9289 |
| Balanced + L2 | 0.9289 |
| Manual (1:5) + L2 | 0.9491 |
| Manual (1:10) + L2 | 0.9245 |

Best configuration:

- No class weighting
- No regularization

---

## 7. Final Test Results

| Metric | Not 6 | Is 6 |
|------|------|------|
| Precision | 0.9931 | 0.9868 |
| Recall | 0.9987 | 0.9342 |
| F1-score | 0.9959 | 0.9598 |

Overall accuracy:

0.9925

---

## 8. Results Analysis

### Effect of Class Weights

Observed behavior:

- Using `class_weight='balanced'` reduced overall performance (F1 decreased)
- Recall for class “6” increased
- Precision for class “6” decreased significantly
- Best performance was achieved without class weighting

Explanation:

Although the dataset is imbalanced (~90% “not 6” vs ~10% “6”), this imbalance is **not severe** and the classes are **well separable** after preprocessing (HOG + PCA). As a result, logistic regression already learns a good decision boundary without needing correction.

Class weighting modifies the gradient as:

\[
w_i (p - y)
\]

which makes errors on the minority class (“6”) contribute much more to the update. This effectively tells the model that missing a “6” is very costly.

Consequences:

- The model becomes biased toward predicting class “6” more often
- False positives increase (predicting “6” when it is not)
- Precision drops while recall increases

From experiments:

- Without weights: balanced precision and recall
- With weights: recall ↑, precision ↓ → F1 decreases

This happens because F1-score depends on both precision and recall:

\[
F1 = rac{2PR}{P + R}
\]

A large drop in precision outweighs the gain in recall.

Additional insight:

- `class_weight='balanced'` often produces **aggressive weighting**
- Manual weights (e.g., {0:1, 1:5}) performed better because they introduce a **controlled bias** instead of overcompensation

Key takeaway:

Class weighting is most useful when:

- The model struggles to detect the minority class
- The imbalance is extreme (e.g., 1% vs 99%)
- Classes are not easily separable

In this problem, none of these conditions strongly apply, so class weighting leads to **overcorrection** rather than improvement.

---

### Effect of Regularization

- Minimal impact on performance
- Indicates low overfitting

---

### Precision vs Recall

- Very high precision for class “6” without weighting
- Slight drop in recall
- Weighted models increase recall but reduce precision

---

### K-Fold Stability

- Consistent performance across folds (~0.95–0.97)
- Indicates strong generalization

---

## 9. Model Behavior Discussion
 Model Behavior Discussion

### Why Logistic Regression Works Well

- Linear boundary sufficient after PCA
- HOG features provide strong signal
- Cross-entropy aligns with probabilistic modeling

---

### Limitations

- Cannot model complex non-linear patterns
- Depends heavily on preprocessing

---

## 10. Model Comparison (Template)

| Model | F1 Score | Accuracy | Notes |
|------|--------|---------|------|
| Logistic Regression | 0.96 | 0.99 | Strong baseline |
| Gaussian Naive Bayes | 0.96 | 0.99 | Fast, probabilistic |
| Decision Tree | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD |

---

## 11. Conclusion

Logistic Regression achieved:

- Accuracy: 99.25%
- F1 Score (class 6): 0.96
- Stable cross-validation performance

Key findings:

- Class weighting is not always beneficial
- Regularization had limited effect
- Feature engineering is critical

---

## Final Takeaways

- Logistic regression directly models probability
- Cross-entropy enables efficient optimization
- Regularization helps control complexity but was not critical here
- Class weighting should be applied carefully
- Proper validation ensures reliable results

