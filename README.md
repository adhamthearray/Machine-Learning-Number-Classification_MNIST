# MNIST Handwritten Digit Recognition from Scratch

A machine learning project developed for **CSE382: Introduction to Machine Learning** at **Ain Shams University (CAIE Program)**.

This project implements multiple machine learning algorithms **from scratch** and evaluates them on the MNIST handwritten digit dataset through two phases:

- Phase 1: Binary Classification ("Is this digit a 6?")
- Phase 2: Multi-Class Classification (Digits 0–9)

## Team

| Student ID | Name |
|------------|------|
| 23P0024 | Adham Walid Said Zaki |
| 23P0248 | Amir Tamer Mohamed Abdelrehim Mohamed |
| 23P0134 | Mohamed Wael El Sayed Ali El Borai |
| 23P0246 | Basem Walid Talaat Mansour Ahmed |
| 23P0049 | Moaz Ahmed Mohamed Fathy Ahmed Ahmed Younis |

---

## Live Demo

Hugging Face Deployment:

https://huggingface.co/spaces/mnistMLProject/MNISTclassification

---

## Project Overview

The MNIST dataset contains 70,000 grayscale handwritten digit images (28×28 pixels).

This project investigates:

- Feature engineering techniques
- Dimensionality reduction
- Class imbalance handling
- Model evaluation and comparison
- Binary vs Multi-Class classification performance

---

# Feature Engineering Pipeline

### 1. Image Normalization
Pixel values scaled to the range:

```python
[0,1]
```

### 2. Histogram of Oriented Gradients (HOG)

Extracts structural shape information:

- Stroke directions
- Edge orientations
- Digit contours

### 3. Principal Component Analysis (PCA)

Used to:

- Reduce dimensionality
- Remove noise
- Decorrelate features
- Improve computational efficiency

---

# Models Implemented

## Phase 1 — Binary Classification

Target:

```text
Digit 6 -> Class 1
All other digits -> Class 0
```

Implemented from scratch:

- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)
- Linear SVM

---

## Phase 2 — Multi-Class Classification

Extended classification to:

```text
0,1,2,3,4,5,6,7,8,9
```

Additional experiments:

- PCA Component Sweep
- VGG Feature Extraction
- Random Forest
- One-vs-Rest Linear SVM
- Softmax Logistic Regression

---

# Results

## Phase 1 (Binary Classification)

| Model | F1 Score | Accuracy |
|---------|---------|---------|
| KNN | **0.984** | **0.9969** |
| Linear SVM | 0.969 | 0.9939 |
| Gaussian Naive Bayes | 0.960 | 0.9918 |
| Logistic Regression | 0.954 | 0.9910 |
| Decision Tree | 0.910 | 0.9800 |

🏆 Best Model: **KNN with HOG + PCA**

---

## Phase 2 (Multi-Class Classification)

| Model | Accuracy | Macro F1 |
|---------|---------|---------|
| Linear SVM | **98%** | **0.98** |
| Logistic Regression | 97% | 0.97 |
| Random Forest | 95% | 0.95 |
| Gaussian Naive Bayes | 94.35% | 0.9435 |
| Decision Tree | 85% | 0.84 |

🏆 Best Model: **Linear SVM (OvR) using HOG Features**

---

# Key Findings

### HOG Features Consistently Perform Best

HOG captures structural information such as:

- Edge orientation
- Stroke direction
- Local shape patterns

making it highly effective for handwritten digit recognition.

### PCA Benefits Most Models

PCA:

- Removes redundancy
- Improves distance-based learning
- Helps Gaussian Naive Bayes satisfy independence assumptions

### KNN Dominates Binary Classification

Achieved:

```text
Accuracy = 99.69%
F1 Score = 0.984
```

### Linear SVM Dominates Multi-Class Classification

Achieved:

```text
Accuracy = 98%
Macro F1 = 0.98
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- OpenCV
- Hugging Face Spaces

---

# Repository Structure

```text
├── data/
├── models/
│   ├── logistic_regression/
│   ├── gaussian_naive_bayes/
│   ├── decision_tree/
│   ├── knn/
│   ├── svm/
│   └── random_forest/
├── feature_engineering/
│   ├── hog/
│   └── pca/
├── notebooks/
├── reports/
├── app.py
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/adhamthearray/Machine-Learning-Number-Classification_MNIST.git
```

Navigate to the project:

```bash
cd Machine-Learning-Number-Classification_MNIST
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

# Academic Context

This project was completed as part of:

**CSE382 — Introduction to Machine Learning**

Faculty of Engineering  
Ain Shams University — CAIE Program  
Spring 2026

---

# Authors

- Adham Walid Said Zaki
- Amir Tamer Mohamed Abdelrehim Mohamed
- Mohamed Wael El Sayed Ali El Borai
- Basem Walid Talaat Mansour Ahmed
- Moaz Ahmed Mohamed Fathy Ahmed Ahmed Younis

---

## License

This project is intended for educational and research purposes.
