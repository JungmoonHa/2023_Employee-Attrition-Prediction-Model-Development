# Employee Attrition Prediction Project  
Predicting employee resignation using PCA, K-Means clustering, and multiple supervised ML models.

---

## Overview  
This project predicts **employee attrition (resign=1)** using a combination of unsupervised and supervised machine learning techniques.

---

## Project Workflow  
1. Exploratory Data Analysis (EDA)  
2. PCA dimensionality reduction  
3. K-Means clustering  
4. Logistic Regression / Decision Tree / Random Forest / AdaBoost  
5. Model performance evaluation via F1 score & Precision–Recall AUC  

---

## Dataset  
- Records: 14,999  
- Features: 9 variables  
- Target: `resign` (0=stay, 1=leave)

---

## Dataset Preview
- satisfaction | evaluation | project | workhour | years | accident | resign | promotion | good

**Add screenshot here**  
`![Dataset Preview](images/dataset.png)`

---

## 1. PCA Analysis

### PCA Results
- Reduced 7 features → 3 principal components  
- Total explained variance: **57.3%**

**PCA Explained Variance Ratio**
`![PCA Variance](images/pca_variance.png)`

### PCA 3D Visualization
Points are colored by attrition status.

`![PCA 3D Scatter](images/pca_3d.png)`

---

## 2. K-Means Clustering

### Choosing K  
SSE (Elbow Method) suggests **K=5**


`![Elbow Method](images/elbow_method.png)`

### Cluster Distribution

| Cluster | Attrition Ratio |
|--------|-----------------|
| 0 | 3.7% |
| 1 | 56.6% |
| 2 | 45.6% |
| 3 | 5.9% |
| 4 | 6.5% |


`![PCA + Cluster Overlay](images/pca_cluster_overlay.png)`

---

## 3. Classification Models

### Models Trained  
- Logistic Regression  
- Decision Tree (5-fold CV tuned)  
- Random Forest (5-fold CV tuned)  
- AdaBoost (5-fold CV tuned)  

---

## 3-1. Logistic Regression

Coefficients plot:  
`![LR Coefficients](images/lr_coef.png)`

---

## 3-2. Decision Tree

Best hyperparameters:
- max_depth = 5
- min_samples_split = 300

  📷 Decision Tree plot:  
`![Decision Tree](images/decision_tree.png)`

---

## 3-3. Random Forest

Best hyperparameters:
- n_estimators = 100
- max_depth = 5
- min_samples_split = 300

Feature importance:  
`![RF Feature Importance](images/rf_importance.png)`

---

## 3-4. AdaBoost  

Best hyperparameter:
- n_estimators = 30

AdaBoost importance:  
`![AdaBoost Feature Importance](images/ada_importance.png)`

---

## 4. Model Performance Comparison

### F1 Score

| Model | F1 Score |
|-------|----------|
| Logistic Regression | 0.33 |
| Decision Tree | **0.943** |
| Random Forest | 0.941 |
| AdaBoost | 0.919 |


`![F1 Comparison](images/f1_scores.png)`

### Precision–Recall AUC

| Model | PR-AUC |
|-------|----------|
| Logistic Regression | 0.49 |
| Decision Tree | 0.966 |
| Random Forest | **0.967** |
| AdaBoost | 0.96 |


`![PR Curve](images/pr_curves.png)`

---

## Best Models

- **Decision Tree** → best F1 score  
- **Random Forest** → best PR-AUC  

### Why Random Forest performs best

- Captures non-linear relationships  
- Reduces overfitting via bagging  
- Random feature selection → de-correlated trees  
- Highly stable on HR/tabular data  

---
