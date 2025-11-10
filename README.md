# Employee Attrition Prediction Project  

**PCA + K-Means + Logistic Regression + Decision Tree + Random Forest + AdaBoost**

---

## Overview  

This project aims to predict **employee attrition (resign=1)** using a combination of unsupervised and supervised machine-learning techniques.  

The workflow includes:

- Exploratory Data Analysis (EDA)
- Principal Component Analysis (PCA)
- K-Means clustering
- Classification modeling: Logistic Regression, Decision Tree, Random Forest, AdaBoost
- Performance comparison via **F1 score** and **Precision–Recall AUC**

**Dataset:** (14,999 rows × 9 variables)  
Variables include job satisfaction, performance evaluation, project count, work hours, tenure, accident history, promotion history, and target label (`resign`).

---

## Dataset Summary

The dataset consists of 14,999 employee records with the following features:

- `satisfaction`: Job satisfaction level (0–1 scale)
- `evaluation`: Performance evaluation score
- `project`: Number of projects completed
- `workhour`: Average monthly work hours
- `years`: Years at company
- `accident`: Whether the employee had an accident
- `promotion`: Promotion within last 5 years
- `resign`: Attrition status (0=stay, 1=leave)
- `good`: Additional indicator column (removed in preprocessing)

---

## 1. Principal Component Analysis (PCA)

### ✔ Feature Preparation
- Target variable separated (`resign`)
- 7 feature variables scaled using `StandardScaler`

### ✔ PCA Transformation
- Reduced 7 features → **3 principal components**
- Explained variance ratio:
  - PC1: 26.1%
  - PC2: 16.1%
  - PC3: 15.1%
- Total variance explained: **57.3%**

### ✔ PCA Loadings Interpretation

- **PC1**: Driven by evaluation, project count, work hour, tenure  
- **PC2**: Higher when satisfaction, project count, and work hours are low  
- **PC3**: Higher when tenure is long, promotion=1, satisfaction and evaluation are low  

---

## 2. K-Means Clustering

### ✔ SSE-based selection
Elbow method applied for K=1–9  
Chosen **K=5** based on SSE flattening.

### ✔ Cluster Assignment  
Cluster labels added to dataset as `cluster_kM`.

### ✔ Cluster-wise Attrition Ratio

| Cluster | Attrition Ratio |
|---------|-----------------|
| 0       | 3.7%            |
| 1       | 56.6%           |
| 2       | 45.6%           |
| 3       | 5.9%            |
| 4       | 6.5%            |

Clusters **1** and **2** contain high attrition groups (~50% leaving).

---

## 3. Classification Modeling

Data split: **80% train / 20% test**

### Models Trained
1. Logistic Regression  
2. Decision Tree + `GridSearchCV` (5-fold CV)  
3. Random Forest + `GridSearchCV` (5-fold CV)  
4. AdaBoost + `GridSearchCV` (5-fold CV)  

---

## 3-1. Logistic Regression

- Penalty: None  
- Requires scaled features  

Coefficient summary indicates:

- Lower satisfaction strongly associated with attrition  
- Longer tenure increases likelihood of resignation  
- More projects and higher work hours also increase resignation risk  

---

## 3-2. Decision Tree

Hyperparameters tuned via grid search:
- max_depth = 5
- min_samples_split = 300

Feature importance:

1. satisfaction (53.2%)
2. years
3. evaluation
4. project
5. workhour

---

## 3-3. Random Forest

Best hyperparameters:
- n_estimators = 100
- max_depth = 5
- min_samples_split = 300

Feature importance ranking differs slightly from Decision Tree, but **satisfaction remains the most important factor**.

---

## 3-4. AdaBoost

Uses depth-1 decision stumps  
Best hyperparameter:
- n_estimators = 30

---

## 4. Model Performance Comparison

### F1 Score

| Model               | F1 Score |
|--------------------|----------|
| Logistic Regression | 0.33     |
| Decision Tree       | **0.943** |
| Random Forest       | 0.941    |
| AdaBoost            | 0.919    |

---

### Precision–Recall AUC

| Model               | AUC       |
|--------------------|-----------|
| Logistic Regression | 0.49      |
| Decision Tree       | 0.966     |
| Random Forest       | **0.967** |
| AdaBoost            | 0.96      |

---

## Best Model & Interpretation  

- **Best F1 Score:** Decision Tree  
- **Best Precision–Recall AUC:** Random Forest  

### Why Random Forest performs best:

- Handles non-linear feature interactions well  
- Reduces variance via bootstrap aggregation  
- Introduces randomness in feature selection → de-correlated trees  
- Avoids overfitting common in single decision trees  
- Stable and robust for tabular HR datasets  
