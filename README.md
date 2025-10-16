# 🍷 Exploratory Data Analysis – Wine Quality Classifier

## 📖 Overview
This project focuses on **classifying wine quality** using **physico-chemical features** from red and white wine datasets.  
The main model used is **Ridge Classifier**, with further experiments on **Random Forest**, **Gradient Boosting**, and **ADASYN oversampling** to handle class imbalance.

The goal is to predict wine quality into **three categories**:
- **Low**: 3–4  
- **Medium**: 5–6  
- **High**: 7–9  

---

## 📊 Dataset
The datasets were obtained from the **UCI Machine Learning Repository**, containing:
- **4,899 samples** of white wine  
- **1,600 samples** of red wine  

Each record includes attributes such as:
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  
- Quality (target label)

The data was split into **80% training** and **20% testing** sets.

---

## 🧠 Models and Methods

### 1. Ridge Classifier (Baseline)
The baseline model used **L2 regularization (alpha=1.0)** and standardized all features using `StandardScaler`.  
Results showed strong predictive power for medium and high classes, but struggled with the low-quality class due to imbalance.

#### Evaluation Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

---

### 2. ADASYN Oversampling
To improve prediction of the **low-quality class**, the **ADASYN algorithm (Adaptive Synthetic Sampling)** was applied.  
This technique generates new synthetic samples for underrepresented classes based on their local data distribution.

---

### 3. Hyperparameter Optimization
`GridSearchCV` was used for cross-validated hyperparameter tuning (5-fold).  
Tested models include:
- **RidgeClassifier**
- **RandomForestClassifier**
- **GradientBoostingClassifier**

Each model was optimized for parameters such as:
- **RidgeClassifier**: `alpha`, `class_weight`
- **RandomForest**: `n_estimators`, `max_depth`, `min_samples_split`
- **GradientBoosting**: `learning_rate`, `n_estimators`, `max_depth`

---

## 📈 Key Findings

### Correlation Insights
**White Wine:**
| Feature | Correlation with Quality |
|----------|--------------------------|
| Alcohol | +0.43 |
| pH | +0.10 |
| Sulphates | +0.06 |
| Density | -0.30 |
| Volatile acidity | -0.20 |
| Chlorides | -0.19 |

**Red Wine:**
| Feature | Correlation with Quality |
|----------|--------------------------|
| Alcohol | +0.48 |
| Sulphates | +0.25 |
| Citric acid | +0.23 |
| Volatile acidity | -0.40 |
| Density | -0.18 |
| Total sulfur dioxide | -0.18 |

**Most influential feature:** `Alcohol` — higher alcohol content is strongly correlated with better wine quality.

---

## 🧪 Results Summary
- **Ridge Classifier (baseline):** Solid performance for medium and high classes.
- **ADASYN + Ridge:** Improved low-class recall.
- **Random Forest:** Strong performance with reduced overfitting.
- **Gradient Boosting:** Best overall accuracy after tuning.

---

## 📦 Tech Stack
- **Language:** Python  
- **Libraries:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`
  - `imbalanced-learn` (for ADASYN)

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/ExploratoryDataAnalysis-WineClassifier.git
   cd ExploratoryDataAnalysis-WineClassifier
