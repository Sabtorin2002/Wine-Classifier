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

# 🤖 ML Models — Training, Optimization, and Deployment

## 📘 Overview
This project focuses on **training, optimizing, and evaluating multiple machine learning models** for classifying **wine quality** using **physico-chemical properties**.  
It compares **Ridge Classifier**, **XGBoost**, and a **Multilayer Perceptron (MLP)** on the **White Wine Quality Dataset**.

The workflow includes:
- Data preprocessing and feature scaling
- Model training and evaluation
- Oversampling using **ADASYN** for imbalanced data
- Hyperparameter tuning and optimization
- Performance comparison across metrics

---

## 🍇 Dataset

- **White Wine Quality Dataset** (UCI Repository)  
- **4,899 samples**, each described by 11 physicochemical variables:
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

- **Target variable:** `quality`, mapped into three categories:
  - **Low:** 3–4  
  - **Medium:** 5–6  
  - **High:** 7–9  

- **Data Split:**  
  - 70% training  
  - 10% validation  
  - 20% testing  

---

## 🧠 Models Implemented

### 1️⃣ Ridge Classifier
Ridge Classifier was selected for its **L2 regularization** properties, reducing coefficient variance and improving model generalization.  
It’s suitable for **multiclass problems** and effectively handles **correlated features**.

#### ⚙️ Baseline Results
| Metric | Accuracy | ROC AUC | MCC |
|:--------|:----------:|:---------:|:---------:|
| RidgeClassifier (α=1.0) | 77.43% | 0.558 | 0.281 |

- **Best performance:** medium-quality wines  
- **Weakness:** failed to predict low-quality wines (recall = 0)

#### 🔧 Optimized RidgeClassifier (α=5.0, class_weight="balanced")
| Metric | Accuracy | ROC AUC | MCC |
|:--------|:----------:|:---------:|:---------:|
| RidgeClassifier (balanced) | 53.47% | — | — |

- Improved recall for minority classes but overall accuracy dropped.  
- Trade-off between balanced recall and total accuracy.

---

### 2️⃣ ADASYN Oversampling
Used **Adaptive Synthetic Sampling (ADASYN)** to balance class distribution.  
This technique generates new synthetic examples for underrepresented classes.

| Sampling Strategy | Accuracy | ROC AUC | Notes |
|:------------------|:---------:|:---------:|:------|
| `minority` | 60.28% | 0.6035 | Boosted recall for `low` class |
| `all` | 49.43% | 0.6035 | Equalized all classes but lower accuracy |

- **Clasa “low” recall** improved from 0.00 → 0.69  
- **Macro-average F1:** increased stability across classes

---

### 3️⃣ XGBoost Classifier
Implemented **XGBoost (v3.0.2)** — a gradient boosting ensemble model optimized for performance and regularization (L1/L2).

#### ⚙️ Baseline Model
| Metric | Accuracy | ROC AUC | MCC |
|:--------|:----------:|:---------:|:---------:|
| XGBoost (multi:softprob) | 77.18% | 0.3289 | 0.3246 |

- **Medium class:** F1 = 0.86  
- **Low class:** poor recall (0.06) due to imbalance  
- **Weighted Avg Accuracy:** good, but macro-average shows class disparity

#### 🔧 Optimized Model (early stopping, tuned params)
| Variant | Accuracy | Key Params |
|:---------|:----------:|:------------|
| Early Stopping + merror | 75.79% | `max_depth=7`, `learning_rate=0.1` |
| Early Stopping + mlogloss | 77.18% | `subsample=1.0`, `min_child_weight=1` |
| **Best Configuration** | **77.81%** | `max_depth=12`, `subsample=0.8`, `min_child_weight=3` |

- **Precision (high):** 0.63  
- **Recall (medium):** 0.93  
- **Macro-average F1:** 0.49 → improved generalization  
- Early stopping prevents overfitting during boosting iterations.

---

### 4️⃣ MLP (Multilayer Perceptron)
An **MLP neural network** was tested to explore non-linear decision boundaries.
- Tuned using `ReLU` activations and `Adam` optimizer.
- Compared against Ridge and XGBoost baselines.
- Performance competitive, especially with feature scaling.

---

## 📈 Evaluation Metrics

| Metric | Description |
|:--------|:-------------|
| **Accuracy** | Percentage of correctly predicted classes |
| **Precision / Recall / F1** | Per-class and macro averages |
| **ROC AUC** | Multiclass (One-vs-Rest) discriminative capability |
| **Matthews Corr. Coef. (MCC)** | Balanced measure accounting for all confusion matrix terms |

---

## ⚗️ Key Observations
- Alcohol content remains the **most predictive feature** for wine quality.  
- RidgeClassifier is interpretable but sensitive to imbalance.  
- ADASYN improves fairness across classes but can lower accuracy.  
- XGBoost consistently delivers **best accuracy (~78%)**.  
- Neural networks can reach similar results with more data tuning.

---

## 🧩 Tech Stack
- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `scikit-learn`, `imbalanced-learn`  
  - `xgboost`, `tensorflow` / `keras` (for MLP)

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/ML-Models-Training-Optimization.git
   cd ML-Models-Training-Optimization
