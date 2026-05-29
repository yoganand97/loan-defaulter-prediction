# Loan Defaulter Prediction

A machine learning project to predict whether a loan applicant will default on repayment, built as part of the **Post Graduate Diploma in Data Science & Engineering** at Great Lakes Institute, Hyderabad.

---

## Problem Statement

Financial institutions face significant risk when borrowers fail to repay loans. The goal of this project is to build a classification model that predicts whether a customer will **default on their loan repayment** (TARGET = 1) or **repay successfully** (TARGET = 0), using applicant data at the time of loan application.

---

## Dataset

- **Source:** [Loan Application Dataset — Kaggle](https://www.kaggle.com/datasets/ramakrushnamohapatra/loan-application-dataset)
- **Size:** 307,511 rows × 122 columns
- **Features:** 67 numerical + 55 categorical columns
- **Target:** Binary — `1` (defaulted) / `0` (repaid)
- **Class Imbalance:** ~90% non-defaulters vs ~10% defaulters

---

## Project Structure

```
loan-defaulter-prediction/
│
├── notebooks/
│   ├── CAPSTONE_EDA_FINAL.ipynb            # EDA on 75-column dataset
│   ├── CAPSTONE_EDA_FOR_122_COL.ipynb      # EDA on full 122-column dataset
│   ├── Modelling.ipynb                     # Baseline model experiments
│   ├── capstone_75_model.ipynb             # Model building on 75-column dataset
│   └── 122.ipynb                           # Model building on 122-column dataset
│
├── data/
│   └── df122_iterative.csv                 # Preprocessed dataset (122 columns, KNN imputed)
│
├── presentation/
│   └── Capstone_final_presentation.pptx    # Final project presentation
│
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## Approach & Methodology

### 1. Data Understanding & Splitting
The dataset contained **50 building/apartment-related columns** with ~50% missing values, tied to whether `HOUSETYPE_MODE` was null. This led to two parallel analysis paths:

| Dataset | Columns | Rows |
|---------|---------|------|
| Full dataset (HOUSETYPE not null) | 122 | ~134,000 |
| Reduced dataset (HOUSETYPE null) | 75 | ~173,000 |

### 2. Exploratory Data Analysis (EDA)
- Analyzed all 122 features individually for their relationship with the TARGET variable
- Used chi-square and t-tests (p-value thresholds) to validate feature significance
- **Flag Documents:** 20 binary columns — weighted by submission rarity; p-value ~10⁻⁹
- **External Source Scores:** 3 normalized score columns averaged into one; p-value ~10⁻⁹
- **Organization Type:** 58 categories consolidated into 15 logical groups
- **Days Employed:** Outlier treatment + cube root transformation; p-value ~10⁻²⁹⁰
- **AMT_INCOME_TOTAL:** Log transformation applied; p-value ~10⁻²³

### 3. Preprocessing
- **Categorical Encoding:** One Hot Encoding on all categorical columns
- **KNN Imputation:** Dataset split into 6 parts for KNN imputation (scalability)
- **Feature Grouping:** CNT_CHILDREN, CNT_FAM_MEMBERS, HOUR_APPR_PROCESS_START, and social circle columns grouped into meaningful bins
- **OWN_CAR_AGE:** Null values imputed with `-1` to logically represent non-car owners

### 4. Handling Class Imbalance — SMOTE
Applied **Synthetic Minority Oversampling Technique (SMOTE)** with two strategies:
- `sampling_strategy=1` → balanced 1:1 ratio
- `sampling_strategy=0.25` → 4:1 ratio (80% class 0, 20% class 1)

### 5. Feature Selection — RFE
Used **Recursive Feature Elimination (RFE)** with a Decision Tree base estimator to reduce features from 122/75 down to the **top 30 most important features**.

### 6. Model Building
Trained and evaluated **7 classification models** on both datasets (with and without RFE):

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Decision Tree | With GridSearchCV tuning |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Stage-wise additive boosting |
| XGBoost | Optimized gradient boosted trees |
| AdaBoost | Adaptive boosting |
| Stacking Classifier | DT + RF → GradientBoosting meta-model |

---

## Results

### 122-Column Dataset

| Model | SMOTE Strategy | RFE | Train F1 | Test F1 |
|-------|---------------|-----|----------|---------|
| Logistic Regression | 1:1 | No | ~0.73 | ~0.72 |
| Decision Tree | 1:1 | No | ~0.91 | ~0.82 |
| Random Forest | 1:1 | No | ~0.96 | ~0.90 |
| **Gradient Boosting** | **0.25** | **No** | **0.933** | **0.932** |
| XGBoost | 0.25 | No | ~0.92 | ~0.91 |
| AdaBoost | 0.25 | No | ~0.85 | ~0.84 |
| Stacking Classifier | 0.25 | No | ~0.94 | ~0.92 |

### 75-Column Dataset
Similar pattern observed; Gradient Boosting consistently performed best across both datasets.

### Final Model: Gradient Boosting Classifier
- **Train F1 Score: 0.933**
- **Test F1 Score: 0.932**
- Selected for its balance of performance and generalization (no overfitting)

> Models with F1 > 0.96 were available but rejected due to overfitting risk.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Statistical Tests | SciPy (chi-square, t-test) |
| Environment | Jupyter Notebook |

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yoganand97/loan-defaulter-prediction.git
cd loan-defaulter-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ramakrushnamohapatra/loan-application-dataset) and place it in the `data/` folder.

### 4. Run the notebooks in order
```
1. CAPSTONE_EDA_FINAL.ipynb          → EDA + Preprocessing (75-col path)
2. CAPSTONE_EDA_FOR_122_COL.ipynb    → EDA + Preprocessing (122-col path)
3. Modelling.ipynb                   → Baseline experiments
4. capstone_75_model.ipynb           → Full modelling on 75-col dataset
5. 122.ipynb                         → Full modelling on 122-col dataset
```

---

## License

This project is for educational purposes. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/ramakrushnamohapatra/loan-application-dataset).
