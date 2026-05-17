# Heart Disease Prediction Using Random Forest Classification

---

## Overview

This project builds a machine learning pipeline to predict the presence and severity of heart disease using the **UCI Heart Disease (Cleveland) dataset**. The model classifies patients into 5 categories (0 = no disease, 1–4 = increasing severity) based on 13 clinical features.

---

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease Dataset
- **Samples:** 303 patient records
- **Features:** 13 clinical attributes (age, cholesterol, blood pressure, etc.)
- **Target:** Heart disease severity (0–4)

---

## Project Structure

```
Heart-Disease-ML-Project/
├── heart_disease_project.py      # Main ML pipeline script
├── heart.csv                     # Dataset
├── Heart_Disease_ML_Project.ipynb # Colab notebook with outputs
└── README.md                     # Project documentation
```

---

## ML Pipeline

| Step | Description |
|------|-------------|
| Data Exploration | Statistics, missing values, class distribution |
| Pre-processing | Imputation, scaling, 80/20 stratified split |
| Feature Engineering | `age_thalach_ratio`, `bp_chol_product` |
| Feature Selection | RFECV with Random Forest estimator |
| Model Comparison | Logistic Regression, Decision Tree, Random Forest, SVM |
| Hyperparameter Tuning | GridSearchCV with 5-fold cross-validation |
| Evaluation | Accuracy, F1-score, MCC, AUC-ROC, Confusion Matrix |

---

## Results

| Model | CV Accuracy |
|-------|-------------|
| Logistic Regression | 79.2% |
| Decision Tree | 76.4% |
| SVM (RBF) | 83.5% |
| **Random Forest (Tuned)** | **87.3%** |

**Final Model — Tuned Random Forest:**
- Test Accuracy: **87.3%**
- Weighted F1-Score: **0.872**
- MCC: **0.814**
- AUC-ROC (macro): **0.946**

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `heart.csv` using the Files panel on the left
3. Paste the contents of `heart_disease_project.py` into a code cell
4. Press `Shift + Enter` to run

### Option 2 — Run Locally
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python heart_disease_project.py
```

---

## Technologies Used

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## References

1. R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease," *Am. J. Cardiol.*, 1989.
2. L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, pp. 5–32, 2001.
3. S. Mohan et al., "Effective heart disease prediction using hybrid machine learning techniques," *IEEE Access*, 2019.
4. UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

---

