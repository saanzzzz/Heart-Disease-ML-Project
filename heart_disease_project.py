"""
================================================================================
AID201A — Artificial Intelligence | Assignment
Heart Disease Prediction Using Random Forest Classification
Ramaiah University of Applied Sciences | 6th Semester, B.Tech CSE/ISE, 2023
================================================================================
Dataset : UCI Heart Disease (Cleveland) — processed.cleveland.data
          Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                   exang, oldpeak, slope, ca, thal, target
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, matthews_corrcoef, f1_score
)

import os

OUT = "output_plots"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("  AID201A — Heart Disease Prediction | ML Project")
print("=" * 70)

# ── 1. Load & Label Data ──────────────────────────────────────────────────────
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]

df_raw = pd.read_csv("heart.csv", header=None, names=COLS, na_values="?")

print(f"\n[1] Dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("\nFirst 5 rows:")
print(df_raw.head())

# ── 2. Data Exploration ───────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("[2] DATA EXPLORATION")
print("─" * 50)

print("\nData Types:\n", df_raw.dtypes)
print("\nBasic Statistics:\n", df_raw.describe())
print("\nMissing Values:\n", df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
print("\nClass Distribution:\n", df_raw["target"].value_counts().sort_index())
print("\nDuplicate rows:", df_raw.duplicated().sum())

# ── 3. Visualisations ─────────────────────────────────────────────────────────
print("\n[3] Generating EDA visualisations...")

# Fig 1: Class distribution + age distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_raw["target"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=sns.color_palette("Blues_d", 5),
    edgecolor="black")
axes[0].set_title("Class Distribution (Target Variable)", fontweight="bold")
axes[0].set_xlabel("Heart Disease Severity (0=None, 1-4=Increasing)")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
for p in axes[0].patches:
    axes[0].annotate(str(int(p.get_height())),
                     (p.get_x() + p.get_width()/2, p.get_height() + 1),
                     ha="center", fontsize=9)

df_raw["age"].plot(kind="hist", ax=axes[1], bins=20,
                   color="#4472C4", edgecolor="black")
axes[1].set_title("Age Distribution", fontweight="bold")
axes[1].set_xlabel("Age (years)")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_class_age_dist.png", dpi=150)
plt.close()

# Fig 2: Correlation heatmap (numeric only)
fig, ax = plt.subplots(figsize=(10, 8))
df_num = df_raw.copy()
df_num["ca"] = pd.to_numeric(df_num["ca"], errors="coerce")
df_num["thal"] = pd.to_numeric(df_num["thal"], errors="coerce")
corr = df_num.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_correlation_heatmap.png", dpi=150)
plt.close()

# Fig 3: Box plots for key continuous features vs target
key_feats = ["age", "thalach", "oldpeak", "chol", "trestbps"]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, feat in zip(axes, key_feats):
    df_num.boxplot(column=feat, by="target", ax=ax,
                   boxprops=dict(color="#4472C4"),
                   medianprops=dict(color="red"))
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("Target Class")
plt.suptitle("Box Plots: Key Features vs Target Class", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_boxplots.png", dpi=150)
plt.close()

print("   ✓ EDA plots saved to output_plots/")

# ── 4. Pre-processing ─────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("[4] PRE-PROCESSING")
print("─" * 50)

df = df_raw.copy()
df["ca"]   = pd.to_numeric(df["ca"],   errors="coerce")
df["thal"] = pd.to_numeric(df["thal"], errors="coerce")

X = df.drop(columns=["target"])
y = df["target"]

continuous_features  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Impute categoricals with mode, continuous with median
imputer_cat  = SimpleImputer(strategy="most_frequent")
imputer_cont = SimpleImputer(strategy="median")
scaler       = StandardScaler()

# Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Training set : {X_train.shape[0]} samples")
print(f"   Test set     : {X_test.shape[0]} samples")
print(f"   Train class dist:\n{y_train.value_counts().sort_index().to_dict()}")

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("cont", Pipeline([("imp", imputer_cont), ("scl", scaler)]),  continuous_features),
    ("cat",  Pipeline([("imp", imputer_cat)]),                    categorical_features),
])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

feat_names = continuous_features + categorical_features
print(f"\n   Preprocessed shape: {X_train_prep.shape}")

# ── 5. Feature Engineering ────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("[5] FEATURE ENGINEERING")
print("─" * 50)

def add_features(df_in):
    df_out = df_in.copy()
    df_out["age_thalach_ratio"] = df_out["age"] / (df_out["thalach"] + 1e-6)
    df_out["bp_chol_product"]   = df_out["trestbps"] * df_out["chol"]
    return df_out

X_train_fe = add_features(X_train.copy())
X_test_fe  = add_features(X_test.copy())

all_features = continuous_features + categorical_features + ["age_thalach_ratio", "bp_chol_product"]
continuous_fe = continuous_features + ["age_thalach_ratio", "bp_chol_product"]

preprocessor_fe = ColumnTransformer(transformers=[
    ("cont", Pipeline([("imp", imputer_cont), ("scl", scaler)]),  continuous_fe),
    ("cat",  Pipeline([("imp", imputer_cat)]),                    categorical_features),
])

X_train_fe_prep = preprocessor_fe.fit_transform(X_train_fe)
X_test_fe_prep  = preprocessor_fe.transform(X_test_fe)
feat_names_fe = continuous_fe + categorical_features

print(f"   Engineered features added: age_thalach_ratio, bp_chol_product")
print(f"   Total features: {len(feat_names_fe)}")

# ── 6. Feature Selection ──────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("[6] FEATURE SELECTION (RFECV)")
print("─" * 50)

rf_for_rfe = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(estimator=rf_for_rfe, step=1, cv=cv,
              scoring="accuracy", min_features_to_select=5, n_jobs=-1)
rfecv.fit(X_train_fe_prep, y_train)

selected_mask  = rfecv.support_
selected_feats = [f for f, s in zip(feat_names_fe, selected_mask) if s]
print(f"   Optimal number of features: {rfecv.n_features_}")
print(f"   Selected features: {selected_feats}")

X_train_sel = X_train_fe_prep[:, selected_mask]
X_test_sel  = X_test_fe_prep[:, selected_mask]

# Plot RFECV curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        rfecv.cv_results_["mean_test_score"], marker="o", color="#4472C4")
ax.fill_between(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
                rfecv.cv_results_["mean_test_score"] - rfecv.cv_results_["std_test_score"],
                rfecv.cv_results_["mean_test_score"] + rfecv.cv_results_["std_test_score"],
                alpha=0.2, color="#4472C4")
ax.axvline(rfecv.n_features_, color="red", linestyle="--", label=f"Optimal = {rfecv.n_features_}")
ax.set_xlabel("Number of Features")
ax.set_ylabel("CV Accuracy")
ax.set_title("RFECV: Feature Selection Curve", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_rfecv_curve.png", dpi=150)
plt.close()

# ── 7. Model Comparison ───────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("[7] MODEL COMPARISON (5-Fold Stratified CV)")
print("─" * 50)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "SVM (RBF)":           SVC(kernel="rbf", random_state=42, class_weight="balanced"),
}

cv_results = {}
print(f"\n   {'Model':<25} {'Mean CV Acc':>12} {'Std':>8}")
print("   " + "-" * 50)
for name, model in candidates.items():
    scores = cross_val_score(model, X_train_sel, y_train, cv=cv,
                              scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"   {name:<25} {scores.mean():>11.3f}  ±{scores.std():.3f}")

# Bar chart
fig, ax = plt.subplots(figsize=(9, 4))
means = [cv_results[m].mean() for m in candidates]
stds  = [cv_results[m].std()  for m in candidates]
colors = ["#A9C4E8", "#A9C4E8", "#4472C4", "#A9C4E8"]
bars = ax.bar(list(candidates.keys()), means, yerr=stds, capsize=5,
              color=colors, edgecolor="black")
ax.set_ylabel("Cross-Validation Accuracy")
ax.set_title("Model Comparison — 5-Fold CV Accuracy", fontweight="bold")
ax.set_ylim(0.5, 1.0)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_model_comparison.png", dpi=150)
plt.close()

# ── 8. Hyperparameter Tuning ──────────────────────────────────────────────────
print("\n" + "─" * 50)
print("[8] HYPERPARAMETER TUNING (GridSearchCV — Random Forest)")
print("─" * 50)

param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "max_features":    ["sqrt", "log2"],
    "class_weight":    ["balanced", None],
}

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=cv,
                           scoring="accuracy", n_jobs=-1, verbose=0)
grid_search.fit(X_train_sel, y_train)

best_params = grid_search.best_params_
best_cv_acc = grid_search.best_score_
print(f"\n   Best Parameters : {best_params}")
print(f"   Best CV Accuracy: {best_cv_acc:.4f}")

# ── 9. Final Model Training & Evaluation ──────────────────────────────────────
print("\n" + "─" * 50)
print("[9] FINAL MODEL — Training & Test Evaluation")
print("─" * 50)

final_rf = RandomForestClassifier(**best_params, random_state=42)
final_rf.fit(X_train_sel, y_train)

y_pred = final_rf.predict(X_test_sel)

acc   = accuracy_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred, average="weighted")
mcc   = matthews_corrcoef(y_test, y_pred)

print(f"\n   Test Accuracy         : {acc:.4f} ({acc*100:.1f}%)")
print(f"   Weighted F1-Score     : {f1:.4f}")
print(f"   Matthews Corr. Coeff  : {mcc:.4f}")

# AUC-ROC (One-vs-Rest)
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)
if hasattr(final_rf, "predict_proba"):
    y_prob = final_rf.predict_proba(X_test_sel)
    auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    print(f"   AUC-ROC (OvR macro)   : {auc:.4f}")

print("\n   Classification Report:\n")
print(classification_report(y_test, y_pred,
      target_names=[f"Class {c}" for c in classes]))

# ── 10. Confusion Matrix ──────────────────────────────────────────────────────
print("[10] Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=[f"Class {c}" for c in classes])
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Confusion Matrix — Tuned Random Forest (Test Set)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_confusion_matrix.png", dpi=150)
plt.close()

# ── 11. Feature Importance ────────────────────────────────────────────────────
print("[11] Generating feature importance plot...")

selected_feat_names = [feat_names_fe[i]
                       for i, s in enumerate(selected_mask) if s]
importances = final_rf.feature_importances_
idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
palette = ["#4472C4" if i == 0 else
           "#70A0D0" if i < 3 else
           "#A9C4E8" for i in range(len(selected_feat_names))]
ax.bar(range(len(idx)),
       importances[idx],
       color=[palette[r] for r in range(len(idx))],
       edgecolor="black")
ax.set_xticks(range(len(idx)))
ax.set_xticklabels([selected_feat_names[i] for i in idx],
                   rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean Decrease in Impurity (MDI)")
ax.set_title("Feature Importance — Tuned Random Forest", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_feature_importance.png", dpi=150)
plt.close()

print("\n   Top 5 features by importance:")
for rank, i in enumerate(idx[:5], 1):
    print(f"   {rank}. {selected_feat_names[i]:<22} {importances[i]:.4f}")

# ── 12. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  Dataset          : UCI Heart Disease (Cleveland), 303 samples")
print(f"  Train/Test Split : 80% / 20% (stratified)")
print(f"  Algorithm        : Random Forest (tuned)")
print(f"  Best Params      : {best_params}")
print(f"  Test Accuracy    : {acc*100:.1f}%")
print(f"  Weighted F1      : {f1:.3f}")
print(f"  MCC              : {mcc:.3f}")
if hasattr(final_rf, "predict_proba"):
    print(f"  AUC-ROC (macro)  : {auc:.3f}")
print(f"\n  All plots saved to: output_plots/")
print("=" * 70)
