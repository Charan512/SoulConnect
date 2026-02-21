"""
Train the Suicide Risk Detection Model
=======================================
Full training pipeline with:
  - Train / Test split (80/20)
  - TF-IDF vectorization
  - Logistic Regression with hyperparameter tuning (GridSearchCV)
  - 5-fold cross-validation
  - Evaluation: accuracy, precision, recall, F1, confusion matrix, classification report
  - Exports production-ready vectorizer.pkl and risk_model.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = "suicide_detection.csv"

if not os.path.exists(DATA_PATH):
    print(f"Error: '{DATA_PATH}' not found. Run generate_dataset.py first.")
    sys.exit(1)

print("=" * 60)
print("SUICIDE RISK DETECTION — MODEL TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding="latin1", on_bad_lines="skip", engine="python")
df = df[["text", "class"]].dropna()
df["class"] = df["class"].apply(lambda x: 1 if str(x).strip().lower() == "suicide" else 0)

print(f"\nDataset loaded: {len(df)} rows")
print(f"  Class distribution:")
print(f"    suicide (1):     {(df['class'] == 1).sum()}")
print(f"    non-suicide (0): {(df['class'] == 0).sum()}")
print(f"    balance ratio:   {(df['class'] == 1).mean():.2%}")


# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["class"],
    test_size=0.2,
    random_state=42,
    stratify=df["class"],
)

print(f"\nTrain/Test split (80/20):")
print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")


# ─────────────────────────────────────────────
# 3. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────
print("\nFitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),       # unigrams + bigrams for richer features
    min_df=2,                 # ignore very rare terms
    max_df=0.95,              # ignore terms in >95% of docs
    sublinear_tf=True,        # apply log normalization to TF
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"  Feature matrix:  {X_train_vec.shape}")


# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING (GridSearchCV)
# ─────────────────────────────────────────────
print("\nRunning hyperparameter search (GridSearchCV, 5-fold CV)...")

param_grid = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [2000],
}

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring="f1",           # optimize for F1 (balance precision & recall)
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train_vec, y_train)

best_model = grid.best_estimator_

print(f"  Best params: {grid.best_params_}")
print(f"  Best CV F1:  {grid.best_score_:.4f}")


# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION SCORES
# ─────────────────────────────────────────────
print("\n5-Fold Cross-Validation (on training set):")
for metric_name in ["accuracy", "precision", "recall", "f1"]:
    scores = cross_val_score(best_model, X_train_vec, y_train, cv=5, scoring=metric_name)
    print(f"  {metric_name:>10s}: {scores.mean():.4f} ± {scores.std():.4f}")


# ─────────────────────────────────────────────
# 6. TEST SET EVALUATION
# ─────────────────────────────────────────────
y_pred = best_model.predict(X_test_vec)
y_prob = best_model.predict_proba(X_test_vec)[:, 1]

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["non-suicide", "suicide"]))


# ─────────────────────────────────────────────
# 7. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("Sample Predictions on Test Data:")
sample_indices = np.random.RandomState(42).choice(len(X_test), size=min(8, len(X_test)), replace=False)
for idx in sample_indices:
    text = X_test.iloc[idx]
    true_label = y_test.iloc[idx]
    pred_prob = y_prob[idx]
    risk = "HIGH" if pred_prob > 0.75 else ("MEDIUM" if pred_prob > 0.45 else "LOW")
    status = "✓" if (pred_prob > 0.5) == true_label else "✗"
    print(f"  {status} [{risk:6s} p={pred_prob:.3f}] (true={true_label}) {text[:80]}...")


# ─────────────────────────────────────────────
# 8. EXPORT PRODUCTION MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPORTING PRODUCTION MODEL")
print("=" * 60)

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(best_model, "risk_model.pkl")

vec_size = os.path.getsize("vectorizer.pkl") / 1024
model_size = os.path.getsize("risk_model.pkl") / 1024

print(f"\n  vectorizer.pkl  ({vec_size:.1f} KB)")
print(f"  risk_model.pkl  ({model_size:.1f} KB)")
print(f"\nTraining complete. Models ready for deployment in app.py.")
