import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# KONFIGURASI MLFLOW
# ──────────────────────────────────────────────
mlflow.set_experiment("Stroke Prediction CI")

# ──────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────
DATA_PATH = "stroke_preprocessing.csv"

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

X = df.drop(columns=['stroke'])
y = df['stroke']

# ──────────────────────────────────────────────
# 2. TRAIN-TEST SPLIT
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")

# ──────────────────────────────────────────────
# 3. SMOTE
# ──────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"[INFO] Setelah SMOTE: {y_train_res.value_counts().to_dict()}")

# ──────────────────────────────────────────────
# 4. TRAINING + MANUAL LOGGING
# ──────────────────────────────────────────────
# mlflow run . sudah membuat active run secara otomatis
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_train_pred = model.predict(X_train_res)
    y_train_prob = model.predict_proba(X_train_res)[:, 1]

    # Log params
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size",    0.2)
    mlflow.log_param("smote",        True)

    # Log test metrics
    mlflow.log_metric("accuracy_score",           accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_score",          precision_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("recall_score",             recall_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("f1_score",                 f1_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("roc_auc_score",            roc_auc_score(y_test, y_prob))
    mlflow.log_metric("log_loss",                 log_loss(y_test, y_prob))

    # Log train metrics
    mlflow.log_metric("training_accuracy_score",  accuracy_score(y_train_res, y_train_pred))
    mlflow.log_metric("training_precision_score", precision_score(y_train_res, y_train_pred, zero_division=0))
    mlflow.log_metric("training_recall_score",    recall_score(y_train_res, y_train_pred, zero_division=0))
    mlflow.log_metric("training_f1_score",        f1_score(y_train_res, y_train_pred, zero_division=0))
    mlflow.log_metric("training_roc_auc_score",   roc_auc_score(y_train_res, y_train_prob))
    mlflow.log_metric("training_log_loss",        log_loss(y_train_res, y_train_prob))

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_test.iloc[:5],
        registered_model_name="stroke-prediction"
    )

    run_id = mlflow.active_run().info.run_id

    print(f"\n[INFO] Run ID    : {run_id}")
    print(f"[INFO] Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"[INFO] ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

    # Simpan run_id ke file untuk dipakai workflow
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"[INFO] run_id disimpan ke run_id.txt")

print("[INFO] Training selesai.")
