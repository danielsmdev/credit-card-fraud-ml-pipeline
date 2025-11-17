from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.config.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    MLFLOW_DIR,
    FINAL_MODEL_NAME,
)
from src.utils.io_utils import load_csv

TARGET_COL = "Class"


def load_train_test():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    df_train = load_csv(train_path)
    df_test = load_csv(test_path)

    if TARGET_COL not in df_train.columns or TARGET_COL not in df_test.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no existe en train/test.")

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_train_test()

    # Modelo base serio, no juguete
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    # Configurar MLflow para usar el directorio local
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("fraud_detection_rf")

    with mlflow.start_run():
        # Log de hiperparámetros
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("class_weight", "balanced")

        # Entrenamiento
        model.fit(X_train, y_train)

        # Predicciones probabilidad
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Umbral fijo 0.5 (luego puedes tunearlo)
        y_train_pred = (y_train_proba > 0.5).astype(int)
        y_test_pred = (y_test_proba > 0.5).astype(int)

        # Métricas train
        train_auc = roc_auc_score(y_train, y_train_proba)
        train_precision = precision_score(y_train, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

        # Métricas test
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        # Log de métricas en MLflow
        mlflow.log_metric("train_auc", float(train_auc))
        mlflow.log_metric("train_precision", float(train_precision))
        mlflow.log_metric("train_recall", float(train_recall))
        mlflow.log_metric("train_f1", float(train_f1))

        mlflow.log_metric("test_auc", float(test_auc))
        mlflow.log_metric("test_precision", float(test_precision))
        mlflow.log_metric("test_recall", float(test_recall))
        mlflow.log_metric("test_f1", float(test_f1))

        # Log del modelo a MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Guardar modelo final para la API
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / FINAL_MODEL_NAME
        joblib.dump(model, model_path)

        print("=== Métricas TRAIN ===")
        print(f"AUC:       {train_auc:.4f}")
        print(f"Precision: {train_precision:.4f}")
        print(f"Recall:    {train_recall:.4f}")
        print(f"F1:        {train_f1:.4f}")

        print("\n=== Métricas TEST ===")
        print(f"AUC:       {test_auc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1:        {test_f1:.4f}")

        print(f"\n[OK] Modelo guardado en: {model_path}")
        print("[OK] Run registrada en MLflow.")


if __name__ == "__main__":
    train_and_evaluate()
