import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
)

from src.config.config import PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR, FINAL_MODEL_NAME, SCALER_NAME
from src.utils.io_utils import load_csv
import joblib


TARGET_COL = "Class"


@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / FINAL_MODEL_NAME
    scaler_path = ARTIFACTS_DIR / SCALER_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler no encontrado en {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


@st.cache_data
def load_test_data():
    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set no encontrado en {test_path}")
    df_test = load_csv(test_path)
    return df_test


def main():
    st.title("Credit Card Fraud Detection - Dashboard")

    st.markdown(
        """
        Este dashboard muestra el rendimiento del modelo de detección de fraude sobre el conjunto de test.
        Incluye distribución de clases, métricas y visualizaciones clave.
        """
    )

    df_test = load_test_data()
    model, _ = load_artifacts()

    # Separar X e y
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    # Predicciones
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # Métricas
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Distribución de clases en test")
    class_counts = y_test.value_counts().rename(index={0: "No fraude", 1: "Fraude"})
    st.bar_chart(class_counts)

    st.subheader("Métricas en test")
    st.write(f"**AUC ROC:** {auc:.4f}")
    st.write(f"**Total transacciones:** {len(y_test)}")
    st.write(f"**Fraudes detectados:** {int((y_pred & (y_test == 1)).sum())}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matriz de confusión")
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No fraude", "Fraude"])
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)

    with col2:
        st.subheader("Curva ROC")
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc)
        st.pyplot(fig_roc)

    st.subheader("Muestra del conjunto de test")
    st.dataframe(df_test.head(50))


if __name__ == "__main__":
    main()
