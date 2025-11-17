from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.config.config import MODELS_DIR, ARTIFACTS_DIR, FINAL_MODEL_NAME, SCALER_NAME

_model = None
_scaler = None
_FEATURE_ORDER: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


def _load_model():
    global _model
    if _model is None:
        model_path = MODELS_DIR / FINAL_MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        _model = joblib.load(model_path)
    return _model


def _load_scaler():
    global _scaler
    if _scaler is None:
        scaler_path = ARTIFACTS_DIR / SCALER_NAME
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler no encontrado en {scaler_path}")
        _scaler = joblib.load(scaler_path)
    return _scaler


def predict_proba(features: Dict[str, float]) -> float:
    """
    Recibe un diccionario con las features de una transacción y devuelve
    la probabilidad de fraude entre 0 y 1.
    """
    model = _load_model()
    scaler = _load_scaler()

    # Ordenar columnas según el orden usado en el entrenamiento
    row = [features[col] for col in _FEATURE_ORDER]
    X = np.array(row).reshape(1, -1)

    # Escalar con el mismo scaler usado en training
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[:, 1][0]
    return float(proba)
