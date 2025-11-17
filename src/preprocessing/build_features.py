from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config.config import RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, SCALER_NAME
from src.utils.io_utils import load_csv, save_csv

TARGET_COL = "Class"  # ajusta si tu columna objetivo se llama distinto


def build_features(
    raw_filename: str = "creditcard.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    raw_path = RAW_DIR / raw_filename
    df = load_csv(raw_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no existe en el dataset.")

    # Separar X e y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Split estratificado (MUY importante en datos desbalanceados)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Escalado (todas las features numéricas; en este dataset lo son todas)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reconstruimos dataframes con mismos nombres de columnas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

    # Guardar datasets procesados
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(train_df, PROCESSED_DIR / "train.csv")
    save_csv(test_df, PROCESSED_DIR / "test.csv")

    # Guardar scaler como artefacto
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = ARTIFACTS_DIR / SCALER_NAME
    joblib.dump(scaler, scaler_path)

    print("[OK] Datos procesados y particionados guardados en data/processed/")
    print(f"[OK] Scaler guardado en: {scaler_path}")


if __name__ == "__main__":
    build_features()
