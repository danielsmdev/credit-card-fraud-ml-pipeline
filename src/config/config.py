from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Directorios de modelos
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

# MLflow local
MLFLOW_DIR = PROJECT_ROOT / "mlflow"

# Nombre del modelo final
FINAL_MODEL_NAME = "fraud_model.pkl"
SCALER_NAME = "scaler.pkl"
