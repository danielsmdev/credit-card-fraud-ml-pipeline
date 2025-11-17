from pathlib import Path
from src.config.config import RAW_DIR
from src.utils.io_utils import load_csv, save_csv


def ingest_raw_data(input_path: Path) -> None:
    """
    Carga el dataset original y lo guarda en data/raw/
    """
    df = load_csv(input_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / input_path.name

    save_csv(df, output_path)
    print(f"[OK] Raw data guardado en: {output_path}")


if __name__ == "__main__":
    # Asumimos que el fichero está en la raíz del proyecto con este nombre
    input_file = Path("creditcard.csv")
    ingest_raw_data(input_file)
