Credit Card Fraud Detection – End-to-End ML Pipeline
🔄 English & Spanish Documentation (Bilingual)
🧩 Overview / Resumen

This repository contains a production-oriented, end-to-end machine learning pipeline for detecting credit card fraud.
It includes:

Data ingestion

Preprocessing & feature engineering

Model training with MLflow

Experiment tracking

Serialized model for inference

REST API using FastAPI

Evaluation dashboard using Streamlit

Este repositorio contiene un pipeline de machine learning completo orientado a producción para detectar fraude en transacciones de tarjeta.
Incluye:

Ingesta de datos

Preprocesado y feature engineering

Entrenamiento de modelos con MLflow

Registro de experimentos

Modelo serializado para inferencia

API REST con FastAPI

Dashboard de evaluación en Streamlit

📁 Project Structure / Estructura del Proyecto
credit-card-fraud-ml-pipeline/
│
├── data/
│   ├── raw/          # Raw dataset (ignored by Git)
│   ├── interim/
│   └── processed/    # Train/test after preprocessing
│
├── models/
│   ├── trained/      # Serialized models (ignored)
│   └── artifacts/    # Scalers, transformers (ignored)
│
├── src/
│   ├── config/       # Global paths & settings
│   ├── ingestion/    # Raw data ingestion
│   ├── preprocessing # Feature engineering
│   ├── training/     # ML training with MLflow
│   ├── inference/    # Prediction utilities
│   └── utils/        # I/O and helpers
│
├── api/              # FastAPI for inference
├── dashboard/        # Streamlit dashboard
├── mlflow/           # Local MLflow experiment store (ignored)
│
├── .gitignore
├── requirements.txt
└── README.md

⚙️ Tech Stack / Tecnologías

Python 3.11

pandas, numpy, scikit-learn

MLflow (experiment tracking)

FastAPI + Uvicorn

Streamlit

matplotlib

joblib

🚀 Setup (EN)
1️⃣ Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Add dataset

Download the public “Credit Card Fraud Detection” dataset and place it at:

creditcard.csv


(do NOT add it to Git)

3️⃣ Run end-to-end pipeline
Ingestion
python -m src.ingestion.make_dataset

Preprocessing
python -m src.preprocessing.build_features

Training (MLflow)
python -m src.training.train_model

Start MLflow UI
mlflow ui --backend-store-uri mlflow


Visit → http://127.0.0.1:5000

4️⃣ API (FastAPI)
uvicorn api.main:app --reload


Open → http://127.0.0.1:8000/docs

5️⃣ Dashboard (Streamlit)
streamlit run dashboard/app.py

🚀 Puesta en marcha (ES)
1️⃣ Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Añadir dataset

Descarga el dataset público de fraude y colócalo como:

creditcard.csv


(no lo subas a GitHub)

3️⃣ Ejecutar el pipeline completo
Ingesta
python -m src.ingestion.make_dataset

Preprocesado
python -m src.preprocessing.build_features

Entrenamiento (MLflow)
python -m src.training.train_model

Ver experimentos en MLflow
mlflow ui --backend-store-uri mlflow

4️⃣ API (FastAPI)
uvicorn api.main:app --reload


Ir a → http://127.0.0.1:8000/docs

5️⃣ Dashboard (Streamlit)
streamlit run dashboard/app.py

📊 Model Performance / Rendimiento del Modelo

Typical performance with the baseline Random Forest:

Metric	Train	Test
AUC ROC	~1.00	~0.95
Precision	High	High
Recall	Moderate-High	Moderate
F1	Strong	Strong
🧪 MLflow Tracking

Full run history

Hyperparameters

Metrics

Artifacts (models, scalers)

Comparison between runs

Esto demuestra flujo realista de trabajo como en una consultora.

🌐 API Example / Ejemplo de API
Request
{
  "Time": 0,
  "V1": -1.3598,
  "V2": -0.0727,
  ...
  "Amount": 149.62
}

Response
{
  "fraud_probability": 0.0123,
  "is_fraud": 0
}

📈 Dashboard Features / Funcionalidades del Dashboard

Class distribution

Key metrics

Confusion matrix

ROC curve

Sample of processed data

Es perfecto para enseñar el proyecto en una demo o entrevista.

🔮 Next Steps / Siguientes Pasos (Roadmap)

Planned enhancements / Mejoras previstas:

Add XGBoost, LightGBM, CatBoost models

Threshold optimization (precision-recall tradeoff)

Cost-based evaluation (business impact)

Drift detection (future extension)

Dockerization (API + dashboard)

CI/CD with GitHub Actions

📌 Author / Autor

Daniel Sánchez – Data Science / ML Engineer
GitHub: https://github.com/danielsmdev

LinkedIn: https://www.linkedin.com/in/daniel-sanchez-datascience/