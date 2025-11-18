# Credit Card Fraud Detection -- End-to-End ML Pipeline

### 🔄 English & Spanish Documentation (Bilingual)

## 🧩 Overview / Resumen

This repository contains a **production-oriented, end-to-end machine
learning pipeline** for detecting credit card fraud.\
It includes: 

- Data ingestion\
- Preprocessing & feature engineering\
- Model training with MLflow\
- Experiment tracking\
- Serialized model for inference\
- REST API using FastAPI\
- Evaluation dashboard using Streamlit

Este repositorio contiene un **pipeline de machine learning completo
orientado a producción** para detectar fraude en transacciones de
tarjeta.\

Incluye: 

- Ingesta de datos\
- Preprocesado y feature engineering\
- Entrenamiento de modelos con MLflow\
- Registro de experimentos\
- Modelo serializado para inferencia\
- API REST con FastAPI\
- Dashboard de evaluación en Streamlit

## 📁 Project Structure / Estructura del Proyecto

``` bash
credit-card-fraud-ml-pipeline/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── trained/
│   └── artifacts/
│
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── preprocessing
│   ├── training/
│   ├── inference/
│   └── utils/
│
├── api/
├── dashboard/
├── mlflow/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Tech Stack / Tecnologías

-   Python 3.11\
-   pandas, numpy, scikit-learn\
-   MLflow\
-   FastAPI + Uvicorn\
-   Streamlit\
-   matplotlib\
-   joblib

## 🚀 Setup (EN)

### 1️⃣ Create virtual environment

    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt

### 2️⃣ Add dataset

Place as:

    creditcard.csv

### 3️⃣ Run pipeline

    python -m src.ingestion.make_dataset
    python -m src.preprocessing.build_features
    python -m src.training.train_model

### 4️⃣ MLflow UI

    mlflow ui --backend-store-uri mlflow

### 5️⃣ API

    uvicorn api.main:app --reload

### 6️⃣ Dashboard

    streamlit run dashboard/app.py

## 📊 Model Performance / Rendimiento del Modelo

  Metric      Train           Test
  ----------- --------------- ----------
  AUC ROC     \~1.00          \~0.95
  Precision   High            High
  Recall      Moderate-High   Moderate
  F1          Strong          Strong

## 🧪 MLflow Tracking

-   Run history\
-   Metrics\
-   Hyperparameters\
-   Artifacts

## 🌐 API Example / Ejemplo de API

Request:

``` json
{"Time":0,"V1":-1.3,"V2":-0.07,"Amount":149.62}
```

## 📈 Dashboard Features

-   Class distribution\
-   Key metrics\
-   Confusion matrix\
-   ROC curve\
-   Data sample

## 🔮 Roadmap

-   XGBoost, LightGBM\
-   Threshold optimization\
-   Cost-based evaluation\
-   Drift detection\
-   Dockerization\
-   CI/CD

## 📌 Author

**Daniel Sánchez**\

GitHub: https://github.com/danielsmdev

LinkedIn: https://www.linkedin.com/in/daniel-sanchez-datascience/