from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import json
import pandas as pd
import numpy as np

app = FastAPI(title="MOS-RF API", version="1.0")

# Cargar modelo y features al arrancar
modelo = joblib.load('models/modelo_temperatura.pkl')
with open('models/modelo_features.json', 'r') as f:
    features = json.load(f)


class DatosEntrada(BaseModel):
    # Features obligatorias (siempre deben venir)
    temp_om: float
    humedad_om: float
    hora_del_dia: int
    dia_del_ano: int

    # Features opcionales (pueden venir como null si el sensor falla)
    presion_om: Optional[float] = None
    lluvia_om: Optional[float] = None
    rocio_om: Optional[float] = None
    radiacion_est: Optional[float] = None


@app.post("/predecir")
def predecir(datos: DatosEntrada):
    # Convertir a DataFrame respetando el orden exacto de features del modelo
    df = pd.DataFrame([datos.dict()])[features]

    # Random Forest maneja NaN de forma nativa: los valores nulos se ignoran
    # al buscar las divisiones en los árboles
    df = df.astype(float)

    # Predecir el error de temperatura
    error_predicho = float(modelo.predict(df)[0])

    # Temperatura corregida = temperatura de Open-Meteo + error predicho
    temp_corregida = datos.temp_om + error_predicho

    return {
        "temp_om": datos.temp_om,
        "error_predicho": round(error_predicho, 2),
        "temp_corregida": round(temp_corregida, 2)
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "features": features,
        "n_features": len(features)
    }
