from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np

app = FastAPI()

# Cargar modelo y features al arrancar
modelo = joblib.load('models/modelo_temperatura.pkl')
with open('models/modelo_features.json', 'r') as f:
    features = json.load(f)

class DatosEntrada(BaseModel):
    temp_om: float
    humedad_om: float
    presion_om: float
    lluvia_om: float
    rocio_om: float
    radiacion_est: float
    radiacion_acum_3h: float
    radiacion_acum_6h: float
    radiacion_acum_12h: float
    radiacion_acum_24h: float
    hora_del_dia: int
    dia_del_ano: int

@app.post("/predecir")
def predecir(datos: DatosEntrada):
    df = pd.DataFrame([datos.dict()])[features]
    error_predicho = float(modelo.predict(df)[0])
    temp_corregida = datos.temp_om + error_predicho
    
    return {
        "temp_om": datos.temp_om,
        "error_predicho": round(error_predicho, 2),
        "temp_corregida": round(temp_corregida, 2)
    }

@app.get("/health")
def health():
    return {"status": "ok", "features": features}
