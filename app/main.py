from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import json
import math
import pandas as pd
import numpy as np

app = FastAPI(title="MOS-RF API", version="2.0")

# ------------------------------------------------------------------
# Cargar modelos y features al arrancar
# ------------------------------------------------------------------
modelo_temp = joblib.load('models/modelo_temperatura.pkl')
modelo_nub = joblib.load('models/modelo_nubosidad.pkl')
modelo_lluvia = joblib.load('models/modelo_lluvia.pkl')

with open('models/modelo_features_temperatura.json') as f:
    features_temp = json.load(f)
with open('models/modelo_features_nubosidad.json') as f:
    features_nub = json.load(f)
with open('models/modelo_features_lluvia.json') as f:
    features_lluvia = json.load(f)

# Constante de calibración del luxómetro (generada por el entrenamiento):
# csi crudo = luz[lx] / toa_om[W/m2]; kt = csi / K_DESPEJADO -> 1.0 = despejado
with open('models/k_despejado.json') as f:
    K_DESPEJADO = json.load(f)['k_despejado']

DESCRIPCION_WMO = {
    0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado",
    3: "Cubierto", 45: "Niebla", 48: "Niebla con escarcha",
    51: "Llovizna ligera", 53: "Llovizna moderada", 55: "Llovizna intensa",
    61: "Lluvia ligera", 63: "Lluvia moderada", 65: "Lluvia fuerte",
    80: "Chubascos ligeros", 81: "Chubascos moderados", 82: "Chubascos fuertes",
    95: "Tormenta", 96: "Tormenta con granizo", 99: "Tormenta fuerte con granizo",
}

# ------------------------------------------------------------------
# Posición solar: para inferir día/noche sin que n8n lo envíe.
# Mismo criterio que el entrenamiento (toa_om > 50 W/m2 ~ sol > 2°).
# ------------------------------------------------------------------
LAT = -32.13
LON = -64.36
TZ_OFFSET = -3.0        # America/Argentina/Cordoba (sin horario de verano)
ELEV_MIN_DEG = 2.0


def _ecuacion_tiempo(n: int) -> float:
    """Ecuación del tiempo en minutos (aproximación estándar)."""
    b = math.radians(360.0 * (n - 81) / 364.0)
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def _elevacion_solar(hora_local: float, dia_del_ano: int) -> float:
    """Elevación del sol en grados para la ubicación de la estación."""
    hora_utc = hora_local - TZ_OFFSET
    hora_solar = hora_utc + LON / 15.0 + _ecuacion_tiempo(dia_del_ano) / 60.0
    h = math.radians(15.0 * (hora_solar - 12.0))
    delta = math.radians(-23.44 * math.cos(math.radians(360.0 * (dia_del_ano + 10) / 365.0)))
    phi = math.radians(LAT)
    cos_z = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)
    return math.degrees(math.asin(min(max(cos_z, -1.0), 1.0)))


class DatosEntrada(BaseModel):
    # Features obligatorias (siempre deben venir)
    temp_om: float
    humedad_om: float
    hora_del_dia: int
    dia_del_ano: int

    # Features opcionales del pronóstico
    presion_om: Optional[float] = None
    lluvia_om: Optional[float] = None
    precipitacion_om: Optional[float] = None
    rocio_om: Optional[float] = None
    viento_om: Optional[float] = None

    # Luminosidad de la estación (hora actual): luz en lx, toa_om en W/m2
    radiacion_est: Optional[float] = None
    luz: Optional[float] = None
    toa_om: Optional[float] = None
    # 1 = día, 0 = noche. Si no viene se infiere de toa_om/luz.
    es_de_dia: Optional[int] = None


class EstadoEntrada(BaseModel):
    """Entrada para corregir estado del tiempo y lluvia.

    Dos modos:
    - Hora actual: mandar 'luz' y 'toa_om' -> kt observado por el luxómetro.
    - Hora futura (pronóstico): omitir 'luz' -> kt predicho con
      modelo_nubosidad a partir del pronóstico de Open-Meteo.
    """
    hora_del_dia: int
    dia_del_ano: int
    nubosidad_om: Optional[float] = None

    # Pronóstico de Open-Meteo para la hora (features de los modelos)
    temp_om: Optional[float] = None
    humedad_om: Optional[float] = None
    presion_om: Optional[float] = None
    lluvia_om: Optional[float] = None
    precipitacion_om: Optional[float] = None
    rocio_om: Optional[float] = None
    viento_om: Optional[float] = None
    radiacion_om: Optional[float] = None
    estado_om: Optional[float] = None

    # Solo para hora actual
    luz: Optional[float] = None
    toa_om: Optional[float] = None
    es_de_dia: Optional[int] = None


def _preparar(features, valores: dict, sentinels: Optional[dict] = None):
    """Arma el DataFrame en el orden exacto de features del modelo."""
    fila = {f: valores.get(f, 0) for f in features}
    df = pd.DataFrame([fila])[features]
    for c, v in (sentinels or {}).items():
        if c in df.columns:
            df[c] = df[c].fillna(v)
    df = df.fillna(0)
    return df.astype(float)


def _es_de_dia(datos) -> int:
    """1 = día, 0 = noche. Orden de prioridad:
    1. es_de_dia explícito (opcional, p.ej. is_day de Open-Meteo)
    2. toa_om (la radiación TOA ya codifica la posición solar)
    3. luz del luxómetro
    4. cálculo de elevación solar con hora_del_dia + dia_del_ano
    """
    if datos.es_de_dia is not None:
        return int(datos.es_de_dia)
    if datos.toa_om is not None:
        return int(datos.toa_om > 50)
    if (datos.luz or 0) > 0:
        return 1
    return int(_elevacion_solar(datos.hora_del_dia, datos.dia_del_ano) > ELEV_MIN_DEG)


def _kt_observado(luz, toa_om):
    """kt medido por el luxómetro, o None si no hay lectura de día."""
    if luz is None or toa_om is None or toa_om <= 50 or luz <= 0:
        return None
    return float(min(max((luz / toa_om) / K_DESPEJADO, 0.0), 1.2))


def _kt_predicho(valores: dict) -> float:
    """kt pronosticado por el modelo de nubosidad a partir de Open-Meteo."""
    X = _preparar(features_nub, valores)
    kt = float(modelo_nub.predict(X)[0])
    return float(min(max(kt, 0.0), 1.2))


def _prob_lluvia(valores: dict) -> float:
    X = _preparar(features_lluvia, valores)
    proba = modelo_lluvia.predict_proba(X)[0]
    clases = list(modelo_lluvia.classes_)
    return float(proba[clases.index(1)] if 1 in clases else proba[-1])


def sugerir_estado(kt: float, prob_lluvia: float, nubosidad_om=None) -> int:
    """Combina kt (1.0 = despejado) y prob. de lluvia en un código WMO."""
    if prob_lluvia >= 0.6:
        return 63 if prob_lluvia >= 0.85 else 61   # lluvia moderada / leve
    if kt < 0.08 and prob_lluvia >= 0.3:
        return 51                                   # llovizna
    if kt >= 0.95:
        return 0                                    # despejado
    if kt >= 0.75:
        return 1                                    # mayormente despejado
    if kt >= 0.50:
        return 2                                    # parcialmente nublado
    if nubosidad_om is not None and nubosidad_om >= 85:
        return 3                                    # cubierto
    return 2


@app.post("/predecir")
def predecir(datos: DatosEntrada):
    dia = _es_de_dia(datos)
    kt_obs = _kt_observado(datos.luz, datos.toa_om)

    if kt_obs is not None:
        kt = kt_obs
    elif dia:
        # Día pero sin lectura del luxómetro: usar el pronóstico de nubosidad
        kt = _kt_predicho(datos.dict())
    else:
        kt = -1.0  # noche (sentinel igual que en el entrenamiento)

    valores = {**datos.dict(), 'kt': kt, 'es_de_dia': dia}
    df = _preparar(features_temp, valores, sentinels={'kt': -1.0})

    error_predicho = float(modelo_temp.predict(df)[0])
    temp_corregida = datos.temp_om + error_predicho

    return {
        "temp_om": datos.temp_om,
        "error_predicho": round(error_predicho, 2),
        "temp_corregida": round(temp_corregida, 2),
        "kt": None if kt < 0 else round(kt, 3),
        "kt_observado": kt_obs is not None,
        "es_de_dia": dia,
    }


@app.post("/estado")
def estado(datos: EstadoEntrada):
    dia = _es_de_dia(datos)
    kt_obs = _kt_observado(datos.luz, datos.toa_om)
    valores = {**datos.dict(), 'es_de_dia': dia}

    if kt_obs is not None:
        kt, fuente = kt_obs, "observado"
    elif dia:
        kt, fuente = _kt_predicho(valores), "predicho"
    else:
        kt, fuente = -1.0, "noche"

    valores['kt'] = kt
    prob = _prob_lluvia(valores)
    wmo = sugerir_estado(kt, prob, datos.nubosidad_om)

    return {
        "kt": None if kt < 0 else round(kt, 3),
        "kt_observado": fuente == "observado",
        "fuente_kt": fuente,
        "es_de_dia": dia,
        "prob_lluvia": round(prob, 3),
        "nubosidad_om": datos.nubosidad_om,
        "estado_wmo": wmo,
        "estado_descripcion": DESCRIPCION_WMO.get(wmo, "Desconocido"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "k_despejado": K_DESPEJADO,
        "modelos": {
            "temperatura": {"features": features_temp, "n_features": len(features_temp)},
            "nubosidad": {"features": features_nub, "n_features": len(features_nub)},
            "lluvia": {"features": features_lluvia, "n_features": len(features_lluvia)},
        },
    }
