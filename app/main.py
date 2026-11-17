from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import json
import math
import pandas as pd
import numpy as np

app = FastAPI(title="MOS-RF API", version="2.1")

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

# Calibración del luxómetro (generada por el entrenamiento):
# csi crudo = luz[lx] / toa_om[W/m2]; la referencia de cielo despejado es una
# ENVELOPENTE EMPÍRICA csi_despejado(cos_zenit) del propio sensor (el BH1750
# no ve el sol directo a sol bajo y se satura en el mediodía, así que la
# referencia no es una constante ni una exponencial).
# kt = csi / csi_despejado(cos_zenit) -> ~1.0 = despejado a cualquier hora.
with open('models/k_despejado.json') as f:
    K_CAL = json.load(f)
COS_Z_TABLA = np.array(K_CAL['cos_zenit'], dtype=float)
CSI_TABLA = np.array(K_CAL['csi_despejado'], dtype=float)
TECHO_LUX = float(K_CAL.get('techo_lux', 54612.5))
TECHO_RAD_EST = TECHO_LUX * 0.0079   # techo del sensor en unidades de la feature

DESCRIPCION_WMO = {
    0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado",
    3: "Cubierto", 45: "Niebla", 48: "Niebla con escarcha",
    51: "Llovizna ligera", 53: "Llovizna moderada", 55: "Llovizna intensa",
    61: "Lluvia ligera", 63: "Lluvia moderada", 65: "Lluvia fuerte",
    80: "Chubascos ligeros", 81: "Chubascos moderados", 82: "Chubascos fuertes",
    95: "Tormenta", 96: "Tormenta con granizo", 99: "Tormenta fuerte con granizo",
}

# ------------------------------------------------------------------
# Posición solar para la ubicación de la estación.
# ------------------------------------------------------------------
LAT = float(K_CAL.get('lat', -32.13))
LON = float(K_CAL.get('lon', -64.36))
TZ_OFFSET = -3.0        # America/Argentina/Cordoba (sin horario de verano)
COS_Z_MIN = 0.10        # debajo: noche/crepúsculo (kt = -1, sentinel)


def _ecuacion_tiempo(n: int) -> float:
    """Ecuación del tiempo en minutos (aproximación estándar)."""
    b = math.radians(360.0 * (n - 81) / 364.0)
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def _cos_zenit(hora_local: float, dia_del_ano: int) -> float:
    """Coseno del ángulo cenital solar para la ubicación de la estación."""
    hora_utc = hora_local - TZ_OFFSET
    hora_solar = hora_utc + LON / 15.0 + _ecuacion_tiempo(dia_del_ano) / 60.0
    h = math.radians(15.0 * (hora_solar - 12.0))
    delta = math.radians(-23.44 * math.cos(math.radians(360.0 * (dia_del_ano + 10) / 365.0)))
    phi = math.radians(LAT)
    return math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)


def _csi_despejado(cos_z: float) -> float:
    """Envolvente empírica interpolada (np.interp recorta fuera de rango)."""
    return float(np.interp(cos_z, COS_Z_TABLA, CSI_TABLA))


class DatosEntrada(BaseModel):
    # Features obligatorias (siempre deben venir)
    temp_om: float
    humedad_om: float
    hora_del_dia: int      # hora LOCAL
    dia_del_ano: int

    # Features opcionales del pronóstico
    presion_om: Optional[float] = None
    lluvia_om: Optional[float] = None
    precipitacion_om: Optional[float] = None
    rocio_om: Optional[float] = None
    viento_om: Optional[float] = None
    radiacion_om: Optional[float] = None

    # Luminosidad de la estación (solo hora actual): luz en lx, toa_om en W/m2
    # de la misma hora física. Para horas futuras NO enviar radiacion_est:
    # la API la estima con la envolvente y el kt predicho.
    radiacion_est: Optional[float] = None
    luz: Optional[float] = None
    toa_om: Optional[float] = None
    # 1 = día, 0 = noche. Si no viene se infiere de la posición solar.
    es_de_dia: Optional[int] = None


class EstadoEntrada(BaseModel):
    """Entrada para corregir estado del tiempo y lluvia.

    Dos modos:
    - Hora actual: mandar 'luz' y 'toa_om' -> kt observado por el luxómetro.
    - Hora futura (pronóstico): omitir 'luz' -> kt predicho con
      modelo_nubosidad a partir del pronóstico de Open-Meteo.
    """
    hora_del_dia: int      # hora LOCAL
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


def _es_de_dia(datos, cos_z: float) -> int:
    """1 = día, 0 = noche. Prioridad:
    1. es_de_dia explícito (opcional, p.ej. is_day de Open-Meteo)
    2. posición solar (cos_zenit > 0.10 ~ elevación > 5.7°)
    """
    if datos.es_de_dia is not None:
        return int(datos.es_de_dia)
    return int(cos_z > COS_Z_MIN)


def _kt_observado(luz, toa_om, cos_z: float):
    """kt medido por el luxómetro, o None si no hay lectura válida de día.

    Requiere sol sobre el horizonte (cos_z > 0.10) y lectura positiva.
    Si el sensor está saturado (techo del BH1750), el valor es un límite
    inferior: el clip a 1.3 lo lleva igual al rango 'despejado'.
    """
    if luz is None or toa_om is None or toa_om <= 0 or luz <= 0:
        return None
    if cos_z <= COS_Z_MIN:
        return None
    csi = luz / toa_om
    return float(min(max(csi / _csi_despejado(cos_z), 0.0), 1.3))


def _kt_predicho(valores: dict) -> float:
    """kt pronosticado por el modelo de nubosidad a partir de Open-Meteo."""
    X = _preparar(features_nub, valores)
    kt = float(modelo_nub.predict(X)[0])
    return float(min(max(kt, 0.0), 1.3))


def _prob_lluvia(valores: dict) -> float:
    X = _preparar(features_lluvia, valores)
    proba = modelo_lluvia.predict_proba(X)[0]
    clases = list(modelo_lluvia.classes_)
    return float(proba[clases.index(1)] if 1 in clases else proba[-1])


def sugerir_estado(kt: float, prob_lluvia: float, nubosidad_om=None,
                   es_de_dia: int = 1) -> int:
    """Combina kt (1.0 = despejado según el sensor) y prob. de lluvia en WMO.

    Umbrales calibrados con la envolvente empírica del sensor:
    horas despejadas ~0.88 de kt medio, cielo cubierto ~0.60.
    De noche el luxómetro no informa: se usa la nubosidad de Open-Meteo.
    """
    if prob_lluvia >= 0.6:
        return 63 if prob_lluvia >= 0.85 else 61   # lluvia moderada / leve

    if not es_de_dia:
        nub = nubosidad_om if nubosidad_om is not None else 40
        if nub >= 85:
            return 3                                # cubierto
        if nub >= 40:
            return 2                                # parcialmente nublado
        if nub >= 10:
            return 1                                # mayormente despejado
        return 0                                    # despejado

    if kt >= 0.95:
        return 0                                    # despejado
    if kt >= 0.72:
        return 1                                    # mayormente despejado
    if nubosidad_om is not None and nubosidad_om >= 85:
        return 3                                    # cubierto
    if kt >= 0.45:
        return 2                                    # parcialmente nublado
    # Zona ambigua (kt 0.30-0.45): puede ser cielo despejado con el sensor
    # obstruido a sol bajo. Open-Meteo desempata: poco nubosidad -> despejado.
    if kt >= 0.30 and nubosidad_om is not None and nubosidad_om <= 25:
        return 1                                    # mayormente despejado
    if kt < 0.25 and prob_lluvia >= 0.35:
        return 51                                   # llovizna (muy oscuro)
    return 2


@app.post("/predecir")
def predecir(datos: DatosEntrada):
    cos_z = _cos_zenit(datos.hora_del_dia, datos.dia_del_ano)
    dia = _es_de_dia(datos, cos_z)
    kt_obs = _kt_observado(datos.luz, datos.toa_om, cos_z)

    if kt_obs is not None:
        kt = kt_obs
    elif dia:
        # Día pero sin lectura del luxómetro: usar el pronóstico de nubosidad
        kt = _kt_predicho(datos.dict())
    else:
        kt = -1.0  # noche (sentinel igual que en el entrenamiento)

    valores = {**datos.dict(), 'kt': kt, 'es_de_dia': dia}

    # radiacion_est de horas futuras: lo que el sensor marcaría con ese kt
    # a esa posición solar: csi_despejado(cos_z) * kt * toa_om * 0.0079,
    # limitado al techo del luxómetro. Sin toa_om, cae al factor mediano.
    if valores.get('radiacion_est') is None and kt >= 0:
        toa = valores.get('toa_om') or 0
        if toa > 0:
            proxy = _csi_despejado(cos_z) * kt * toa * 0.0079
        else:
            proxy = (valores.get('radiacion_om') or 0) \
                * float(K_CAL.get('factor_lux_om', 13.9)) * 0.0079
        valores['radiacion_est'] = float(min(max(proxy, 0.0), TECHO_RAD_EST))

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
        "cos_zenit": round(cos_z, 3),
        "radiacion_est_usada": valores.get('radiacion_est'),
    }


@app.post("/estado")
def estado(datos: EstadoEntrada):
    cos_z = _cos_zenit(datos.hora_del_dia, datos.dia_del_ano)
    dia = _es_de_dia(datos, cos_z)
    kt_obs = _kt_observado(datos.luz, datos.toa_om, cos_z)
    saturado = bool(datos.luz is not None and datos.luz >= TECHO_LUX - 1)
    valores = {**datos.dict(), 'es_de_dia': dia}

    if kt_obs is not None:
        kt, fuente = kt_obs, "observado"
    elif dia:
        kt, fuente = _kt_predicho(valores), "predicho"
    else:
        kt, fuente = -1.0, "noche"

    valores['kt'] = kt
    prob = _prob_lluvia(valores)
    wmo = sugerir_estado(kt, prob, datos.nubosidad_om, dia)

    return {
        "kt": None if kt < 0 else round(kt, 3),
        "kt_observado": fuente == "observado",
        "fuente_kt": fuente,
        "sensor_saturado": saturado,
        "es_de_dia": dia,
        "cos_zenit": round(cos_z, 3),
        "prob_lluvia": round(prob, 3),
        "nubosidad_om": datos.nubosidad_om,
        "estado_wmo": wmo,
        "estado_descripcion": DESCRIPCION_WMO.get(wmo, "Desconocido"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.1",
        "calibracion": {
            "modelo": K_CAL.get('modelo'),
            "cos_zenit": K_CAL.get('cos_zenit'),
            "csi_despejado": K_CAL.get('csi_despejado'),
            "techo_lux": TECHO_LUX,
            "n": K_CAL.get('n'),
        },
        "modelos": {
            "temperatura": {"features": features_temp, "n_features": len(features_temp)},
            "nubosidad": {"features": features_nub, "n_features": len(features_nub)},
            "lluvia": {"features": features_lluvia, "n_features": len(features_lluvia)},
        },
    }
