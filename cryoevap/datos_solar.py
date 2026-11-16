import pandas as pd
import numpy as np

def cargar_datos(ruta):
    """Lee el extracto horario del Explorador Solar (un dia, 24 filas).

    Devuelve un DataFrame con la hora de reloj (UTC-4, en horas) y las
    columnas dni, dirh, difh, ghi (W/m2), temp (C) y vel (m/s).
    """
    df = pd.read_csv(ruta, parse_dates=["Fecha/Hora"])
    hora = df["Fecha/Hora"].dt.hour + df["Fecha/Hora"].dt.minute / 60.0
    out = df[["dni", "dirh", "difh", "ghi", "temp", "vel"]].copy()
    out.insert(0, "hora_reloj", hora)
    return out.reset_index(drop=True)

def perfil_radiacion(t, t0, datos):
    """Radiacion en el instante de simulacion t (segundos).

    t0: hora de reloj (UTC-4, en horas) a la que parte la simulacion.
    datos: DataFrame de cargar_datos().
    Interpola linealmente entre las horas del archivo y repite el dia
    cada 24 h, para poder simular varios dias seguidos.
    Devuelve (dni, difh, ghi) en W/m2.
    """
    hora = t0 + np.asarray(t, dtype=float) / 3600.0
    h = datos["hora_reloj"].to_numpy()

    def interp(col):
        return np.interp(hora, h, datos[col].to_numpy(), period=24.0)

    return interp("dni"), interp("difh"), interp("ghi")


# ---------------------------------------------------------------------------
# Mezcla de radiacion: magnitud INIA 2025 + reparto Explorador Solar 2016
# ---------------------------------------------------------------------------

DESFASE_INIA_H = 0.5
"""Corrimiento [h] que se suma a la etiqueta de hora del INIA.

Hipotesis de trabajo (no confirmada por INIA): la etiqueta marca el INICIO
del promedio horario, por lo que cada dato representa hora + 0,5 h.
- Radiacion: respaldado por la comparacion con el Explorador Solar 2016
  (etiqueta centrada, comprobada con dirh/dni contra sin(beta_1)).
- Temperatura y viento: se aplica por consistencia (mismo registrador),
  SIN verificacion independiente.
Cambiar aqui para el analisis de sensibilidad (por ejemplo 0.0).
"""

UMBRAL_GHI = 10.0
"""Umbral [W/m2] de ghi del Explorador bajo el cual el factor de escala c = 0.

Bajo este nivel el cociente ghi_INIA / ghi_Exp mide ruido, no cielo: es del
orden del zero offset de un piranometro (ISO 9060:2018, clases A y B,
~7-15 W/m2). Clase del sensor INIA por confirmar.
"""

_COLS_RAD = ["dni", "dirh", "difh", "ghi"]


def cargar_inia(ruta):
    """Lee el CSV horario de la estacion INIA (un dia, 24 filas).

    Salta las 5 lineas de cabecera y descarta el pie de pagina (filas sin
    fecha valida). Devuelve un DataFrame con:
      hora_reloj  [h]     hora de la etiqueta (UTC-4) + DESFASE_INIA_H
      ghi         [W/m2]  radiacion global horizontal
      T_amb_K     [K]     T[C] + 273.15
      v_viento_ms [m/s]   v[km/h] / 3.6
    """
    df = pd.read_csv(ruta, skiprows=5)
    # Columnas por posicion; se verifica el encabezado para no leer la equivocada.
    nombres = [str(c) for c in df.columns[:4]]
    for i, clave in [(1, "Temperatura"), (2, "Radiaci"), (3, "Viento")]:
        if clave not in nombres[i]:
            raise ValueError(f"Columna {i} inesperada en {ruta}: {nombres[i]!r}")

    fecha = pd.to_datetime(df.iloc[:, 0], format="%d-%m-%Y %H:%M", errors="coerce")
    ok = fecha.notna().to_numpy()
    df, fecha = df[ok], fecha[ok]

    hora = (fecha.dt.hour + fecha.dt.minute / 60.0).to_numpy(dtype=float)
    return pd.DataFrame({
        "hora_reloj": hora + DESFASE_INIA_H,
        "ghi": df.iloc[:, 2].astype(float).to_numpy(),
        "T_amb_K": df.iloc[:, 1].astype(float).to_numpy() + 273.15,
        "v_viento_ms": df.iloc[:, 3].astype(float).to_numpy() / 3.6,
    })


def factor_escala(datos_exp, datos_inia):
    """Factor c_k = ghi_INIA,k / ghi_Exp(t_k) en los nodos INIA t_k (hh:30).

    ghi_Exp(t_k): ghi del Explorador interpolada linealmente a t_k.
    c_k = 0 si ghi_Exp(t_k) < UMBRAL_GHI (amanecer, atardecer y noche).
    Sin tope superior. Devuelve (t_k [h], c_k [-]).
    """
    t_k = datos_inia["hora_reloj"].to_numpy(dtype=float)
    I_inia = datos_inia["ghi"].to_numpy(dtype=float)
    I_exp = np.interp(t_k, datos_exp["hora_reloj"].to_numpy(),
                      datos_exp["ghi"].to_numpy(), period=24.0)
    sobre = I_exp >= UMBRAL_GHI
    c_k = np.where(sobre, I_inia / np.where(sobre, I_exp, 1.0), 0.0)
    return t_k, c_k


def mezclar(datos_exp, datos_inia):
    """Tabla de radiacion mezclada cada 10 min, lista para perfil_radiacion.

    Para cada columna X en (dni, dirh, difh, ghi):
        X(t) = c(t) * X_Exp(t)
    con c(t) interpolado entre los nodos INIA y X_Exp(t) entre los nodos del
    Explorador, por separado (el producto se calcula aqui, no se interpola).
    T_amb_K y v_viento_ms se interpolan desde los nodos INIA.
    Grilla: np.arange(144) / 6 h (contiene hh:00 y hh:30 exactos).
    """
    g = np.arange(144) / 6.0
    t_k, c_k = factor_escala(datos_exp, datos_inia)
    c_g = np.interp(g, t_k, c_k, period=24.0)
    h_exp = datos_exp["hora_reloj"].to_numpy()

    out = {"hora_reloj": g}
    for col in _COLS_RAD:
        out[col] = c_g * np.interp(g, h_exp, datos_exp[col].to_numpy(), period=24.0)
    for col in ["T_amb_K", "v_viento_ms"]:
        out[col] = np.interp(g, t_k, datos_inia[col].to_numpy(), period=24.0)
    return pd.DataFrame(out)