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