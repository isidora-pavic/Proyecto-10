"""Tests de la conexion tabla mezclada -> solar.py (I_pared, I_techo).

Datos de trabajo: mezclar(cargar_datos(...), cargar_inia(...)), es decir,
magnitud INIA 2025 + reparto directa/difusa del Explorador Solar 2016.
El Explorador puro (cargar_datos) solo se usa como punto de comparacion.
No se modifica cryoevap/: solo cambia la entrada `datos`.

Ejecutar con `pytest -s` para ver el reporte del test 2 (Delta t y nodos).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryoevap.datos_solar import cargar_datos, cargar_inia, factor_escala, mezclar
from cryoevap.solar import I_pared, I_techo, beta1

DATA = Path(__file__).parents[1] / "data"
RUTA_EXP = DATA / "explorador_solar_mejillones_2016-12-21.csv"
RUTA_INIA = DATA / "mejillones_inia_2025-12-21.csv"

T0 = 0.0                                          # hora de reloj (UTC-4) al inicio [h]
T_S = np.arange(0, 86400 + 1, 600, dtype=float)   # un dia cada 10 min [s]
RHO_G = 0.2   # Kanoglu, Cengel y Cimbala, cap. 3. Supuesto para Mejillones.
# d = 355 y la geografia por defecto de beta1 (no se pasan).


# ---------- fixtures y ayudas ----------

@pytest.fixture
def exp():
    return cargar_datos(RUTA_EXP)


@pytest.fixture
def inia():
    return cargar_inia(RUTA_INIA)


@pytest.fixture
def tabla(exp, inia):
    return mezclar(exp, inia)


def inia_sintetico(exp):
    """INIA ficticio con ghi_INIA(t_k) = ghi_Exp(t_k) en los nodos hh:30 -> c = 1.

    T_amb_K = 290 y v_viento_ms = 3 son valores ficticios (mezclar los exige).
    """
    t_k = np.arange(24) + 0.5
    ghi = np.interp(t_k, exp["hora_reloj"].to_numpy(), exp["ghi"].to_numpy(),
                    period=24.0)
    return pd.DataFrame({
        "hora_reloj": t_k,
        "ghi": ghi,
        "T_amb_K": np.full(24, 290.0),
        "v_viento_ms": np.full(24, 3.0),
    })


# ---------- 1. limite de consistencia: c = 1 reproduce el Explorador ----------

def test_limite_c_igual_1(exp):
    """Si el INIA midiera lo mismo que el Explorador (c = 1), la mezcla debe dar
    las mismas I_pared(t) e I_techo(t) que el Explorador puro.

    Solo en los tramos [t_k, t_k+1] con ambos nodos sobre UMBRAL_GHI: fuera de
    ellos c baja a 0 por el umbral y la igualdad no aplica.
    """
    inia1 = inia_sintetico(exp)
    tabla1 = mezclar(exp, inia1)

    t_k, c_k = factor_escala(exp, inia1)
    sobre = c_k > 0
    assert sobre.sum() >= 12                     # que el test no pase con cero nodos
    assert np.all(c_k[sobre] == 1.0)

    h = T0 + T_S / 3600.0
    k = np.floor(h - 0.5).astype(int) % 24       # nodo hh:30 a la izquierda de h
    tramo = sobre[k] & sobre[(k + 1) % 24]
    assert tramo.sum() >= 60

    casos = [
        ("pared", I_pared(T_S, T0, tabla1, rho_G=RHO_G),
                  I_pared(T_S, T0, exp, rho_G=RHO_G)),
        ("techo", I_techo(T_S, T0, tabla1), I_techo(T_S, T0, exp)),
    ]
    for nombre, I_mezcla, I_exp in casos:
        assert I_exp[tramo].max() > 100.0, nombre   # que no compare solo ceros
        np.testing.assert_allclose(I_mezcla[tramo], I_exp[tramo],
                                   rtol=1e-8, atol=1e-9, err_msg=nombre)


# ---------- 2. hora y geometria: dirh/dni vs sin(beta_1) de solar.py ----------

def test_hora_geometria(exp, inia, tabla):
    """dirh = I_DN sin(beta_1)  =>  sin(beta_1)^datos = dirh/dni.

    El cociente no depende de c (dirh y dni se escalan igual), asi que compara
    la geometria del Explorador con beta1() de solar.py.
    Nodos: hh:00 (datos del Explorador, no interpolados) con beta_1 >= 10 grados,
    c > 0. Tolerancia: 0,02 en sin(beta_1) en cada nodo.
    Delta t (indicador principal, solo se informa): desfase que minimiza
        sum_k (dirh_k/dni_k - sin beta_1^solar.py(t_k + Delta t))^2
    barriendo de -120 a +120 min cada 1 min.
    """
    horas = np.arange(24.0)
    fila = tabla.set_index("hora_reloj").loc[horas]
    dni = fila["dni"].to_numpy()
    dirh = fila["dirh"].to_numpy()

    t_k, c_k = factor_escala(exp, inia)
    c = np.interp(horas, t_k, c_k, period=24.0)
    t_nodos = (horas - T0) * 3600.0
    b = beta1(t_nodos, T0)

    usar = (b >= 10.0) & (c > 0) & (dni > 0)
    assert usar.sum() >= 8                       # que el test no pase con pocos nodos

    sin_datos = dirh[usar] / dni[usar]
    sin_solar = np.sin(np.radians(b[usar]))
    err = sin_datos - sin_solar

    dt_min = np.arange(-120.0, 121.0, 1.0)
    sse = np.array([
        np.sum((sin_datos
                - np.sin(np.radians(beta1(t_nodos[usar] + d * 60.0, T0)))) ** 2)
        for d in dt_min
    ])
    i_min = int(np.argmin(sse))

    print("\n[test 2] dirh/dni vs sin(beta_1) de solar.py, nodos hh:00 (UTC-4)")
    print(f"{'hora':>5} {'beta_1':>8} {'c':>7} {'dirh/dni':>9} {'sin b1':>8} {'dif':>8}")
    for h, bb, cc, sd, ss, e in zip(horas[usar], b[usar], c[usar],
                                    sin_datos, sin_solar, err):
        print(f"{h:5.0f} {bb:8.2f} {cc:7.4f} {sd:9.4f} {ss:8.4f} {e:+8.4f}")
    print(f"Nodos usados: {usar.sum()} -> {horas[usar].astype(int).tolist()}")
    print(f"max |dif| = {np.abs(err).max():.4f} (tolerancia 0,02)")
    print(f"Delta t ajustado = {dt_min[i_min]:+.0f} min "
          f"(SSE {sse[i_min]:.3e}; con Delta t = 0: {sse[dt_min == 0][0]:.3e})")
    print("Convencion: sin beta_1^solar.py(t_k + Delta t) ~ dirh_k / dni_k")
    if i_min in (0, len(dt_min) - 1):
        print("AVISO: Delta t en el borde del barrido; el minimo puede estar fuera.")

    assert np.all(np.abs(err) <= 0.02), (
        f"nodos fuera de tolerancia: {horas[usar][np.abs(err) > 0.02].tolist()}")


# ---------- 3. integridad: dirh = ghi - difh ----------

def test_dirh_igual_ghi_menos_difh(exp, tabla):
    """solar.py usa ghi - difh como directa del techo; los datos traen dirh.

    CSV del Explorador: cada valor con 2 decimales (error <= 0,005 W/m2), el
    residuo combina 3 valores -> atol = 3 * 0,005 = 0,015 W/m2.
    Tabla mezclada: igualdad exacta por linealidad de mezclar -> atol = 1e-9 W/m2.
    """
    assert (exp["ghi"] > 0).sum() >= 12
    assert (tabla["ghi"] > 0).sum() >= 70

    res_exp = (exp["ghi"] - exp["difh"] - exp["dirh"]).abs().max()
    res_tab = (tabla["ghi"] - tabla["difh"] - tabla["dirh"]).abs().max()
    print(f"\n[test 3] max |ghi - difh - dirh|: Explorador {res_exp:.2e}, "
          f"tabla {res_tab:.2e} W/m2")

    np.testing.assert_allclose(exp["dirh"], exp["ghi"] - exp["difh"],
                               rtol=0, atol=0.015,
                               err_msg="CSV Explorador (redondeo a 2 decimales)")
    np.testing.assert_allclose(tabla["dirh"], tabla["ghi"] - tabla["difh"],
                               rtol=0, atol=1e-9,
                               err_msg="tabla mezclada (exacta por linealidad)")
