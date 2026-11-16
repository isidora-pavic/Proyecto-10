"""Tests de la mezcla de radiacion INIA 2025 + Explorador Solar 2016.

Capa de datos: cryoevap/datos_solar.py. Escritos antes de la implementacion.
Regla de mezcla: c_k = ghi_INIA / ghi_Exp(t_k) en los nodos INIA (hh:30),
c = 0 si ghi_Exp(t_k) < UMBRAL_GHI; tabla = c(t) * Explorador(t) cada 10 min.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cryoevap.datos_solar as ds
from cryoevap.datos_solar import (
    cargar_datos, cargar_inia, factor_escala, mezclar, perfil_radiacion,
)

DATA = Path(__file__).parents[1] / "data"
RUTA_EXP = DATA / "explorador_solar_mejillones_2016-12-21.csv"
RUTA_INIA = DATA / "mejillones_inia_2025-12-21.csv"
COLS_RAD = ["dni", "dirh", "difh", "ghi"]


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


def ghi_exp_en(exp, horas):
    """ghi del Explorador interpolada a horas de reloj [h]."""
    return np.interp(horas, exp["hora_reloj"].to_numpy(),
                     exp["ghi"].to_numpy(), period=24.0)


def referencia_exacta(exp, inia, horas):
    """c(t) * ghi_Exp(t), con las dos curvas interpoladas por separado."""
    t_k, c_k = factor_escala(exp, inia)
    return np.interp(horas, t_k, c_k, period=24.0) * ghi_exp_en(exp, horas)


def energia_Wh(f_horas):
    """Integral en 24 h con paso de 1 s [Wh/m2]. f_horas: funcion de la hora [h]."""
    t_h = np.arange(86400) / 3600.0
    return f_horas(t_h).sum() / 3600.0


# ---------- 1. identidad en nodos ----------

def test_identidad_en_nodos(exp, inia):
    t_k, c_k = factor_escala(exp, inia)
    np.testing.assert_array_equal(t_k, inia["hora_reloj"].to_numpy())
    I_exp = ghi_exp_en(exp, t_k)
    sobre = I_exp >= ds.UMBRAL_GHI
    assert sobre.sum() >= 12            # que el test no pase con cero nodos
    np.testing.assert_allclose(c_k[sobre] * I_exp[sobre],
                               inia["ghi"].to_numpy()[sobre], rtol=1e-10)


# ---------- 2. noche y rango ----------

def test_noche_cero(exp, inia, tabla):
    t_k, c_k = factor_escala(exp, inia)
    bajo = ghi_exp_en(exp, t_k) < ds.UMBRAL_GHI
    assert bajo.sum() >= 8
    assert np.all(c_k[bajo] == 0.0)
    filas = tabla.set_index("hora_reloj").loc[t_k[bajo], COLS_RAD]
    assert np.all(filas.to_numpy() == 0.0)


def test_rango_c_de_dia(exp, inia):
    t_k, c_k = factor_escala(exp, inia)
    assert np.all(np.isfinite(c_k))
    dia = ghi_exp_en(exp, t_k) >= ds.UMBRAL_GHI
    assert np.all((c_k[dia] >= 0.0) & (c_k[dia] <= 1.2))


# ---------- 3. energia diaria vs INIA ----------

def test_energia_diaria(inia, tabla):
    E_mezcla = energia_Wh(lambda h: perfil_radiacion(h * 3600.0, 0.0, tabla)[2])
    E_inia = inia["ghi"].sum() * 1.0    # nodos horarios: 1 h cada uno
    assert E_mezcla == pytest.approx(E_inia, rel=5e-3)


# ---------- 4. corrimiento +0,5 h con valores del archivo ----------

@pytest.mark.parametrize("hora, ghi_inia", [(6.5, 79.7), (12.5, 1068.6)])
def test_corrimiento(tabla, hora, ghi_inia):
    fila = tabla.loc[tabla["hora_reloj"] == hora]
    assert len(fila) == 1
    np.testing.assert_allclose(fila["ghi"].iloc[0], ghi_inia, rtol=1e-10)


# ---------- 5. unidades y hora sobre la fila cruda ----------

def test_unidades_temperatura(inia):
    np.testing.assert_allclose(inia["T_amb_K"].iloc[0], 291.95, rtol=1e-12)


def test_unidades_viento(inia):
    np.testing.assert_allclose(inia["v_viento_ms"].iloc[0], 12.8 / 3.6, rtol=1e-12)


def test_hora_inia_corrida(inia):
    assert len(inia) == 24                      # sin cabecera ni pie de pagina
    np.testing.assert_array_equal(inia["hora_reloj"].to_numpy(),
                                  np.arange(24) + 0.5)


# ---------- 6. tabla de 10 min vs calculo exacto ----------

def test_tabla_vs_exacto_puntos_medios(exp, inia, tabla):
    medios = np.arange(144) / 6 + 1 / 12
    aprox = perfil_radiacion(medios * 3600.0, 0.0, tabla)[2]
    np.testing.assert_allclose(aprox, referencia_exacta(exp, inia, medios),
                               rtol=0, atol=2.0)


def test_tabla_energia_vs_exacto(exp, inia, tabla):
    E_tabla = energia_Wh(lambda h: perfil_radiacion(h * 3600.0, 0.0, tabla)[2])
    E_exacta = energia_Wh(lambda h: referencia_exacta(exp, inia, h))
    assert E_tabla == pytest.approx(E_exacta, rel=1e-3)


def test_nodos_tabla_reproducen_inia(inia, tabla):
    t_k = inia["hora_reloj"].to_numpy()
    ghi_tabla = tabla.set_index("hora_reloj").loc[t_k, "ghi"].to_numpy()
    np.testing.assert_allclose(ghi_tabla, inia["ghi"].to_numpy(),
                               rtol=1e-10, atol=1e-12)


def test_grilla(tabla):
    assert list(tabla.columns) == ["hora_reloj", "dni", "dirh", "difh", "ghi",
                                   "T_amb_K", "v_viento_ms"]
    np.testing.assert_array_equal(tabla["hora_reloj"].to_numpy(),
                                  np.arange(144) / 6)


# ---------- 7. umbral con datos sinteticos ----------

def test_umbral_sintetico():
    """El umbral debe anular c aunque el INIA mida radiacion.

    Datos inventados para que el umbral sea lo unico que decide:
      nodo 05:30 -> I_exp = U/2 (bajo el umbral), I_inia = 40  -> c = 0
      nodo 06:30 -> I_exp = U   (justo en el umbral), I_inia = 0,8 U -> c = 0,8
      nodo 12:30 -> I_exp = 50 U (de dia), I_inia = 40 U            -> c = 0,8
    Sin umbral, el nodo 05:30 daria c = 40 / (U/2) != 0.
    """
    U = ds.UMBRAL_GHI
    ghi_exp = np.zeros(24)
    ghi_exp[6] = ghi_exp[7] = U           # 05:30 -> U/2 ; 06:30 -> U
    ghi_exp[12] = ghi_exp[13] = 50 * U    # 12:30 -> 50 U
    exp = pd.DataFrame({"hora_reloj": np.arange(24.0), "ghi": ghi_exp})

    ghi_inia = np.zeros(24)
    ghi_inia[5], ghi_inia[6], ghi_inia[12] = 40.0, 0.8 * U, 40 * U
    inia = pd.DataFrame({"hora_reloj": np.arange(24) + 0.5, "ghi": ghi_inia})

    t_k, c_k = factor_escala(exp, inia)
    assert c_k[5] == 0.0                                  # bajo el umbral: anulado
    np.testing.assert_allclose(c_k[6], 0.8, rtol=1e-12)   # en el umbral: se calcula
    np.testing.assert_allclose(c_k[12], 0.8, rtol=1e-12)  # control de dia
