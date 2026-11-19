"""Figura: radiacion horaria sobre pared y techo (21 dic, Mejillones).

Compara la tabla mezclada (INIA 2025 + Explorador Solar 2016) con el
Explorador Solar puro. Eje x: hora de reloj (UTC-4) desde t0 = 0.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin ventana, solo archivo
import matplotlib.pyplot as plt

# Raiz del proyecto calculada desde este archivo
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from cryoevap.datos_solar import cargar_datos, cargar_inia, mezclar
from cryoevap.solar import I_pared, I_techo

RUTA_EXP = RAIZ / "data" / "explorador_solar_mejillones_2016-12-21.csv"
RUTA_INIA = RAIZ / "data" / "mejillones_inia_2025-12-21.csv"
RUTA_FIG = RAIZ / "figs" / "fig_radiacion_pared_techo.pdf"

# Caso: 21 de diciembre, de 0 a 24 h cada 10 min
T0 = 0.0                                   # hora de reloj inicial [h]
D = 355                                    # dia del ano (21 dic)
RHO_G = 0.2                                # albedo del suelo [-]
t = np.arange(0, 86400 + 600, 600)         # [s]


def main():
    datos_exp = cargar_datos(RUTA_EXP)
    datos_inia = cargar_inia(RUTA_INIA)
    datos_mez = mezclar(datos_exp, datos_inia)

    # Misma geometria (d) y albedo para ambas tablas
    series = {}
    for nombre, tabla in [("mezclada", datos_mez), ("explorador", datos_exp)]:
        series[nombre] = (
            I_pared(t, T0, tabla, rho_G=RHO_G, d=D),
            I_techo(t, T0, tabla, d=D),
        )

    h = t / 3600.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h, series["mezclada"][0], "C0-", lw=2,
            label="Pared, INIA 2025")
    ax.plot(h, series["mezclada"][1], "-", color="#C49A00", lw=2,  # mostaza
            label="Techo, INIA 2025")
    ax.plot(h, series["explorador"][0], "C0--", lw=1.5,
            label="Pared, Explorador 2016")
    ax.plot(h, series["explorador"][1], "--", color="#C49A00", lw=1.5,
            label="Techo, Explorador 2016")

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hora (UTC-4) [h]", fontsize=12)
    ax.set_ylabel("Radiación solar [W/m$^2$]", fontsize=12)
    ax.set_title("Pared y techo, 21 de diciembre (Mejillones)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")  # entre 0 y 6 h no hay curvas

    # Nota de fuentes bajo el gráfico
    fig.text(0.5, 0.01,
             "Magnitud: INIA 2025. Reparto directa/difusa: Explorador Solar "
             "2016 (se supone igual en 2025)",
             ha="center", va="bottom", fontsize=11, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # deja espacio abajo para la nota

    RUTA_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIG, format="pdf")
    print(f"Figura guardada en {RUTA_FIG}")


if __name__ == "__main__":
    main()
