# Modelo solar con cielo despejado mismo que Wu et al. (2026) -> ASHRAE

import numpy as np

from cryoevap.datos_solar import perfil_radiacion


def declinacion(d):
    """Declinación solar delta [grados]. d: día del año (1 = 1 de enero a 365 días).
    Wu et al. (2026), Ec. 10."""
    return 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + d)))

def ecuacion_tiempo(d):
    """Ecuación del tiempo EoT [min]. d: día del año (1 = 1 de enero).
    Fórmula estándar aproximada (tipo Spencer / Duffie-Beckman)."""
    B = np.radians(360.0 / 364.0 * (d - 81))
    return 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)


def hora_solar(t_reloj, d, lon_oeste, utc_offset=0.0):
    """Hora solar [h] a partir de una hora de reloj [h].
    t_reloj: hora medida en el huso UTC + utc_offset [h] (INIA: -4; DMC: 0).
    d: día del año (1 = 1 de enero).
    lon_oeste: longitud del lugar [grados], POSITIVA al oeste (Mejillones: 70.443).
    Fórmula de clases: t_solar = t_reloj + 4 min * (LME - lon_oeste) + EoT
    """
    LME = -15.0 * utc_offset                        # meridiano del huso [grados al oeste]
    correccion_longitud = 4.0 * (LME - lon_oeste)   # [min]: 4 min por grado
    correccion_EoT = ecuacion_tiempo(d)             # [min]
    correccion_total = correccion_longitud + correccion_EoT   # [min]
    return t_reloj + correccion_total / 60.0        # de min a h

def angulo_horario(t_solar):
    """Ángulo horario H [grados]. t_solar: hora solar [h].
    Negativo en la mañana, 0 al mediodía, positivo en la tarde."""
    return 15.0 * (t_solar - 12.0)


def altitud_solar(L, delta, H):
    """Altitud solar beta1 [grados]. L: latitud [grados], negativa en el
    hemisferio sur (Chile). Wu et al. (2026), Ec. 8."""
    L, delta, H = np.radians(L), np.radians(delta), np.radians(H)
    sin_b1 = np.cos(L) * np.cos(delta) * np.cos(H) + np.sin(L) * np.sin(delta)
    return np.degrees(np.arcsin(np.clip(sin_b1, -1, 1)))


def irradiancia_normal(A, B, beta1):
    """Irradiancia directa normal a los rayos I_DN [W/m2], cielo despejado.
    A [W/m2], B [-]: constantes ASHRAE del mes. beta1: altitud solar [grados].
    I_DN = A exp(-B / sin(beta1)). Wu et al. (2026), Ec. 5 (con k_L = 1).
    Vale 0 si el sol está bajo el horizonte (NOche)."""
    sin_b1 = np.sin(np.radians(beta1))
    arriba = sin_b1 > 0   # DIA. Si sen(b1) <= 0, el sol está bajo el horizonte (noche)
    seguro = np.where(arriba, sin_b1, 1.0)   # evita dividir por cero
    return np.where(arriba, A * np.exp(-B / seguro), 0.0)


def irradiancia_techo(A, B, C, beta1):
    """Irradiancia total sobre el techo (superficie horizontal) I_roof [W/m2].
    C [-]: razón difusa/directa. I_roof = I_DN (sin(beta1) + C).
    Wu et al. (2026), Ec. 2, 3 y 6 para superficie horizontal."""
    I_DN = irradiancia_normal(A, B, beta1)
    sin_b1 = np.maximum(np.sin(np.radians(beta1)), 0.0)
    return I_DN * (sin_b1 + C)


def irradiancia_pared(A, B, C, rho_G, beta1): # Revisar, si usar promedio o no, o separar por radiación, en función del tiempo
    """Irradiancia total sobre la pared vertical I_wall [W/m2], promediada
    sobre toda la circunferencia (modelo axisimétrico).
    Como no a todo el estanque le da el mismo sol, se promedia sobre toda la circunferencia. 
    A [W/m2], B [-]: constantes ASHRAE del mes. C [-]: razón difusa/directa.
    rho_G [-]: reflectividad del suelo. beta1: altitud solar [grados].
    I_wall = I_DN [cos(beta1)/pi + C/2 + rho_G/2 (sin(beta1) + C)]
    Directa: área proyectada del cilindro / área lateral.
    Difusa y reflejada: Wu et al. (2026), Ec. 3 y 4 con beta_n = 0."""
    I_DN = irradiancia_normal(A, B, beta1)
    sin_b1 = np.maximum(np.sin(np.radians(beta1)), 0.0)
    cos_b1 = np.cos(np.radians(beta1))
    return I_DN * (cos_b1 / np.pi + C / 2 + rho_G / 2 * (sin_b1 + C))

def beta1(t, t0, d=355, L=-23.1002, lon_oeste=70.4483, utc_offset=-4):
    """Altitud solar beta_1 (grados) en el instante de simulacion t (s).

    t0: hora de reloj (UTC-4, en horas) al inicio de la simulacion.
    d: dia del ano (355 = 21 de diciembre).
    """
    t_reloj = t0 + np.asarray(t, dtype=float) / 3600.0
    H = angulo_horario(hora_solar(t_reloj, d, lon_oeste, utc_offset))
    return altitud_solar(L, declinacion(d), H)


def rad_directa(t, t0, datos, **geo):
    """Radiacion directa (W/m2) sobre (pared, techo).

    Pared: I_DN cos(beta_1)/pi, promedio en toda la circunferencia.
    Techo: directa horizontal, ghi - difh. Ambas son 0 con el sol bajo el horizonte.
    """
    dni, difh, ghi = perfil_radiacion(t, t0, datos)
    b = beta1(t, t0, **geo)
    dia = b > 0
    pared = np.where(dia, dni * np.cos(np.radians(b)) / np.pi, 0.0)
    techo = np.where(dia, np.maximum(ghi - difh, 0.0), 0.0)
    return pared, techo

def rad_difusa(t, t0, datos):
    """Difusa (W/m2) sobre (pared, techo).

    Pared: difh/2 (factor de vista 1/2). Techo: difh (ve todo el cielo).
    """
    dni, difh, ghi = perfil_radiacion(t, t0, datos)
    return difh / 2, difh


def rad_reflejada(t, t0, datos, rho_G=0.2):
    """Reflejada del suelo (W/m2) sobre (pared, techo).

    Pared: rho_G/2 * ghi (la pared ve la mitad suelo). Techo: 0.
    """
    dni, difh, ghi = perfil_radiacion(t, t0, datos)
    return rho_G / 2 * ghi, np.zeros_like(ghi)


def I_pared(t, t0, datos, rho_G=0.2, **geo):
    """I_wall(t) = I_D + I_d + I_R sobre la pared (W/m2)."""
    dia = beta1(t, t0, **geo) > 0
    total = (rad_directa(t, t0, datos, **geo)[0]
             + rad_difusa(t, t0, datos)[0]
             + rad_reflejada(t, t0, datos, rho_G)[0])
    return np.where(dia, total, 0.0)


def I_techo(t, t0, datos, **geo):
    """I_roof(t) = I_D + I_d sobre el techo (W/m2)."""
    dia = beta1(t, t0, **geo) > 0
    total = (rad_directa(t, t0, datos, **geo)[1]
             + rad_difusa(t, t0, datos)[1])
    return np.where(dia, total, 0.0)