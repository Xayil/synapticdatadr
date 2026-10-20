"""Física del lanzamiento tomada de modelos/Modelo_general_STUFF+.ipynb.

El notebook integra gravedad, drag y Magnus con solve_ivp y define

    HB = y_C(t_plato) - y_B(t_plato)
    IVB = z_C(t_plato) - z_B(t_plato)

junto con el factor de spin S = R |ω × v| / |v|² y C_L(S).

Para poder calificar cientos de miles de lanzamientos, el quiebre inducido
se aproxima con aceleración de Magnus constante (las mismas F_M, S, C_L y ρ).
Si el eje de giro es perpendicular a la velocidad, S se reduce a R ω / V,
que es el caso que usa el notebook cuando ω ⊥ v.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Constantes del notebook
G = 9.80665
M_BOLA = 0.145
R_BOLA = 0.0367
A_BOLA = float(np.pi * R_BOLA**2)
CD = 0.5
R_AIR = 287.05

P0 = 101325.0
T0 = 288.15
L_ISA = 0.0065
M_MOLAR = 0.02895644
R_UNIV = 8.31447

MPH_A_MS = 0.44704
M_A_PULGADAS = 39.3701
FT_A_M = 0.3048
DISTANCIA_MONTE_FT = 60.5

# La base histórica solo trae tres categorías. El recluta se califica con la
# altitud real del estadio, y ρ sale de la atmósfera estándar a esos metros.
ALTITUD_M = {
    "No Altitude": 0.0,
    "Medium Altitude": 1200.0,
    "Extreme Altitude": 2240.0,
}

ESTADIOS = [
    {"id": "alfredo_harp_helu", "nombre": "Alfredo Harp Helú", "metros": 2232, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "hermanos_serdan", "nombre": "Hermanos Serdán", "metros": 2192, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "alberto_romo_chavez", "nombre": "Alberto Romo Chávez", "metros": 1921, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "francisco_villa", "nombre": "Francisco Villa", "metros": 1906, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "domingo_santana", "nombre": "Domingo Santana", "metros": 1820, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "finsus", "nombre": "Finsus", "metros": 1820, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "coors_field", "nombre": "Coors Field", "metros": 1610, "clase": "Extreme", "aprox": False, "lugar": "Colorado", "detalle": "5,280 ft"},
    {"id": "yuva_vasconcelos", "nombre": "Yu'Va / Eduardo Vasconcelos", "metros": 1566, "clase": "Extreme", "aprox": True, "lugar": ""},
    {"id": "francisco_madero", "nombre": "Francisco I. Madero", "metros": 1565, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "panamericano", "nombre": "Panamericano", "metros": 1555, "clase": "Extreme", "aprox": False, "lugar": ""},
    {"id": "monumental_chihuahua", "nombre": "Monumental Chihuahua", "metros": 1415, "clase": "Medium", "aprox": False, "lugar": ""},
    {"id": "revolucion", "nombre": "Revolución", "metros": 1135, "clase": "Medium", "aprox": False, "lugar": ""},
]
for _estadio in ESTADIOS:
    ALTITUD_M[_estadio["id"]] = float(_estadio["metros"])

ESTADIO_DEFAULT = "alfredo_harp_helu"
ALTITUD_DEFAULT = ESTADIO_DEFAULT


def etiqueta_estadio(estadio: dict) -> str:
    """Nombre, metros sobre el nivel del mar y clasificación."""
    metros = f"{estadio['metros']:,}"
    if estadio.get("aprox"):
        metros = f"≈{metros}"
    texto = f"{metros} msnm"
    if estadio.get("detalle"):
        texto = f"{texto} ({estadio['detalle']})"
    lugar = f" · {estadio['lugar']}" if estadio.get("lugar") else ""
    return f"{estadio['nombre']}{lugar} · {texto} · {estadio['clase']}"


def presion_isa(altura_m: float | np.ndarray) -> float | np.ndarray:
    """P(h) = P0 (1 - L h / T0) ^ (g M / (R L))."""
    h = np.asarray(altura_m, dtype=float)
    base = 1.0 - (L_ISA * h) / T0
    base = np.clip(base, 1e-6, None)
    exponente = (G * M_MOLAR) / (R_UNIV * L_ISA)
    return P0 * np.power(base, exponente)


def temperatura_isa_c(altura_m: float | np.ndarray) -> float | np.ndarray:
    """Temperatura de la atmósfera estándar, en °C."""
    return (T0 - L_ISA * np.asarray(altura_m, dtype=float)) - 273.15


def densidad_aire(presion_pa, temperatura_c):
    """ρ = P / (R_air T), con T en kelvin."""
    t_k = np.asarray(temperatura_c, dtype=float) + 273.15
    return np.asarray(presion_pa, dtype=float) / (R_AIR * t_k)


def densidad_por_altitud(categoria) -> float | np.ndarray:
    """Densidad ISA a partir de altitude_category."""
    if isinstance(categoria, str) or categoria is None or (isinstance(categoria, float) and np.isnan(categoria)):
        nombre = ALTITUD_DEFAULT if categoria is None or (isinstance(categoria, float) and np.isnan(categoria)) else str(categoria)
        altura = ALTITUD_M.get(nombre, ALTITUD_M[ALTITUD_DEFAULT])
        return float(densidad_aire(presion_isa(altura), temperatura_isa_c(altura)))

    nombres = pd.Series(categoria).fillna(ALTITUD_DEFAULT).astype(str)
    alturas = nombres.map(ALTITUD_M).fillna(ALTITUD_M[ALTITUD_DEFAULT]).to_numpy(dtype=float)
    return densidad_aire(presion_isa(alturas), temperatura_isa_c(alturas))


def mph_a_ms(v_mph):
    return np.asarray(v_mph, dtype=float) * MPH_A_MS


def rpm_a_rad_s(rpm):
    return np.asarray(rpm, dtype=float) * 2.0 * np.pi / 60.0


def coeficiente_lift(factor_s):
    """C_L(S) = 1.5 S si S ≤ 0.1; 0.09 + 0.6 S si S > 0.1."""
    s = np.asarray(factor_s, dtype=float)
    return np.where(s <= 0.1, 1.5 * s, 0.09 + 0.6 * s)


def factor_spin(rpm, v_mph):
    """S = R ω / V cuando el eje es perpendicular a la velocidad."""
    v = np.maximum(mph_a_ms(v_mph), 1e-6)
    return R_BOLA * rpm_a_rad_s(rpm) / v


def tilt_a_spin_axis(tilt: str) -> float:
    """Convierte un tilt de reloj ('11:30') al SpinAxis de esta base.

    En los datos, horas = ((SpinAxis + 180) mód 360) / 30.
    """
    texto = str(tilt).strip().replace(".", ":")
    horas_txt, _, minutos_txt = texto.partition(":")
    horas = float(horas_txt)
    minutos = float(minutos_txt or 0)
    horas_dec = (horas % 12) + minutos / 60.0
    return (horas_dec * 30.0 - 180.0) % 360.0


def spin_axis_a_tilt(spin_axis) -> str:
    """Representación de reloj, redondeada a 15 minutos."""
    eje = float(spin_axis) % 360.0
    horas_dec = ((eje + 180.0) % 360.0) / 30.0
    horas_dec = round(horas_dec * 4.0) / 4.0
    horas_dec = horas_dec % 12.0
    horas = int(horas_dec)
    minutos = int(round((horas_dec - horas) * 60.0))
    if minutos == 60:
        horas = (horas + 1) % 12
        minutos = 0
    if horas == 0:
        horas = 12
    return f"{horas}:{minutos:02d}"


def _direccion_magnus(spin_axis_grados):
    """Dirección de ω × v en el plano del plato para el eje de TrackMan.

    180° (tilt 12:00) es backspin: Magnus hacia arriba.
    El signo horizontal sigue el HorzBreak de esta base (positivo = lado
    del brazo de un derecho).
    """
    phi = np.deg2rad(np.asarray(spin_axis_grados, dtype=float))
    vertical = -np.cos(phi)
    horizontal = -np.sin(phi)
    return vertical, horizontal


def quiebre_modelo(rel_speed, spin_rate, spin_axis, extension, rho, zone_speed=None):
    """IVB y HB (pulgadas) que predice el modelo de Magnus.

    Usa a_M = (ρ A C_L / 2m) |v|² en la dirección del eje, integrada con
    aceleración constante hasta el plato: desplazamiento = ½ a t².
    """
    v_rel = np.maximum(mph_a_ms(rel_speed), 1e-6)
    if zone_speed is None:
        v_zona = v_rel * 0.93
    else:
        v_zona = mph_a_ms(zone_speed)
        v_zona = np.where(np.isfinite(v_zona), np.maximum(v_zona, 1e-6), v_rel * 0.93)
    v_media = 0.5 * (v_rel + v_zona)

    s_factor = R_BOLA * rpm_a_rad_s(spin_rate) / v_media
    c_lift = coeficiente_lift(s_factor)
    rho_arr = np.asarray(rho, dtype=float)
    a_magnus = (rho_arr * A_BOLA * c_lift) / (2.0 * M_BOLA) * v_media**2

    extension_arr = np.asarray(extension, dtype=float)
    distancia_m = np.maximum((DISTANCIA_MONTE_FT - extension_arr) * FT_A_M, 0.5)
    t_vuelo = distancia_m / v_media
    desplazamiento_m = 0.5 * a_magnus * t_vuelo**2

    vertical, horizontal = _direccion_magnus(spin_axis)
    ivb = desplazamiento_m * vertical * M_A_PULGADAS
    hb = desplazamiento_m * horizontal * M_A_PULGADAS
    return ivb, hb


def signo_lado_brazo(pitcher_throws) -> np.ndarray:
    """+1 para derechos, -1 para zurdos.

    En esta base el HorzBreak positivo corre al lado del brazo de un derecho
    y el negativo al lado del brazo de un zurdo. El producto deja el movimiento
    horizontal en un solo marco: positivo = lado del brazo.
    """
    mano = pd.Series(pitcher_throws).astype(str).str.strip().str.lower()
    return np.where(mano.isin(["left", "l", "zurdo", "izquierda", "izq"]), -1.0, 1.0)
