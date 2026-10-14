"""Percentiles y z-scores del recluta contra los pitchers de la organización."""

from __future__ import annotations

import numpy as np
import pandas as pd

ETIQUETAS_TIPO = {
    "Four-Seam": "Four-Seam",
    "Sinker": "Sinker",
    "Slider": "Slider",
    "Curveball": "Curveball",
    "Changeup": "Changeup",
    "Cutter": "Cutter",
    "Splitter": "Splitter",
    "Sweeper": "Sweeper",
}

ETIQUETAS_MANO = {
    "Right": "derecho",
    "Left": "zurdo",
}

METRICAS = [
    {"col": "RelSpeed", "nombre": "Velocidad de salida", "unidad": "mph", "grupo": "Velocidad", "radar": True},
    {"col": "EffectiveVelo", "nombre": "Velocidad efectiva", "unidad": "mph", "grupo": "Velocidad", "radar": False},
    {"col": "ZoneSpeed", "nombre": "Velocidad en el plato", "unidad": "mph", "grupo": "Velocidad", "radar": False},
    {"col": "SpinRate", "nombre": "Spin rate", "unidad": "rpm", "grupo": "Rotación", "radar": True},
    {"col": "InducedVertBreak", "nombre": "Quiebre vertical inducido", "unidad": "in", "grupo": "Movimiento", "radar": True},
    {"col": "HorzBreak", "nombre": "Quiebre horizontal", "unidad": "in", "grupo": "Movimiento", "radar": True},
    {"col": "VertBreak", "nombre": "Quiebre vertical total", "unidad": "in", "grupo": "Movimiento", "radar": False},
    {"col": "Extension", "nombre": "Extensión", "unidad": "ft", "grupo": "Liberación", "radar": True},
    {"col": "RelHeight", "nombre": "Altura de liberación", "unidad": "ft", "grupo": "Liberación", "radar": False},
    {"col": "RelSide", "nombre": "Lado de liberación", "unidad": "ft", "grupo": "Liberación", "radar": False},
    {"col": "VertApprAngle", "nombre": "Ángulo vertical de aproximación", "unidad": "°", "grupo": "Ángulos", "radar": True},
    {"col": "HorzApprAngle", "nombre": "Ángulo horizontal de aproximación", "unidad": "°", "grupo": "Ángulos", "radar": False},
    {"col": "stuff_plus", "nombre": "Stuff+", "unidad": "", "grupo": "Decisión", "radar": False},
]


def cohorte(referencia: pd.DataFrame, tipo: str, mano: str, minimo: int) -> pd.DataFrame:
    """Pitchers reales del mismo tipo de lanzamiento y de la misma mano."""
    mascara = (
        (referencia["AutoPitchType"] == tipo)
        & (referencia["PitcherThrows"] == mano)
        & (referencia["n_lanzamientos"] >= minimo)
    )
    return referencia.loc[mascara].copy()


def _finito(valor) -> bool:
    if valor is None:
        return False
    try:
        return bool(np.isfinite(float(valor)))
    except (TypeError, ValueError):
        return False


def percentil_empirico(valor, distribucion: pd.Series) -> float:
    muestra = pd.to_numeric(distribucion, errors="coerce").dropna().to_numpy(dtype=float)
    if len(muestra) == 0 or not _finito(valor):
        return float("nan")
    return float((muestra <= float(valor)).mean() * 100.0)


def z_score(valor, distribucion: pd.Series) -> float:
    muestra = pd.to_numeric(distribucion, errors="coerce").dropna()
    if len(muestra) < 2 or not _finito(valor):
        return float("nan")
    sigma = float(muestra.std(ddof=0))
    if sigma < 1e-9:
        return float("nan")
    return (float(valor) - float(muestra.mean())) / sigma


def ficha_comparacion(recluta: dict, grupo: pd.DataFrame) -> pd.DataFrame:
    """Una fila por métrica: valor del recluta, mediana del roster y su lugar."""
    filas = []
    for metrica in METRICAS:
        columna = metrica["col"]
        valor = recluta.get(columna)
        if not _finito(valor):
            continue
        if columna not in grupo.columns or grupo.empty:
            dist = pd.Series(dtype=float)
        else:
            dist = grupo[columna]
        muestra = pd.to_numeric(dist, errors="coerce").dropna()
        filas.append(
            {
                "col": columna,
                "grupo": metrica["grupo"],
                "metrica": metrica["nombre"],
                "unidad": metrica["unidad"],
                "radar": metrica["radar"],
                "recluta": float(valor),
                "mediana": float(muestra.median()) if len(muestra) else float("nan"),
                "p25": float(muestra.quantile(0.25)) if len(muestra) else float("nan"),
                "p75": float(muestra.quantile(0.75)) if len(muestra) else float("nan"),
                "z": z_score(valor, muestra),
                "percentil": percentil_empirico(valor, muestra),
                "n": int(len(muestra)),
            }
        )
    return pd.DataFrame(filas)


def stuff_ponderado(repertorio: list[dict]) -> float:
    """Promedio de Stuff+ por uso, por cantidad de lanzamientos, o parejo."""
    if not repertorio:
        return float("nan")
    valores = np.array([float(fila["stuff_plus"]) for fila in repertorio], dtype=float)
    usos = np.array([float(fila.get("uso_pct") or 0) for fila in repertorio], dtype=float)
    if np.nansum(usos) > 0:
        pesos = usos
    else:
        cuentas = np.array(
            [float(fila.get("n_lanzamientos") or 0) for fila in repertorio],
            dtype=float,
        )
        pesos = cuentas if np.nansum(cuentas) > 0 else np.ones(len(repertorio))
    return float(np.average(valores, weights=pesos))


def banda_stuff(stuff: float) -> str:
    if not np.isfinite(stuff):
        return "Sin Stuff+"
    if stuff >= 110:
        return "Por encima del estándar"
    if stuff >= 90:
        return "En el estándar"
    return "Por debajo del estándar"


def _lugar_percentil(percentil: float) -> str:
    if percentil >= 70:
        return "por encima de la mayoría de sus pares"
    if percentil <= 30:
        return "por debajo de la mayoría de sus pares"
    return "en el centro del roster"


def texto_decision(repertorio: list[dict], referencia: pd.DataFrame, minimo: int) -> str:
    """Lectura para el coach: Stuff+ global y percentil contra el mismo tipo y mano."""
    if not repertorio:
        return "Agrega al menos un tipo de lanzamiento para ver la recomendación."

    global_stuff = stuff_ponderado(repertorio)
    lectura = (
        f"El repertorio queda en Stuff+ {global_stuff:.0f}. "
        "100 es el lanzamiento promedio de toda la organización y 10 puntos son una desviación estándar. "
        "Un slider suele vivir arriba de 100 y un sinker abajo, porque no producen el mismo whiff. "
        "Para decidir si el recluta compite con el roster, el número que importa es el percentil "
        "contra pitchers del mismo tipo y de la misma mano."
    )

    percentiles = []
    detalles = []
    cerca_de_la_mediana = []
    for fila in sorted(repertorio, key=lambda item: item["stuff_plus"], reverse=True):
        tipo = ETIQUETAS_TIPO.get(fila["AutoPitchType"], fila["AutoPitchType"])
        mano = ETIQUETAS_MANO.get(fila["PitcherThrows"], fila["PitcherThrows"])
        grupo = cohorte(referencia, fila["AutoPitchType"], fila["PitcherThrows"], minimo)
        if grupo.empty or "stuff_plus" not in grupo.columns:
            detalles.append(
                f"El {tipo} califica en {fila['stuff_plus']:.0f}, sin suficientes pares para un percentil."
            )
            continue
        percentil = percentil_empirico(fila["stuff_plus"], grupo["stuff_plus"])
        mediana = float(grupo["stuff_plus"].median())
        percentiles.append(percentil)
        cerca = abs(fila["stuff_plus"] - mediana) < 3
        cerca_de_la_mediana.append(cerca)
        if cerca:
            lugar = "junto a la mediana de ese grupo"
        else:
            lugar = _lugar_percentil(percentil)
        detalles.append(
            f"El {tipo} de {mano} queda en {fila['stuff_plus']:.0f}, "
            f"contra una mediana de {mediana:.0f} (percentil {percentil:.0f}), {lugar}."
        )

    if cerca_de_la_mediana and all(cerca_de_la_mediana):
        cierre = (
            " En puntos de Stuff+, el reporte queda al lado de la mediana del roster. "
            "Dentro de un mismo tipo los pitchers de la organización están muy juntos, "
            "así que el percentil se mueve con pocos puntos."
        )
    elif percentiles and min(percentiles) >= 60:
        cierre = " El material, pitcheo por pitcheo, está a la altura del roster actual o por encima."
    elif percentiles and max(percentiles) >= 75 and min(percentiles) <= 40:
        cierre = " El repertorio es desigual: hay un pitcheo que sí compite y otro que se queda corto."
    elif percentiles and max(percentiles) <= 40:
        cierre = " Ningún pitcheo del reporte alcanza el centro del roster actual en su Stuff+."
    else:
        cierre = " Conviene leer cada pitcheo por separado antes de cerrar la evaluación."

    aviso_velocidad = []
    for fila in repertorio:
        grupo_fila = cohorte(referencia, fila["AutoPitchType"], fila["PitcherThrows"], minimo)
        ficha = ficha_comparacion(fila, grupo_fila)
        velo = ficha.loc[ficha["col"] == "RelSpeed"]
        if velo.empty or not np.isfinite(velo["percentil"].iloc[0]):
            continue
        percentil_velo = float(velo["percentil"].iloc[0])
        grupo_stuff = grupo_fila["stuff_plus"] if not grupo_fila.empty else pd.Series(dtype=float)
        percentil_stuff = percentil_empirico(fila["stuff_plus"], grupo_stuff)
        if percentil_velo >= 70 and np.isfinite(percentil_stuff) and percentil_stuff <= 40:
            aviso_velocidad.append(ETIQUETAS_TIPO.get(fila["AutoPitchType"], fila["AutoPitchType"]))
    aviso = ""
    if aviso_velocidad:
        aviso = (
            " En "
            + ", ".join(aviso_velocidad)
            + " la velocidad está arriba del roster, pero el Stuff+ no: el giro y el movimiento "
            "no convierten esa velocidad en la probabilidad de whiff del modelo."
        )

    return lectura + " " + " ".join(detalles) + cierre + aviso
