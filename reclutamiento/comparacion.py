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
    {"col": "location_plus", "nombre": "Location+", "unidad": "", "grupo": "Decisión", "radar": False},
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


def pitching_ponderado(repertorio: list[dict]) -> float:
    """Promedio de Pitching+ con los mismos pesos que el Stuff+."""
    con_valor = [fila for fila in repertorio if _finito(fila.get("pitching_plus"))]
    if not con_valor:
        return float("nan")
    return stuff_ponderado(
        [{**fila, "stuff_plus": fila["pitching_plus"]} for fila in con_valor]
    )


def location_ponderado(repertorio: list[dict]) -> float:
    """Promedio de Location+ con los mismos pesos que el Stuff+."""
    con_ubicacion = [fila for fila in repertorio if _finito(fila.get("location_plus"))]
    if not con_ubicacion:
        return float("nan")
    return stuff_ponderado(
        [{**fila, "stuff_plus": fila["location_plus"]} for fila in con_ubicacion]
    )


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
        "100 es el lanzamiento promedio de ese mismo tipo y 10 puntos son una desviación estándar dentro del tipo. "
        "Una recta ya no se compara contra un slider: cada uno tiene su propia escala. "
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
        frase = (
            f"El {tipo} de {mano} queda en {fila['stuff_plus']:.0f}, "
            f"contra una mediana de {mediana:.0f} (percentil {percentil:.0f}), {lugar}."
        )
        if _finito(fila.get("location_plus")) and "location_plus" in grupo.columns:
            percentil_loc = percentil_empirico(fila["location_plus"], grupo["location_plus"])
            mediana_loc = float(pd.to_numeric(grupo["location_plus"], errors="coerce").median())
            frase += (
                f" Su Location+ es {fila['location_plus']:.0f} "
                f"(mediana {mediana_loc:.0f}, percentil {percentil_loc:.0f}): "
                "eso mide la ubicación después de descontar el Stuff+."
            )
        detalles.append(frase)

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


def _pesos(filas: list[dict]) -> np.ndarray:
    usos = np.array([float(fila.get("uso_pct") or 0) for fila in filas], dtype=float)
    if np.nansum(usos) > 0:
        return usos
    cuentas = np.array([float(fila.get("n_lanzamientos") or 0) for fila in filas], dtype=float)
    if np.nansum(cuentas) > 0:
        return cuentas
    return np.ones(len(filas), dtype=float)


def representante(filas: list[dict]) -> dict:
    """Un pitcheo por tipo. Si hay varios, promedia con el mismo peso que el Stuff+."""
    if len(filas) == 1:
        return dict(filas[0])
    pesos = _pesos(filas)
    salida = dict(filas[0])
    columnas = (
        "RelSpeed",
        "EffectiveVelo",
        "ZoneSpeed",
        "SpinRate",
        "InducedVertBreak",
        "HorzBreak",
        "VertBreak",
        "Extension",
        "RelHeight",
        "RelSide",
        "VertApprAngle",
        "HorzApprAngle",
        "stuff_plus",
        "location_plus",
        "PlateLocSide",
        "PlateLocHeight",
    )
    for columna in columnas:
        valores = []
        pesos_validos = []
        for fila, peso in zip(filas, pesos):
            valor = fila.get(columna)
            if _finito(valor):
                valores.append(float(valor))
                pesos_validos.append(float(peso))
        salida[columna] = float(np.average(valores, weights=pesos_validos)) if valores else float("nan")
    salida["uso_pct"] = float(np.nansum([fila.get("uso_pct") or 0 for fila in filas]))
    return salida


def arsenal(repertorio: list[dict]) -> tuple[list[tuple], dict[tuple, dict]]:
    grupos: dict[tuple, list[dict]] = {}
    orden: list[tuple] = []
    for fila in repertorio:
        llave = (fila["AutoPitchType"], fila["PitcherThrows"])
        if llave not in grupos:
            orden.append(llave)
            grupos[llave] = []
        grupos[llave].append(fila)
    return orden, {llave: representante(grupos[llave]) for llave in orden}


def _mediana_cohorte(grupo: pd.DataFrame, columna: str) -> float:
    if grupo.empty or columna not in grupo.columns:
        return float("nan")
    muestra = pd.to_numeric(grupo[columna], errors="coerce").dropna()
    if muestra.empty:
        return float("nan")
    return float(muestra.median())


def _percentil_pitch(pitch: dict | None, grupo: pd.DataFrame, columna: str) -> float:
    if pitch is None or not _finito(pitch.get(columna)) or grupo.empty or columna not in grupo.columns:
        return float("nan")
    return percentil_empirico(pitch[columna], grupo[columna])


def filas_duelo(
    repertorio_a: list[dict],
    repertorio_b: list[dict],
    referencia: pd.DataFrame,
    minimo: int,
) -> list[dict]:
    """Una fila por pitcheo, con los dos reclutas y la mediana del equipo."""
    orden_a, arsenal_a = arsenal(repertorio_a)
    orden_b, arsenal_b = arsenal(repertorio_b)
    llaves = list(dict.fromkeys([*orden_a, *orden_b]))
    filas = []
    for tipo, mano in llaves:
        grupo = cohorte(referencia, tipo, mano, minimo)
        pitch_a = arsenal_a.get((tipo, mano))
        pitch_b = arsenal_b.get((tipo, mano))
        filas.append(
            {
                "tipo": tipo,
                "mano": mano,
                "a": pitch_a,
                "b": pitch_b,
                "n_equipo": int(len(grupo)),
                "mediana_stuff": _mediana_cohorte(grupo, "stuff_plus"),
                "mediana_location": _mediana_cohorte(grupo, "location_plus"),
                "mediana_velo": _mediana_cohorte(grupo, "RelSpeed"),
                "mediana_spin": _mediana_cohorte(grupo, "SpinRate"),
                "mediana_ivb": _mediana_cohorte(grupo, "InducedVertBreak"),
                "mediana_hb": _mediana_cohorte(grupo, "HorzBreak"),
                "percentil_stuff_a": _percentil_pitch(pitch_a, grupo, "stuff_plus"),
                "percentil_stuff_b": _percentil_pitch(pitch_b, grupo, "stuff_plus"),
                "percentil_location_a": _percentil_pitch(pitch_a, grupo, "location_plus"),
                "percentil_location_b": _percentil_pitch(pitch_b, grupo, "location_plus"),
                "percentil_velo_a": _percentil_pitch(pitch_a, grupo, "RelSpeed"),
                "percentil_velo_b": _percentil_pitch(pitch_b, grupo, "RelSpeed"),
                "grupo": grupo,
            }
        )
    return filas


def _puntos(cantidad: float) -> str:
    numero = int(round(abs(cantidad)))
    return f"{numero} punto" if numero == 1 else f"{numero} puntos"


def _frase_pitcheo(fila: dict, nombre_a: str, nombre_b: str) -> str:
    tipo = ETIQUETAS_TIPO.get(fila["tipo"], fila["tipo"])
    mano = ETIQUETAS_MANO.get(fila["mano"], fila["mano"])
    pitch_a = fila["a"]
    pitch_b = fila["b"]
    mediana = fila["mediana_stuff"]
    tiene_equipo = np.isfinite(mediana) and fila["n_equipo"] > 0

    if pitch_a and pitch_b:
        ventaja = float(pitch_a["stuff_plus"]) - float(pitch_b["stuff_plus"])
        if abs(ventaja) < 1:
            quien = f"{nombre_a} y {nombre_b} quedan parejos"
        elif ventaja > 0:
            quien = f"{nombre_a} va {_puntos(ventaja)} arriba de {nombre_b}"
        else:
            quien = f"{nombre_b} va {_puntos(ventaja)} arriba de {nombre_a}"
        frase = (
            f"En {tipo} de {mano}, {quien} "
            f"({nombre_a} {pitch_a['stuff_plus']:.0f}, {nombre_b} {pitch_b['stuff_plus']:.0f})."
        )
        if tiene_equipo and np.isfinite(fila["percentil_stuff_a"]) and np.isfinite(fila["percentil_stuff_b"]):
            frase += (
                f" Contra el equipo, {nombre_a} queda en el percentil {fila['percentil_stuff_a']:.0f} "
                f"y {nombre_b} en el {fila['percentil_stuff_b']:.0f} "
                f"(mediana {mediana:.0f}, {fila['n_equipo']} pitchers)."
            )
        return frase

    unico = pitch_a or pitch_b
    nombre = nombre_a if pitch_a else nombre_b
    percentil = fila["percentil_stuff_a"] if pitch_a else fila["percentil_stuff_b"]
    frase = f"Solo {nombre} trae {tipo} de {mano}, con Stuff+ {unico['stuff_plus']:.0f}."
    if tiene_equipo and np.isfinite(percentil):
        frase += (
            f" En el equipo ese pitcheo queda en el percentil {percentil:.0f} "
            f"(mediana {mediana:.0f})."
        )
    return frase


def texto_duelo(
    nombre_a: str,
    repertorio_a: list[dict],
    nombre_b: str,
    repertorio_b: list[dict],
    referencia: pd.DataFrame,
    minimo: int,
) -> str:
    """Lectura para elegir entre dos reclutas, también contra el roster."""
    if not repertorio_a and not repertorio_b:
        return "Carga el repertorio de los dos reclutas para ver quién queda más cerca del equipo."
    if not repertorio_a or not repertorio_b:
        presente = nombre_a if repertorio_a else nombre_b
        faltante = nombre_b if repertorio_a else nombre_a
        return (
            f"Ya está el repertorio de {presente}. "
            f"Captura a {faltante} en el panel izquierdo para compararlos entre sí y contra el equipo."
        )

    stuff_a = stuff_ponderado(repertorio_a)
    stuff_b = stuff_ponderado(repertorio_b)
    location_a = location_ponderado(repertorio_a)
    location_b = location_ponderado(repertorio_b)
    delta = stuff_a - stuff_b
    if abs(delta) < 1:
        cabeza = (
            f"{nombre_a} y {nombre_b} quedan parejos en Stuff+ del repertorio "
            f"({stuff_a:.0f} y {stuff_b:.0f})."
        )
    elif delta > 0:
        cabeza = (
            f"{nombre_a} queda {_puntos(delta)} de Stuff+ por encima de {nombre_b} "
            f"en el repertorio ({stuff_a:.0f} contra {stuff_b:.0f})."
        )
    else:
        cabeza = (
            f"{nombre_b} queda {_puntos(delta)} de Stuff+ por encima de {nombre_a} "
            f"en el repertorio ({stuff_b:.0f} contra {stuff_a:.0f})."
        )

    if np.isfinite(location_a) and np.isfinite(location_b):
        delta_loc = location_a - location_b
        if abs(delta_loc) < 1:
            cabeza += f" En Location+ también están parejos ({location_a:.0f} y {location_b:.0f})."
        elif delta_loc > 0:
            cabeza += f" En Location+, {nombre_a} va adelante ({location_a:.0f} contra {location_b:.0f})."
        else:
            cabeza += f" En Location+, {nombre_b} va adelante ({location_b:.0f} contra {location_a:.0f})."

    detalles = [
        _frase_pitcheo(fila, nombre_a, nombre_b)
        for fila in filas_duelo(repertorio_a, repertorio_b, referencia, minimo)
    ]
    return cabeza + " " + " ".join(detalles)
