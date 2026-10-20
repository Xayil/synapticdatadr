"""Sistema de reclutamiento para el coach.

El recluta se captura a mano o con un CSV de TrackMan/Rapsodo. Cada tipo de
lanzamiento se compara con los pitchers reales de la misma mano y se califica
con el Stuff+ definido en modelos/Modelo_general_STUFF+.ipynb.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reclutamiento.comparacion import (
    ETIQUETAS_MANO,
    ETIQUETAS_TIPO,
    banda_stuff,
    cohorte,
    ficha_comparacion,
    location_ponderado,
    stuff_ponderado,
    texto_decision,
)
from reclutamiento.fisica import (
    ESTADIO_DEFAULT,
    ESTADIOS,
    etiqueta_estadio,
    spin_axis_a_tilt,
    tilt_a_spin_axis,
)
from reclutamiento.location_plus import grilla_location
from reclutamiento.stuff_plus import (
    TIPOS_MODELO,
    cargar_modelo,
    cargar_referencia,
    puntuar_lanzamiento,
)

ROJO = "#8E1B1B"
TINTA = "#1C1917"
PIEDRA = "#78716C"
CREMA = "#F6F3EE"

ALTITUDES = {estadio["id"]: etiqueta_estadio(estadio) for estadio in ESTADIOS}

NOMBRES_BETA = {
    "RelSpeed": "Velocidad de salida",
    "SpinRate": "Spin rate",
    "InducedVertBreak": "IVB",
    "HB_mano": "Quiebre horizontal, lado del brazo",
    "delta_ivb": "ΔIVB (real − físico)",
    "delta_hb": "ΔHB (real − físico)",
    "S": "Factor de spin S",
    "Extension": "Extensión",
    "RelHeight": "Altura de liberación",
    "RelSide_mano": "Release side, lado del brazo",
    "rho": "Densidad del aire",
}


def clave(texto: str) -> str:
    return "".join(caracter for caracter in str(texto).strip().lower() if caracter.isalnum())


def _alias(*nombres: str) -> dict[str, str]:
    canonico = nombres[0]
    return {clave(nombre): canonico for nombre in nombres}


ALIAS_COLUMNAS = {}
for grupo in (
    _alias("AutoPitchType", "pitch type", "pitchtype", "tagged pitch type", "auto pitch type"),
    _alias("PitcherThrows", "throws", "pitcher throws", "mano", "p_throws"),
    _alias("RelSpeed", "release speed", "velocity", "velo", "rel speed"),
    _alias("EffectiveVelo", "effective velocity", "effective velo"),
    _alias("ZoneSpeed", "zone speed", "plate speed"),
    _alias("SpinRate", "spin rate", "spinrate"),
    _alias("SpinAxis", "spin axis", "spinaxis"),
    _alias("Tilt", "tilt"),
    _alias("RelHeight", "release height", "rel height"),
    _alias("RelSide", "release side", "rel side"),
    _alias("Extension", "extension"),
    _alias("InducedVertBreak", "induced vert break", "ivb", "induced vertical break"),
    _alias("HorzBreak", "horz break", "horizontal break", "hb", "pfx x"),
    _alias("VertBreak", "vert break", "vertical break"),
    _alias("VertApprAngle", "vert appr angle", "vaa", "vertical approach angle"),
    _alias("HorzApprAngle", "horz appr angle", "haa", "horizontal approach angle"),
    _alias("altitude_category", "altitud", "altitude", "estadio", "stadium"),
    _alias("uso_pct", "uso", "usage"),
    _alias("PlateLocSide", "plate loc side", "plate location side", "platelocside", "px"),
    _alias("PlateLocHeight", "plate loc height", "plate location height", "platelocheight", "pz"),
    _alias("Balls", "bolas", "balls"),
    _alias("Strikes", "strikes", "strike"),
    _alias("BatterSide", "batter side", "batterside", "lado del bateador", "mano del bateador"),
    _alias("count", "cuenta", "conteo"),
):
    ALIAS_COLUMNAS.update(grupo)

ALIAS_TIPO = {
    "fourseam": "Four-Seam",
    "fourseamfastball": "Four-Seam",
    "4seam": "Four-Seam",
    "ff": "Four-Seam",
    "fastball": "Four-Seam",
    "sinker": "Sinker",
    "si": "Sinker",
    "twoseamfastball": "Sinker",
    "slider": "Slider",
    "sl": "Slider",
    "curveball": "Curveball",
    "curve": "Curveball",
    "cu": "Curveball",
    "cb": "Curveball",
    "changeup": "Changeup",
    "change": "Changeup",
    "ch": "Changeup",
    "cutter": "Cutter",
    "fc": "Cutter",
    "splitter": "Splitter",
    "split": "Splitter",
    "fs": "Splitter",
    "sweeper": "Sweeper",
    "st": "Sweeper",
}
for tipo in TIPOS_MODELO:
    ALIAS_TIPO.setdefault(clave(tipo), tipo)

ALIAS_MANO = {
    "right": "Right",
    "r": "Right",
    "derecha": "Right",
    "derecho": "Right",
    "diestro": "Right",
    "left": "Left",
    "l": "Left",
    "izquierda": "Left",
    "izquierdo": "Left",
    "zurdo": "Left",
    "zurda": "Left",
}

ALIAS_ALTITUD = {
    "extremealtitude": "Extreme Altitude",
    "extrema": "Extreme Altitude",
    "cdmx": ESTADIO_DEFAULT,
    "mediumaltitude": "Medium Altitude",
    "media": "Medium Altitude",
    "noaltitude": "No Altitude",
    "niveldelmar": "No Altitude",
}
for _estadio in ESTADIOS:
    ALIAS_ALTITUD[clave(_estadio["id"])] = _estadio["id"]
    ALIAS_ALTITUD[clave(_estadio["nombre"])] = _estadio["id"]
ALIAS_ALTITUD[clave("Coors Field Colorado")] = "coors_field"
ALIAS_ALTITUD[clave("YuVa Eduardo Vasconcelos")] = "yuva_vasconcelos"


def numero(etiqueta, key, ayuda, minimo, maximo, paso, formato="%.2f"):
    return st.number_input(
        etiqueta,
        min_value=float(minimo),
        max_value=float(maximo),
        value=None,
        step=float(paso),
        format=formato,
        help=ayuda,
        key=key,
    )


def opcional(valor):
    if valor is None:
        return None
    try:
        numero_valor = float(valor)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numero_valor):
        return None
    return numero_valor


def resolver_eje(spin_axis, tilt) -> float:
    if opcional(spin_axis) is not None:
        return float(spin_axis)
    if tilt is None or not str(tilt).strip():
        raise ValueError("Falta el eje de giro: captura SpinAxis o un tilt de reloj, por ejemplo 11:30.")
    try:
        return float(tilt_a_spin_axis(str(tilt).strip()))
    except ValueError as exc:
        raise ValueError("El tilt tiene que venir como reloj, por ejemplo 1:30 o 11:15.") from exc


def construir_lanzamiento(campos: dict, modelo: dict) -> dict:
    eje = resolver_eje(campos.get("SpinAxis"), campos.get("Tilt"))
    lanzamiento = {
        "AutoPitchType": campos["AutoPitchType"],
        "PitcherThrows": campos["PitcherThrows"],
        "altitude_category": campos.get("altitude_category") or ESTADIO_DEFAULT,
        "PlateLocSide": opcional(campos.get("PlateLocSide")),
        "PlateLocHeight": opcional(campos.get("PlateLocHeight")),
        "Balls": opcional(campos.get("Balls")),
        "Strikes": opcional(campos.get("Strikes")),
        "BatterSide": campos.get("BatterSide") or None,
        "count": campos.get("count"),
        "RelSpeed": float(campos["RelSpeed"]),
        "EffectiveVelo": opcional(campos.get("EffectiveVelo")),
        "ZoneSpeed": opcional(campos.get("ZoneSpeed")),
        "SpinRate": float(campos["SpinRate"]),
        "SpinAxis": eje,
        "Tilt": spin_axis_a_tilt(eje),
        "RelHeight": float(campos["RelHeight"]),
        "RelSide": float(campos["RelSide"]),
        "Extension": float(campos["Extension"]),
        "InducedVertBreak": float(campos["InducedVertBreak"]),
        "HorzBreak": float(campos["HorzBreak"]),
        "VertBreak": opcional(campos.get("VertBreak")),
        "VertApprAngle": opcional(campos.get("VertApprAngle")),
        "HorzApprAngle": opcional(campos.get("HorzApprAngle")),
        "uso_pct": opcional(campos.get("uso_pct")),
        "n_lanzamientos": opcional(campos.get("n_lanzamientos")),
    }
    lanzamiento.update(puntuar_lanzamiento(lanzamiento, modelo))
    lanzamiento["id"] = int(st.session_state.siguiente_id)
    st.session_state.siguiente_id += 1
    return lanzamiento


def renombrar_csv(datos: pd.DataFrame) -> pd.DataFrame:
    mapa = {}
    for columna in datos.columns:
        canonico = ALIAS_COLUMNAS.get(clave(columna))
        if canonico:
            mapa[columna] = canonico
    return datos.rename(columns=mapa)


def normalizar_tipo(valor) -> str:
    texto = ALIAS_TIPO.get(clave(valor))
    if texto is None:
        raise ValueError(f"No reconozco el tipo de lanzamiento '{valor}'.")
    return texto


def normalizar_mano(valor) -> str:
    texto = ALIAS_MANO.get(clave(valor))
    if texto is None:
        raise ValueError(f"No reconozco la mano '{valor}'. Usa Right o Left.")
    return texto


def media_circular(grados: pd.Series) -> float:
    radianes = np.deg2rad(pd.to_numeric(grados, errors="coerce").dropna().to_numpy(dtype=float))
    if len(radianes) == 0:
        return float("nan")
    angulo = np.rad2deg(np.arctan2(np.sin(radianes).mean(), np.cos(radianes).mean()))
    return float(angulo % 360.0)


def lanzamientos_desde_csv(archivo, altitud_default: str, mano_default: str, modelo: dict) -> list[dict]:
    crudo = pd.read_csv(archivo)
    datos = renombrar_csv(crudo)
    if "AutoPitchType" not in datos.columns:
        raise ValueError("El archivo necesita una columna AutoPitchType (o Pitch Type).")

    datos = datos.dropna(subset=["AutoPitchType"])
    datos["AutoPitchType"] = datos["AutoPitchType"].map(normalizar_tipo)
    if "PitcherThrows" in datos.columns:
        datos["PitcherThrows"] = datos["PitcherThrows"].fillna(mano_default).map(normalizar_mano)
    else:
        datos["PitcherThrows"] = mano_default

    if "altitude_category" in datos.columns:
        datos["altitude_category"] = datos["altitude_category"].map(
            lambda valor: ALIAS_ALTITUD.get(clave(valor), altitud_default)
        )
    else:
        datos["altitude_category"] = altitud_default

    if "SpinAxis" not in datos.columns:
        datos["SpinAxis"] = np.nan
    if "Tilt" in datos.columns:
        def eje_desde_tilt(valor):
            if pd.isna(valor) or not str(valor).strip():
                return np.nan
            try:
                return tilt_a_spin_axis(str(valor).strip())
            except ValueError:
                return np.nan

        ejes_tilt = datos["Tilt"].map(eje_desde_tilt)
        datos["SpinAxis"] = pd.to_numeric(datos["SpinAxis"], errors="coerce").fillna(ejes_tilt)

    numericas = [
        "RelSpeed",
        "EffectiveVelo",
        "ZoneSpeed",
        "SpinRate",
        "RelHeight",
        "RelSide",
        "Extension",
        "InducedVertBreak",
        "HorzBreak",
        "VertBreak",
        "VertApprAngle",
        "HorzApprAngle",
        "PlateLocSide",
        "PlateLocHeight",
        "Balls",
        "Strikes",
        "uso_pct",
    ]
    for columna in numericas:
        if columna in datos.columns:
            datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    faltantes = [col for col in ("RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak", "Extension", "RelHeight", "RelSide") if col not in datos.columns]
    if faltantes:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltantes) + ".")

    grupos = []
    avisos = []
    claves = ["AutoPitchType", "PitcherThrows", "altitude_category"]
    for llave, bloque in datos.groupby(claves, dropna=False):
        fila = {nombre: valor for nombre, valor in zip(claves, llave)}
        fila["n_lanzamientos"] = int(len(bloque))
        fila["SpinAxis"] = media_circular(bloque["SpinAxis"])
        for columna in numericas:
            if columna in bloque.columns:
                fila[columna] = bloque[columna].mean(skipna=True)
        if "uso_pct" in bloque.columns and bloque["uso_pct"].notna().any():
            fila["uso_pct"] = float(bloque["uso_pct"].mean(skipna=True))
        if "BatterSide" in bloque.columns:
            manos_bate = bloque["BatterSide"].map(lambda valor: ALIAS_MANO.get(clave(valor)))
            manos_bate = manos_bate.dropna()
            if not manos_bate.empty:
                fila["BatterSide"] = manos_bate.mode().iloc[0]
        if "count" in bloque.columns and (fila.get("Balls") is None or fila.get("Strikes") is None):
            cuentas = bloque["count"].dropna().astype(str).str.strip()
            if not cuentas.empty:
                fila["count"] = cuentas.mode().iloc[0]
        try:
            grupos.append(construir_lanzamiento(fila, modelo))
        except ValueError as exc:
            avisos.append(f"{fila['AutoPitchType']} {fila['PitcherThrows']}: {exc}")
    if not grupos:
        detalle = " ".join(avisos) if avisos else "El archivo no tiene lanzamientos para agregar."
        raise ValueError(detalle)
    st.session_state.csv_avisos = avisos
    return grupos


def plantilla_csv() -> str:
    columnas = [
        "AutoPitchType",
        "PitcherThrows",
        "altitude_category",
        "RelSpeed",
        "EffectiveVelo",
        "ZoneSpeed",
        "SpinRate",
        "SpinAxis",
        "Tilt",
        "RelHeight",
        "RelSide",
        "Extension",
        "InducedVertBreak",
        "HorzBreak",
        "VertBreak",
        "VertApprAngle",
        "HorzApprAngle",
        "PlateLocSide",
        "PlateLocHeight",
        "Balls",
        "Strikes",
        "BatterSide",
        "uso_pct",
    ]
    return ",".join(columnas) + "\n"


def perfil_mediano(referencia: pd.DataFrame, modelo: dict) -> list[dict]:
    """Dos pitcheos con la mediana del roster derecho, para recorrer la herramienta."""
    usos = {"Four-Seam": 55.0, "Slider": 45.0}
    filas = []
    for tipo, uso in usos.items():
        grupo = cohorte(referencia, tipo, "Right", 30)
        if grupo.empty:
            continue
        mediana = grupo.median(numeric_only=True)
        campos = {
            "AutoPitchType": tipo,
            "PitcherThrows": "Right",
            "altitude_category": ESTADIO_DEFAULT,
            "Balls": 1,
            "Strikes": 1,
            "BatterSide": "Right",
            "uso_pct": uso,
            "n_lanzamientos": None,
        }
        for columna in (
            "RelSpeed",
            "EffectiveVelo",
            "ZoneSpeed",
            "SpinRate",
            "SpinAxis",
            "RelHeight",
            "RelSide",
            "Extension",
            "InducedVertBreak",
            "HorzBreak",
            "VertBreak",
            "VertApprAngle",
            "HorzApprAngle",
            "PlateLocSide",
            "PlateLocHeight",
        ):
            campos[columna] = mediana.get(columna)
        filas.append(construir_lanzamiento(campos, modelo))
    return filas


def estilo(figura: go.Figure, titulo: str | None = None) -> go.Figure:
    figura.update_layout(
        title=dict(text=titulo or "", font=dict(size=16, color=TINTA)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", color=TINTA, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=56, b=10),
    )
    figura.update_xaxes(showgrid=True, gridcolor="#EFEAE3", zerolinecolor="#D6D3D1")
    figura.update_yaxes(showgrid=True, gridcolor="#EFEAE3", zerolinecolor="#D6D3D1")
    return figura


def figura_radar(ficha: pd.DataFrame) -> go.Figure | None:
    radar = ficha[ficha["radar"] & ficha["percentil"].notna()]
    if len(radar) < 3:
        return None
    categorias = radar["metrica"].tolist()
    valores = radar["percentil"].round(0).tolist()
    figura = go.Figure()
    figura.add_trace(
        go.Scatterpolar(
            r=valores + [valores[0]],
            theta=categorias + [categorias[0]],
            fill="toself",
            name="Recluta",
            line=dict(color=ROJO, width=2),
            fillcolor="rgba(142, 27, 27, 0.22)",
        )
    )
    figura.add_trace(
        go.Scatterpolar(
            r=[50] * (len(categorias) + 1),
            theta=categorias + [categorias[0]],
            name="Mediana del roster",
            line=dict(color=PIEDRA, width=1.5, dash="dash"),
        )
    )
    figura.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], dtick=25, gridcolor="#E7E0D6", linecolor="#E7E0D6"),
            angularaxis=dict(gridcolor="#E7E0D6"),
        ),
        showlegend=True,
    )
    return estilo(figura, "Percentil dentro del roster")


def figura_barras(ficha: pd.DataFrame) -> go.Figure | None:
    datos = ficha[ficha["z"].notna()].sort_values("z")
    if datos.empty:
        return None
    colores = [ROJO if valor >= 0 else "#A8A29E" for valor in datos["z"]]
    figura = go.Figure(
        go.Bar(
            x=datos["z"],
            y=datos["metrica"],
            orientation="h",
            marker_color=colores,
            hovertemplate="%{y}<br>z = %{x:.2f}<extra></extra>",
        )
    )
    figura.add_vline(x=0, line_color=TINTA, line_width=1)
    return estilo(figura, "Z-score contra el roster")


def figura_movimiento(grupo: pd.DataFrame, recluta: dict) -> go.Figure | None:
    if grupo.empty or "HorzBreak" not in grupo or not np.isfinite(recluta.get("HorzBreak", np.nan)):
        return None
    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=grupo["HorzBreak"],
            y=grupo["InducedVertBreak"],
            mode="markers",
            name="Pitchers de la organización",
            marker=dict(size=9, color="#D6D3D1"),
            customdata=np.stack([grupo["RelSpeed"], grupo["stuff_plus"]], axis=1),
            hovertemplate=(
                "HorzBreak %{x:.1f} in<br>IVB %{y:.1f} in"
                "<br>%{customdata[0]:.1f} mph<br>Stuff+ %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    figura.add_trace(
        go.Scatter(
            x=[recluta["HorzBreak"]],
            y=[recluta["InducedVertBreak"]],
            mode="markers+text",
            text=["Recluta"],
            textposition="top center",
            name="Recluta",
            marker=dict(size=16, color=ROJO, symbol="diamond", line=dict(color="white", width=1)),
            hovertemplate="Recluta<br>HorzBreak %{x:.1f} in<br>IVB %{y:.1f} in<extra></extra>",
        )
    )
    figura.update_xaxes(title="Quiebre horizontal (in)")
    figura.update_yaxes(title="IVB (in)")
    return estilo(figura, "Movimiento del pitcheo")


def figura_stuff(repertorio: list[dict], referencia: pd.DataFrame, minimo: int) -> go.Figure:
    etiquetas = []
    valores = []
    medianas = []
    colores = []
    paleta = {
        "Por encima del estándar": "#1F7A4D",
        "En el estándar": "#A6843D",
        "Por debajo del estándar": "#9A3412",
    }
    for fila in repertorio:
        etiquetas.append(ETIQUETAS_TIPO.get(fila["AutoPitchType"], fila["AutoPitchType"]))
        valores.append(fila["stuff_plus"])
        colores.append(paleta[banda_stuff(fila["stuff_plus"])])
        grupo = cohorte(referencia, fila["AutoPitchType"], fila["PitcherThrows"], minimo)
        medianas.append(float(grupo["stuff_plus"].median()) if not grupo.empty else np.nan)
    figura = go.Figure()
    figura.add_trace(
        go.Bar(
            x=etiquetas,
            y=valores,
            marker_color=colores,
            name="Recluta",
            hovertemplate="%{x}<br>Stuff+ %{y:.1f}<extra></extra>",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=etiquetas,
            y=medianas,
            mode="markers",
            name="Mediana del roster",
            marker=dict(symbol="line-ew", size=22, color=TINTA, line=dict(width=3, color=TINTA)),
            hovertemplate="%{x}<br>Mediana %{y:.1f}<extra></extra>",
        )
    )
    figura.add_hline(y=100, line_dash="dash", line_color=PIEDRA, annotation_text="100 · promedio del tipo")
    figura.update_yaxes(title="Stuff+")
    return estilo(figura, "Stuff+ por tipo de lanzamiento")


def figura_mapa_location(fila: dict, modelo: dict) -> go.Figure | None:
    """Mapa de calor de Location+ con el Stuff+ de este lanzamiento fijo."""
    estado = modelo.get("location")
    if not estado or not np.isfinite(fila.get("location_plus", np.nan)):
        return None
    cuenta = fila.get("count") or f"{int(fila.get('Balls') or 0)}-{int(fila.get('Strikes') or 0)}"
    xs, zs, valores = grilla_location(
        estado,
        fila["AutoPitchType"],
        float(fila["stuff_plus"]),
        fila["PitcherThrows"],
        fila.get("BatterSide") or "Right",
        cuenta,
    )
    figura = go.Figure(
        go.Heatmap(
            x=xs,
            y=zs,
            z=valores,
            colorscale="RdBu",
            reversescale=True,
            zmid=100,
            colorbar=dict(title="Location+"),
            hovertemplate="Lado %{x:.2f} ft<br>Altura %{y:.2f} ft<br>Location+ %{z:.0f}<extra></extra>",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=[fila["PlateLocSide"]],
            y=[fila["PlateLocHeight"]],
            mode="markers",
            name="Este lanzamiento",
            marker=dict(size=14, color=TINTA, symbol="x"),
        )
    )
    figura.update_xaxes(title="PlateLocSide (ft)")
    figura.update_yaxes(title="PlateLocHeight (ft)")
    return estilo(figura, "Location+ en el plato, con este Stuff+ fijo")


def tabla_para_mostrar(ficha: pd.DataFrame) -> pd.DataFrame:
    def formato(valor, unidad, decimales):
        if not np.isfinite(valor):
            return "n/d"
        texto = f"{valor:.{decimales}f}"
        return f"{texto} {unidad}".strip()

    vista = pd.DataFrame(
        {
            "Grupo": ficha["grupo"],
            "Métrica": ficha["metrica"],
            "Recluta": [
                formato(valor, unidad, 0 if columna in {"SpinRate", "stuff_plus"} else 1)
                for valor, unidad, columna in zip(ficha["recluta"], ficha["unidad"], ficha["col"])
            ],
            "Mediana del roster": [
                formato(valor, unidad, 0 if columna in {"SpinRate", "stuff_plus"} else 1)
                for valor, unidad, columna in zip(ficha["mediana"], ficha["unidad"], ficha["col"])
            ],
            "P25 a P75": [
                f"{formato(p25, unidad, 1)} a {formato(p75, unidad, 1)}"
                if np.isfinite(p25) and np.isfinite(p75)
                else "n/d"
                for p25, p75, unidad in zip(ficha["p25"], ficha["p75"], ficha["unidad"])
            ],
            "Z-score": ficha["z"].map(lambda valor: f"{valor:+.2f}" if np.isfinite(valor) else "n/d"),
            "Percentil": ficha["percentil"].map(lambda valor: f"{valor:.0f}" if np.isfinite(valor) else "n/d"),
        }
    )
    return vista


def valor_metrica(ficha: pd.DataFrame, columna: str, campo: str = "percentil") -> float:
    fila = ficha.loc[ficha["col"] == columna]
    if fila.empty:
        return float("nan")
    return float(fila[campo].iloc[0])


def iniciar_estado():
    if "repertorio" not in st.session_state:
        st.session_state.repertorio = []
    if "siguiente_id" not in st.session_state:
        st.session_state.siguiente_id = 1
    if "form_id" not in st.session_state:
        st.session_state.form_id = 0
    if "recluta" not in st.session_state:
        st.session_state.recluta = ""
    if "estadio_escenario" not in st.session_state:
        st.session_state.estadio_escenario = ESTADIO_DEFAULT


def repertorio_en_escenario(repertorio: list[dict], estadio: str, modelo: dict) -> list[dict]:
    """Mismos pitcheos, recalculados con el aire del estadio que el reclutador está probando."""
    vista = []
    for fila in repertorio:
        copia = dict(fila)
        copia["estadio_medicion"] = fila.get("altitude_category")
        copia["stuff_medicion"] = fila.get("stuff_plus")
        copia["location_medicion"] = fila.get("location_plus")
        copia["altitude_category"] = estadio
        try:
            copia.update(puntuar_lanzamiento(copia, modelo))
        except ValueError:
            pass
        vista.append(copia)
    return vista


def nota_escenario(fila: dict) -> str:
    destino = ALTITUDES.get(fila.get("altitude_category"), str(fila.get("altitude_category")))
    origen = ALTITUDES.get(fila.get("estadio_medicion"), str(fila.get("estadio_medicion")))
    rho = fila.get("rho")
    aire = f"ρ = {rho:.3f} kg/m³" if rho is not None and np.isfinite(rho) else "densidad del escenario"
    if fila.get("estadio_medicion") == fila.get("altitude_category"):
        return f"Escenario en vivo: {destino}. {aire}."
    stuff_origen = fila.get("stuff_medicion")
    movimiento = ""
    if stuff_origen is not None and np.isfinite(stuff_origen) and np.isfinite(fila.get("stuff_plus", np.nan)):
        movimiento = f" El Stuff+ se movió {fila['stuff_plus'] - stuff_origen:+.0f} puntos respecto a la medición."
    return (
        f"Mismo reporte, ahora con el aire de {destino} ({aire}). "
        f"Se midió en {origen}.{movimiento}"
    )


def etiqueta_pitcheo(fila: dict) -> str:
    tipo = ETIQUETAS_TIPO.get(fila["AutoPitchType"], fila["AutoPitchType"])
    mano = ETIQUETAS_MANO.get(fila["PitcherThrows"], fila["PitcherThrows"])
    texto = f"{tipo} · {mano} · {fila['RelSpeed']:.1f} mph · Stuff+ {fila['stuff_plus']:.0f}"
    if fila.get("location_plus") is not None and np.isfinite(fila["location_plus"]):
        texto += f" · Location+ {fila['location_plus']:.0f}"
    return texto


def inyectar_estilo():
    st.markdown(
        f"""
        <style>
            .stApp, [data-testid="stHeader"], [data-testid="stSidebar"],
            [data-testid="stSidebarContent"], section.main {{
                background: {CREMA};
                color: {TINTA};
            }}
            .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
            .stApp h1, .stApp h2, .stApp h3, .stApp h4,
            [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
            [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
            [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
            [data-baseweb="tab"], [data-testid="stTab"] {{
                color: {TINTA} !important;
                -webkit-text-fill-color: {TINTA} !important;
            }}
            input, textarea, select,
            div[data-baseweb="select"] *,
            div[data-baseweb="input"] input,
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploader"] section * {{
                color: {TINTA} !important;
                -webkit-text-fill-color: {TINTA} !important;
            }}
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            [data-testid="stFileUploader"] section,
            .stTextInput input, .stNumberInput input, .stTextArea textarea {{
                background-color: #ffffff !important;
                border-color: #D6D3D1 !important;
            }}
            input::placeholder, textarea::placeholder {{
                color: #57534E !important;
                -webkit-text-fill-color: #57534E !important;
            }}
            .encabezado h1 {{
                font-size: 2.1rem;
                line-height: 1.1;
                margin: 0.1rem 0 0.4rem 0;
                letter-spacing: -0.03em;
                color: {TINTA} !important;
                -webkit-text-fill-color: {TINTA} !important;
            }}
            .marca {{
                color: {ROJO} !important;
                -webkit-text-fill-color: {ROJO} !important;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-size: 0.78rem;
                margin: 0;
            }}
            .bajada {{
                color: #44403C !important;
                -webkit-text-fill-color: #44403C !important;
                font-size: 1.05rem;
                max-width: 46rem;
                margin-top: 0;
            }}
            .decision {{
                background: white;
                border: 1px solid #E7E0D6;
                border-left: 4px solid {ROJO};
                border-radius: 10px;
                padding: 0.9rem 1rem;
                margin: 0.4rem 0 1rem 0;
            }}
            .decision p {{
                color: {TINTA} !important;
                -webkit-text-fill-color: {TINTA} !important;
            }}
            div[data-testid="stMetric"] {{
                background: white;
                border: 1px solid #E7E0D6;
                border-radius: 10px;
                padding: 0.55rem 0.75rem;
            }}
            div[data-testid="stMetric"] * {{
                color: {TINTA} !important;
                -webkit-text-fill-color: {TINTA} !important;
            }}
            .nota {{
                color: #57534E !important;
                -webkit-text-fill-color: #57534E !important;
                font-size: 0.92rem;
            }}
            button, button p, button span {{
                color: {TINTA} !important;
                -webkit-text-fill-color: {TINTA} !important;
            }}
            div[data-testid="stElementContainer"]:has(#traduccion-rango) {{
                display: none !important;
                height: 0 !important;
                margin: 0 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    traducir_avisos_rango()


def traducir_avisos_rango():
    """Streamlit muestra en inglés el aviso del número fuera de rango."""
    st.html(
        """
        <div id="traduccion-rango"></div>
        <script>
        (function () {
          if (window.__avisosRangoEs) return;
          window.__avisosRangoEs = true;
          const reglas = [
            [/Number is outside the allowed range\\. Please enter a value between (.+?) and (.+?)\\./g,
             "El número está fuera del rango permitido. Escribe un valor entre $1 y $2."],
            [/Number is below the allowed range\\. Please enter a value greater than or equal to (.+?)\\./g,
             "El número está por debajo del rango permitido. Escribe un valor mayor o igual que $1."],
            [/Number is above the allowed range\\. Please enter a value less than or equal to (.+?)\\./g,
             "El número está por encima del rango permitido. Escribe un valor menor o igual que $1."],
            [/Number is below the allowed range\\./g, "El número está por debajo del rango permitido."],
            [/Number is above the allowed range\\./g, "El número está por encima del rango permitido."]
          ];
          function traducir(texto) {
            let salida = texto;
            for (const [patron, reemplazo] of reglas) {
              salida = salida.replace(patron, reemplazo);
            }
            salida = salida.replace(/\\*\\*Error\\*\\*/g, "**Aviso**");
            salida = salida.replace(/\\bError:/g, "Aviso:");
            return salida;
          }
          function recorrer(nodo) {
            if (!nodo) return;
            if (nodo.nodeType === Node.TEXT_NODE) {
              const nuevo = traducir(nodo.nodeValue || "");
              if (nuevo !== nodo.nodeValue) nodo.nodeValue = nuevo;
              return;
            }
            if (nodo.nodeType !== Node.ELEMENT_NODE) return;
            if (nodo.tagName === "SCRIPT" || nodo.tagName === "STYLE") return;
            for (const atributo of ["title", "aria-label"]) {
              const valor = nodo.getAttribute(atributo);
              if (!valor) continue;
              const nuevo = traducir(valor);
              if (nuevo !== valor) nodo.setAttribute(atributo, nuevo);
            }
            for (const hijo of nodo.childNodes) recorrer(hijo);
          }
          const observador = new MutationObserver(function (cambios) {
            for (const cambio of cambios) {
              if (cambio.type === "characterData" || cambio.type === "attributes") recorrer(cambio.target);
              else for (const nodo of cambio.addedNodes) recorrer(nodo);
            }
          });
          observador.observe(document.body, {
            subtree: true,
            childList: true,
            characterData: true,
            attributes: true,
            attributeFilter: ["title", "aria-label"]
          });
          recorrer(document.body);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
        width="content",
    )


def tab_captura(modelo: dict, referencia: pd.DataFrame, vista: list[dict]):
    avisos = st.session_state.pop("csv_avisos", None)
    if avisos:
        st.warning("Algunos tipos del CSV no entraron. " + " ".join(avisos))
    st.subheader("Datos del lanzamiento")
    st.markdown(
        '<p class="nota">Un slider se compara contra sliders de la misma mano. '
        "Puedes cargar promedios por tipo o un CSV con los lanzamientos sueltos: "
        "el sistema los promedia antes de calificarlos.</p>",
        unsafe_allow_html=True,
    )

    formulario, archivo = st.columns([1.25, 0.85], gap="large")
    fid = st.session_state.form_id

    with formulario:
        with st.form(f"captura_{fid}"):
            contexto_a, contexto_b, contexto_c = st.columns(3)
            tipo = contexto_a.selectbox(
                "Tipo de lanzamiento",
                TIPOS_MODELO,
                format_func=lambda valor: ETIQUETAS_TIPO[valor],
            )
            mano = contexto_b.selectbox(
                "Mano",
                ["Right", "Left"],
                format_func=lambda valor: "Derecha" if valor == "Right" else "Izquierda",
            )
            altitud = contexto_c.selectbox(
                "Estadio de la medición",
                list(ALTITUDES),
                format_func=lambda valor: ALTITUDES[valor],
                help="Define la densidad del aire con la atmósfera estándar del modelo. "
                "Úsala según el estadio donde se tomó el reporte.",
            )

            st.markdown("**Velocidad**")
            v1, v2, v3 = st.columns(3)
            with v1:
                rel_speed = numero(
                    "RelSpeed (mph)",
                    f"rel_{fid}",
                    "Velocidad en el punto de liberación.",
                    50,
                    110,
                    0.1,
                    "%.1f",
                )
            with v2:
                effective = numero(
                    "EffectiveVelo (mph)",
                    f"eff_{fid}",
                    "Velocidad efectiva, ya con la extensión hacia el plato.",
                    50,
                    115,
                    0.1,
                    "%.1f",
                )
            with v3:
                zone = numero(
                    "ZoneSpeed (mph)",
                    f"zone_{fid}",
                    "Velocidad al cruzar el plato.",
                    50,
                    110,
                    0.1,
                    "%.1f",
                )

            st.markdown("**Rotación**")
            s1, s2, s3 = st.columns(3)
            with s1:
                spin = numero("SpinRate (rpm)", f"spin_{fid}", "Tasa de giro.", 200, 4000, 1.0, "%.0f")
            with s2:
                axis = numero(
                    "SpinAxis (°)",
                    f"axis_{fid}",
                    "Dirección del eje de giro. 180° equivale al tilt 12:00.",
                    0,
                    360,
                    1.0,
                    "%.0f",
                )
            with s3:
                tilt = st.text_input(
                    "Tilt (opcional)",
                    placeholder="11:30",
                    help="Si no tienes el eje en grados, un reloj basta. Si capturas ambos, se usa SpinAxis.",
                    key=f"tilt_{fid}",
                )

            st.markdown("**Punto de liberación**")
            r1, r2, r3 = st.columns(3)
            with r1:
                altura = numero("RelHeight (ft)", f"rh_{fid}", "Altura vertical de liberación.", 3, 8, 0.01, "%.2f")
            with r2:
                lado = numero(
                    "RelSide (ft)",
                    f"rs_{fid}",
                    "Posición lateral respecto al centro. Negativo y positivo siguen el signo de TrackMan.",
                    -5,
                    5,
                    0.01,
                    "%.2f",
                )
            with r3:
                extension = numero(
                    "Extension (ft)",
                    f"ext_{fid}",
                    "Distancia hacia el plato al soltar la bola.",
                    3,
                    9,
                    0.01,
                    "%.2f",
                )

            st.markdown("**Movimiento**")
            m1, m2, m3 = st.columns(3)
            with m1:
                ivb = numero(
                    "InducedVertBreak (in)",
                    f"ivb_{fid}",
                    "Quiebre vertical inducido, sin la caída de la gravedad.",
                    -30,
                    30,
                    0.1,
                    "%.2f",
                )
            with m2:
                hb = numero(
                    "HorzBreak (in)",
                    f"hb_{fid}",
                    "Quiebre horizontal. En esta base, positivo corre al lado del brazo de un derecho.",
                    -30,
                    30,
                    0.1,
                    "%.2f",
                )
            with m3:
                vb = numero(
                    "VertBreak (in)",
                    f"vb_{fid}",
                    "Quiebre vertical total, gravedad incluida.",
                    -80,
                    30,
                    0.1,
                    "%.2f",
                )

            st.markdown("**Ángulos de aproximación**")
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                vaa = numero("VertApprAngle (°)", f"vaa_{fid}", "Ángulo vertical al llegar al plato.", -15, 5, 0.01, "%.2f")
            with a2:
                haa = numero("HorzApprAngle (°)", f"haa_{fid}", "Ángulo horizontal al llegar al plato.", -10, 10, 0.01, "%.2f")
            with a3:
                uso = numero("Uso en el repertorio (%)", f"uso_{fid}", "Opcional. Pondera el Stuff+ global.", 0, 100, 1.0, "%.0f")
            with a4:
                cuenta = numero(
                    "Lanzamientos del reporte",
                    f"n_{fid}",
                    "Opcional. Si este renglón es un promedio, anota de cuántos lanzamientos sale.",
                    1,
                    5000,
                    1.0,
                    "%.0f",
                )

            st.markdown("**Ubicación en el plato**")
            st.caption("Location+ necesita el punto donde cruzó el plato. Si lo dejas vacío, solo se calcula el Stuff+.")
            u1, u2, u3, u4, u5 = st.columns(5)
            with u1:
                plate_side = numero(
                    "PlateLocSide (ft)",
                    f"px_{fid}",
                    "Lado en el plato. Negativo y positivo siguen el signo de TrackMan.",
                    -3,
                    3,
                    0.01,
                    "%.2f",
                )
            with u2:
                plate_height = numero(
                    "PlateLocHeight (ft)",
                    f"pz_{fid}",
                    "Altura en el plato, en pies.",
                    0.5,
                    4.5,
                    0.01,
                    "%.2f",
                )
            with u3:
                bolas = st.selectbox("Bolas", [0, 1, 2, 3], key=f"balls_{fid}")
            with u4:
                strikes = st.selectbox("Strikes", [0, 1, 2], key=f"strikes_{fid}")
            with u5:
                bate = st.selectbox(
                    "Lado del bateador",
                    ["Right", "Left"],
                    format_func=lambda valor: "Derecha" if valor == "Right" else "Izquierda",
                    key=f"bate_{fid}",
                )

            enviar = st.form_submit_button("Agregar pitcheo al repertorio", type="primary")

        if enviar:
            requeridos = {
                "RelSpeed": rel_speed,
                "SpinRate": spin,
                "InducedVertBreak": ivb,
                "HorzBreak": hb,
                "RelHeight": altura,
                "RelSide": lado,
                "Extension": extension,
            }
            vacios = [nombre for nombre, valor in requeridos.items() if valor is None]
            if axis is None and not str(tilt).strip():
                vacios.append("SpinAxis o Tilt")
            ubicacion_incompleta = (plate_side is None) ^ (plate_height is None)
            if ubicacion_incompleta:
                vacios.append("PlateLocSide y PlateLocHeight, los dos")
            if vacios:
                st.warning("Faltan datos obligatorios: " + ", ".join(vacios) + ".")
            else:
                try:
                    st.session_state.repertorio.append(
                        construir_lanzamiento(
                            {
                                "AutoPitchType": tipo,
                                "PitcherThrows": mano,
                                "altitude_category": altitud,
                                "RelSpeed": rel_speed,
                                "EffectiveVelo": effective,
                                "ZoneSpeed": zone,
                                "SpinRate": spin,
                                "SpinAxis": axis,
                                "Tilt": tilt,
                                "RelHeight": altura,
                                "RelSide": lado,
                                "Extension": extension,
                                "InducedVertBreak": ivb,
                                "HorzBreak": hb,
                                "VertBreak": vb,
                                "VertApprAngle": vaa,
                                "HorzApprAngle": haa,
                                "PlateLocSide": plate_side,
                                "PlateLocHeight": plate_height,
                                "Balls": bolas,
                                "Strikes": strikes,
                                "BatterSide": bate,
                                "uso_pct": uso,
                                "n_lanzamientos": cuenta,
                            },
                            modelo,
                        )
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    st.session_state.form_id += 1
                    st.rerun()

    with archivo:
        st.markdown("**CSV de TrackMan o Rapsodo**")
        st.markdown(
            '<p class="nota">Una fila por lanzamiento o una fila por tipo. '
            "Si vienen varios lanzamientos del mismo tipo y mano, se usa el promedio.</p>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Descargar plantilla",
            data=plantilla_csv().encode("utf-8-sig"),
            file_name="plantilla_recluta.csv",
            mime="text/csv",
        )
        csv = st.file_uploader("Archivo CSV", type=["csv"], label_visibility="collapsed")
        mano_csv = st.selectbox(
            "Mano, si el archivo no la trae",
            ["Right", "Left"],
            format_func=lambda valor: "Derecha" if valor == "Right" else "Izquierda",
            key="mano_csv",
        )
        altitud_csv = st.selectbox(
            "Estadio, si el archivo no lo trae",
            list(ALTITUDES),
            format_func=lambda valor: ALTITUDES[valor],
            key="altitud_csv",
        )
        if st.button("Cargar CSV", disabled=csv is None):
            try:
                nuevos = lanzamientos_desde_csv(csv, altitud_csv, mano_csv, modelo)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.warning(f"No pude leer el archivo. {exc}")
            else:
                st.session_state.repertorio.extend(nuevos)
                st.rerun()

        if st.button("Cargar el perfil mediano de un derecho"):
            st.session_state.repertorio = perfil_mediano(referencia, modelo)
            st.rerun()

    repertorio = vista
    st.subheader("Repertorio cargado")
    if not repertorio:
        st.info("Todavía no hay pitcheos. Agrega uno en el formulario o carga un CSV.")
        return

    st.markdown(f'<p class="nota">{html.escape(nota_escenario(repertorio[0]))}</p>', unsafe_allow_html=True)

    resumen = pd.DataFrame(
        [
            {
                "Pitcheo": ETIQUETAS_TIPO.get(fila["AutoPitchType"], fila["AutoPitchType"]),
                "Mano": "Derecha" if fila["PitcherThrows"] == "Right" else "Izquierda",
                "Velo": fila["RelSpeed"],
                "Spin": fila["SpinRate"],
                "IVB": fila["InducedVertBreak"],
                "HorzBreak": fila["HorzBreak"],
                "Stuff+": fila["stuff_plus"],
                "Location+": fila.get("location_plus"),
                "Uso %": fila.get("uso_pct"),
                "n": fila.get("n_lanzamientos"),
            }
            for fila in repertorio
        ]
    )
    st.dataframe(
        resumen,
        hide_index=True,
        width="stretch",
        column_config={
            "Velo": st.column_config.NumberColumn(format="%.1f mph"),
            "Spin": st.column_config.NumberColumn(format="%.0f rpm"),
            "IVB": st.column_config.NumberColumn(format="%.1f in"),
            "HorzBreak": st.column_config.NumberColumn(format="%.1f in"),
            "Stuff+": st.column_config.NumberColumn(format="%.1f"),
            "Location+": st.column_config.NumberColumn(format="%.1f"),
            "Uso %": st.column_config.NumberColumn(format="%.0f"),
            "n": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    indice = st.selectbox(
        "Quitar un pitcheo",
        range(len(repertorio)),
        format_func=lambda i: etiqueta_pitcheo(repertorio[i]),
    )
    if st.button("Quitar seleccionado"):
        st.session_state.repertorio.pop(indice)
        st.rerun()


def tab_comparacion(modelo: dict, referencia: pd.DataFrame, minimo: int, vista: list[dict]):
    repertorio = vista
    st.subheader("Recluta contra la organización")
    if not repertorio:
        st.info("Carga un pitcheo en la pestaña de captura para abrir la comparativa.")
        return

    indice = st.selectbox(
        "Pitcheo a comparar",
        range(len(repertorio)),
        format_func=lambda i: etiqueta_pitcheo(repertorio[i]),
    )
    elegido = repertorio[indice]
    st.markdown(f'<p class="nota">{html.escape(nota_escenario(elegido))}</p>', unsafe_allow_html=True)
    grupo = cohorte(referencia, elegido["AutoPitchType"], elegido["PitcherThrows"], minimo)
    mano = ETIQUETAS_MANO.get(elegido["PitcherThrows"], elegido["PitcherThrows"])
    tipo = ETIQUETAS_TIPO.get(elegido["AutoPitchType"], elegido["AutoPitchType"])
    lanzamientos = int(grupo["n_lanzamientos"].sum()) if not grupo.empty else 0
    st.markdown(
        f'<p class="nota">{tipo} de pitcher {mano}, contra {len(grupo)} pitchers de la organización '
        f"con al menos {minimo} lanzamientos de ese tipo ({lanzamientos:,} lanzamientos en total). "
        "El percentil 80 quiere decir que el valor supera al 80% de ese grupo. "
        "No significa, por sí solo, que el pitcheo sea mejor: en el quiebre horizontal el signo "
        "importa, y dentro de la misma mano un valor más positivo es simplemente más positivo.</p>",
        unsafe_allow_html=True,
    )
    if len(grupo) < 8:
        st.warning(
            "Hay pocos pitchers en este corte. El percentil se puede mover mucho si bajas el mínimo de lanzamientos."
        )

    ficha = ficha_comparacion(elegido, grupo)
    c1, c2, c3, c4 = st.columns(4)
    delta_stuff = elegido["stuff_plus"] - 100
    texto_delta_stuff = f"{delta_stuff:+.0f} vs 100"
    if elegido.get("estadio_medicion") != elegido.get("altitude_category") and np.isfinite(elegido.get("stuff_medicion", np.nan)):
        texto_delta_stuff = f"{elegido['stuff_plus'] - elegido['stuff_medicion']:+.0f} al cambiar el aire"
    c1.metric("Stuff+", f"{elegido['stuff_plus']:.0f}", texto_delta_stuff)
    percentil_stuff = valor_metrica(ficha, "stuff_plus")
    c2.metric(
        "Percentil de Stuff+",
        "—" if not np.isfinite(percentil_stuff) else f"{percentil_stuff:.0f}",
    )
    location = elegido.get("location_plus")
    tiene_location = location is not None and np.isfinite(location)
    c3.metric(
        "Location+",
        "—" if not tiene_location else f"{location:.0f}",
        None
        if not tiene_location
        else (
            f"{location - elegido['location_medicion']:+.0f} al cambiar el aire"
            if elegido.get("estadio_medicion") != elegido.get("altitude_category")
            and np.isfinite(elegido.get("location_medicion", np.nan))
            else f"{location - 100:+.0f} vs 100"
        ),
    )
    percentil_loc = valor_metrica(ficha, "location_plus")
    c4.metric(
        "Percentil de Location+",
        "—" if not np.isfinite(percentil_loc) else f"{percentil_loc:.0f}",
    )

    st.markdown(
        f'<p class="nota">El modelo físico esperaba IVB de {elegido["ivb_modelo"]:.1f} in '
        f"y HorzBreak de {elegido['hb_modelo']:.1f} in con esta velocidad, este giro y ρ = {elegido['rho']:.3f} kg/m³. "
        f"El reporte trae {elegido['InducedVertBreak']:.1f} y {elegido['HorzBreak']:.1f}. "
        f"ΔIVB = {elegido['delta_ivb']:+.1f} in. "
        f"ΔHB, medido hacia el lado del brazo, = {elegido['delta_hb']:+.1f} in.</p>",
        unsafe_allow_html=True,
    )

    radar = figura_radar(ficha)
    barras = figura_barras(ficha)
    grafico_a, grafico_b = st.columns(2)
    with grafico_a:
        if radar is None:
            st.info("Faltan métricas para armar el radar.")
        else:
            st.plotly_chart(radar, width="stretch")
    with grafico_b:
        if barras is None:
            st.info("No hay z-scores para graficar.")
        else:
            st.plotly_chart(barras, width="stretch")

    movimiento = figura_movimiento(grupo, elegido)
    if movimiento is not None:
        st.plotly_chart(movimiento, width="stretch")

    mapa = figura_mapa_location(elegido, modelo)
    if mapa is not None:
        st.plotly_chart(mapa, width="stretch")
        st.markdown(
            '<p class="nota">El mapa deja fijo el Stuff+ de este pitcheo y mueve solo el punto en el plato. '
            "100 es la ubicación típica de ese tipo. El punto negro es el lanzamiento capturado. "
            "Una zona roja suma whiff por dónde cayó; una zona azul lo resta.</p>",
            unsafe_allow_html=True,
        )

    tabla = tabla_para_mostrar(ficha)
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.download_button(
        "Descargar esta comparativa",
        data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"comparativa_{elegido['AutoPitchType']}_{elegido['PitcherThrows']}.csv",
        mime="text/csv",
    )


def tab_decision(modelo: dict, referencia: pd.DataFrame, minimo: int, vista: list[dict]):
    repertorio = vista
    st.subheader("Stuff+ y recomendación")
    if not repertorio:
        st.info("Carga un pitcheo en la pestaña de captura para calificar el repertorio.")
        return

    st.markdown(f'<p class="nota">{html.escape(nota_escenario(repertorio[0]))}</p>', unsafe_allow_html=True)

    global_stuff = stuff_ponderado(repertorio)
    lectura = texto_decision(repertorio, referencia, minimo)
    st.markdown(f'<div class="decision"><p>{html.escape(lectura)}</p></div>', unsafe_allow_html=True)

    pesos = [float(fila.get("uso_pct") or 0) for fila in repertorio]
    if sum(pesos) > 0:
        como = "ponderado por el uso que capturaste"
    elif sum(float(fila.get("n_lanzamientos") or 0) for fila in repertorio) > 0:
        como = "ponderado por la cantidad de lanzamientos de cada tipo"
    else:
        como = "promedio simple de los tipos cargados"
    st.metric(
        "Stuff+ del repertorio",
        f"{global_stuff:.0f}",
        help=f"Se calcula {como}. 100 es el promedio de ese tipo de lanzamiento.",
    )
    global_location = location_ponderado(repertorio)
    if np.isfinite(global_location):
        st.metric(
            "Location+ del repertorio",
            f"{global_location:.0f}",
            help="100 es la ubicación típica del tipo, después de descontar el Stuff+.",
        )

    st.plotly_chart(figura_stuff(repertorio, referencia, minimo), width="stretch")

    tarjetas = st.columns(min(len(repertorio), 4))
    for indice, fila in enumerate(repertorio):
        grupo = cohorte(referencia, fila["AutoPitchType"], fila["PitcherThrows"], minimo)
        percentil = (
            float((grupo["stuff_plus"] <= fila["stuff_plus"]).mean() * 100) if not grupo.empty else float("nan")
        )
        with tarjetas[indice % len(tarjetas)]:
            st.markdown(f"**{ETIQUETAS_TIPO.get(fila['AutoPitchType'], fila['AutoPitchType'])}**")
            st.metric("Stuff+", f"{fila['stuff_plus']:.0f}")
            if np.isfinite(percentil):
                st.caption(f"Percentil {percentil:.0f} entre sus pares. {banda_stuff(fila['stuff_plus'])} en la escala de ese tipo.")
            st.caption(f"Probabilidad de whiff que ve el Stuff+: {fila['p_whiff'] * 100:.1f}%.")
            if fila.get("location_plus") is not None and np.isfinite(fila["location_plus"]):
                st.caption(
                    f"Location+ {fila['location_plus']:.0f}. "
                    f"En ese punto del plato el modelo ve {fila['p_location'] * 100:.1f}% de whiff, "
                    f"contra {fila['p_location_ref'] * 100:.1f}% en la ubicación típica del tipo."
                )

    with st.expander("Cómo se calcula este Stuff+"):
        st.markdown(
            """
El notebook fija la escala así:

`p = 1 / (1 + e^(−z))`, con `z` la combinación lineal de las variables físicas ya estandarizadas.

`Stuff+ = 100 + 10 × (p − μ_T) / σ_T`

`p` es la probabilidad de whiff. `μ_T` y `σ_T` son la media y la desviación de ese tipo de lanzamiento en el entrenamiento, no del recluta. 100 es el promedio de una recta si el pitcheo es recta, o el promedio de un slider si es slider. 10 puntos son una desviación estándar dentro de ese tipo.

El vector usa velocidad de salida, spin, IVB, quiebre horizontal (puesto en el marco del lado del brazo), el residuo contra el modelo de Magnus (ΔIVB y ΔHB), el factor de spin `S = R ω / V`, extensión, altura y lado de liberación, y la densidad del aire del estadio. El tipo de pitcheo y la mano entran como indicadoras. La ubicación en el plato no entra: el Stuff+ describe el material, no el comando.

Location+ usa otra probabilidad: la del whiff dadas la ubicación, la cuenta, la mano del bateador, la mano del pitcher y el Stuff+ ya calculado. Se resta la probabilidad de la ubicación típica de ese tipo y esa diferencia se pone en la misma escala de 100. Así un pitcheo con mucho Stuff+ no se lleva el crédito de la ubicación.

Los coeficientes son asociaciones dentro del modelo, no efectos causales. Velocidad, spin rate y el factor S comparten información, porque S se arma con las otras dos, así que esos tres betas no se leen por separado.
            """
        )
        st.caption(
            f"Regresión logística entrenada con {modelo['n_train']:,} lanzamientos y evaluada en "
            f"{modelo['n_test']:,}. AUC de prueba: {modelo['auc']:.3f}. "
            f"Tasa de whiff en la base: {modelo['whiff_rate'] * 100:.1f}%."
        )
        def nombre_beta(variable: str) -> str:
            if variable in NOMBRES_BETA:
                return NOMBRES_BETA[variable]
            if variable == "intercepto":
                return "Intercepto"
            if variable.startswith("AutoPitchType_"):
                tipo = variable.split("_", 1)[1]
                return f"Tipo {ETIQUETAS_TIPO.get(tipo, tipo)} (contra Four-Seam)"
            if variable.startswith("PitcherThrows_"):
                return "Mano zurda (contra derecha)"
            return variable

        coefs = pd.DataFrame(modelo["coeficientes"])
        coefs["Variable"] = coefs["variable"].map(nombre_beta)
        coefs["Beta estandarizado"] = coefs["beta"]
        coefs = coefs.sort_values("beta", key=lambda serie: serie.abs(), ascending=False)
        st.dataframe(
            coefs[["Variable", "Beta estandarizado"]],
            hide_index=True,
            width="stretch",
            column_config={"Beta estandarizado": st.column_config.NumberColumn(format="%+.3f")},
        )


def main():
    st.set_page_config(
        page_title="Reclutamiento · DRITAM",
        page_icon="⚾",
        layout="wide",
    )
    inyectar_estilo()
    iniciar_estado()

    with st.sidebar:
        st.markdown(f"<p class='marca'>DRITAM</p>", unsafe_allow_html=True)
        st.header("Recluta")
        st.session_state.recluta = st.text_input("Nombre", value=st.session_state.recluta, placeholder="Prospecto")
        minimo = st.slider(
            "Mínimo de lanzamientos para entrar a la comparación",
            min_value=10,
            max_value=150,
            value=30,
            step=5,
            help="Un pitcher de la organización entra al percentil solo si tiene al menos estos lanzamientos del mismo tipo y mano.",
        )
        st.selectbox(
            "Estadio del escenario",
            list(ALTITUDES),
            format_func=lambda valor: ALTITUDES[valor],
            key="estadio_escenario",
            help="Al cambiar el estadio, Stuff+ y Location+ se recalculan al momento con la densidad de ese aire. "
            "Velocidad, giro, quiebre y ubicación del reporte se quedan iguales.",
        )
        st.caption(
            "Este control es en vivo. Prueba otro parque y la comparación y la decisión se actualizan solas."
        )
        if st.button("Limpiar repertorio", disabled=not st.session_state.repertorio):
            st.session_state.repertorio = []
            st.rerun()

    try:
        with st.spinner("Cargando el roster y el Stuff+ de la organización..."):
            modelo = cargar_modelo()
            referencia = cargar_referencia()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    nombre = st.session_state.recluta.strip()
    titulo = "Reclutamiento de pitchers"
    if nombre:
        titulo = f"Recluta: {nombre}"
    st.markdown(
        f"""
        <div class="encabezado">
            <p class="marca">DRITAM · Diablos Rojos</p>
            <h1>{html.escape(titulo)}</h1>
            <p class="bajada">Captura el material del prospecto y compáralo con los pitchers reales
            de la organización. En Stuff+ y en Location+, 100 es el promedio de ese tipo de lanzamiento.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    vista = repertorio_en_escenario(
        st.session_state.repertorio,
        st.session_state.get("estadio_escenario", ESTADIO_DEFAULT),
        modelo,
    )

    captura, comparacion, decision = st.tabs(["1 · Captura", "2 · Comparación", "3 · Decisión"])
    with captura:
        tab_captura(modelo, referencia, vista)
    with comparacion:
        tab_comparacion(modelo, referencia, minimo, vista)
    with decision:
        tab_decision(modelo, referencia, minimo, vista)


if __name__ == "__main__":
    main()
