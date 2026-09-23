# src/plots.py
import pandas as pd
import plotly.graph_objects as go

COLORES_LANZAMIENTO = {
    "Four-Seam": "#e74c3c",
    "OneSeamFastBall": "#c0392b",
    "TwoSeamFastBall": "#d35400",
    "Sinker": "#e67e22",
    "Cutter": "#f1c40f",
    "Slider": "#3498db",
    "Sweeper": "#5dade2",
    "Curveball": "#9b59b6",
    "Changeup": "#2ecc71",
    "Splitter": "#1abc9c",
    "Knuckleball": "#95a5a6",
    "Other": "#bdc3c7",
}

PALETA_EXTRA = ["#1abc9c", "#e84393", "#fdcb6e", "#00cec9", "#6c5ce7", "#fab1a0"]

ZONA_STRIKE = {"x0": -0.83, "x1": 0.83, "y0": 1.5, "y1": 3.5}


def resolver_columnas(df: pd.DataFrame) -> dict:
    """Localiza las columnas de ubicación, tipo e identificador del lanzamiento."""

    def primera(*nombres):
        for nombre in nombres:
            if nombre in df.columns:
                return nombre
        return None

    return {
        "x": primera("PlateLocSide", "plate_x"),
        "y": primera("PlateLocHeight", "plate_z"),
        "tipo": primera("AutoPitchType", "TaggedPitchType", "pitch_type"),
        "id": primera("PitchUID", "pitch_uid"),
        "pitcher": primera("pitcher_anon_id", "PitcherId", "Pitcher"),
    }


def color_de(tipo: str) -> str:
    if tipo in COLORES_LANZAMIENTO:
        return COLORES_LANZAMIENTO[tipo]
    return PALETA_EXTRA[sum(ord(c) for c in tipo) % len(PALETA_EXTRA)]


GRAFICAS = [
    {
        "id": "zona",
        "nombre": "Zona de strike",
        "modo": "scatter",
        "forma": "zona",
        "x": ["PlateLocSide", "plate_x"],
        "y": ["PlateLocHeight", "plate_z"],
        "x_titulo": "Ubicación horizontal (ft)",
        "y_titulo": "Ubicación vertical (ft)",
        "x_rango": [-2.5, 2.5],
        "y_rango": [0, 5],
    },
    {
        "id": "movimiento",
        "nombre": "Movimiento",
        "modo": "scatter",
        "x": ["HorzBreak"],
        "y": ["InducedVertBreak"],
        "x_titulo": "Movimiento horizontal (in)",
        "y_titulo": "Movimiento vertical inducido (in)",
    },
    {
        "id": "velocidad",
        "nombre": "Velocidad y spin",
        "modo": "scatter",
        "x": ["RelSpeed"],
        "y": ["SpinRate"],
        "x_titulo": "Velocidad de salida (mph)",
        "y_titulo": "Spin rate (rpm)",
    },
    {
        "id": "liberacion",
        "nombre": "Punto de liberación",
        "modo": "scatter",
        "x": ["RelSide"],
        "y": ["RelHeight"],
        "x_titulo": "Liberación lateral (ft)",
        "y_titulo": "Altura de liberación (ft)",
    },
    {
        "id": "tipos",
        "nombre": "Conteo por tipo",
        "modo": "bar",
        "categoria": ["AutoPitchType", "TaggedPitchType", "pitch_type"],
        "x_titulo": "Tipo de lanzamiento",
        "y_titulo": "Lanzamientos",
    },
    {
        "id": "resultado",
        "nombre": "Resultado del pitcheo",
        "modo": "bar",
        "categoria": ["PitchCall"],
        "x_titulo": "Resultado",
        "y_titulo": "Lanzamientos",
    },
    {
        "id": "bateo",
        "nombre": "Tipo de batazo",
        "modo": "bar",
        "categoria": ["hit_type"],
        "x_titulo": "Tipo de bola bateada",
        "y_titulo": "Lanzamientos",
    },
]


def _primera(df: pd.DataFrame, nombres) -> str | None:
    for nombre in nombres:
        if nombre in df.columns:
            return nombre
    return None


def _ids(grupo: pd.DataFrame, columna_id: str | None) -> list:
    if columna_id:
        return grupo[columna_id].astype(str).tolist()
    return [str(i) for i in grupo.index]


def _dispersion(df: pd.DataFrame, spec: dict, max_puntos: int = 8000) -> dict:
    x_col = _primera(df, spec["x"])
    y_col = _primera(df, spec["y"])
    if not x_col or not y_col:
        faltan = ", ".join(spec["x"][:1] + spec["y"][:1])
        raise ValueError(f"Faltan columnas para esta gráfica: {faltan}.")

    datos = df.dropna(subset=[x_col, y_col]).copy()
    total = int(len(datos))
    muestra = False
    if total > max_puntos:
        datos = datos.sample(max_puntos, random_state=0)
        muestra = True

    tipo_col = _primera(datos, ["AutoPitchType", "TaggedPitchType", "pitch_type"])
    id_col = _primera(datos, ["PitchUID", "pitch_uid"])
    if tipo_col:
        datos["_tipo_grafico"] = datos[tipo_col].astype(str)
        grupos = datos.groupby("_tipo_grafico", sort=False)
    else:
        grupos = [("Lanzamientos", datos)]

    series = []
    for tipo, grupo in grupos:
        tipo = str(tipo)
        series.append(
            {
                "tipo": tipo,
                "color": color_de(tipo),
                "x": [round(float(v), 4) for v in grupo[x_col]],
                "y": [round(float(v), 4) for v in grupo[y_col]],
                "id": _ids(grupo, id_col),
            }
        )

    return {
        "modo": "scatter",
        "series": series,
        "total": total,
        "mostrados": int(len(datos)),
        "muestra": muestra,
    }


def _barras(df: pd.DataFrame, spec: dict) -> dict:
    col = _primera(df, spec["categoria"])
    if not col:
        raise ValueError(f"Falta la columna {spec['categoria'][0]} para esta gráfica.")

    conteo = df[col].dropna().astype(str)
    conteo = conteo[conteo.str.lower() != "nan"].value_counts()
    return {
        "modo": "bar",
        "series": [
            {
                "tipo": spec["nombre"],
                "x": conteo.index.tolist(),
                "y": [int(v) for v in conteo.values],
                "color": [color_de(str(nombre)) for nombre in conteo.index],
                "id": [],
            }
        ],
        "total": int(conteo.sum()),
        "mostrados": int(conteo.sum()),
        "muestra": False,
    }


def graficas_disponibles(df: pd.DataFrame) -> list:
    opciones = []
    for spec in GRAFICAS:
        if spec["modo"] == "bar":
            lista = _primera(df, spec["categoria"]) is not None
            faltan = spec["categoria"][0]
        else:
            lista = _primera(df, spec["x"]) is not None and _primera(df, spec["y"]) is not None
            faltan = f"{spec['x'][0]} y {spec['y'][0]}"
        opciones.append(
            {
                "id": spec["id"],
                "nombre": spec["nombre"],
                "disponible": lista,
                "motivo": "" if lista else f"Faltan {faltan}",
            }
        )
    return opciones


def construir_grafica(df: pd.DataFrame, grafica_id: str = "zona") -> dict:
    """Arma una gráfica distinta según el id, siempre con el dataframe ya filtrado."""
    spec = next((g for g in GRAFICAS if g["id"] == grafica_id), GRAFICAS[0])
    if spec["modo"] == "bar":
        paquete = _barras(df, spec)
    else:
        paquete = _dispersion(df, spec)

    paquete.update(
        {
            "id": spec["id"],
            "titulo": spec["nombre"],
            "forma": spec.get("forma"),
            "ejes": {
                "x": spec["x_titulo"],
                "y": spec["y_titulo"],
                "x_rango": spec.get("x_rango"),
                "y_rango": spec.get("y_rango"),
            },
            "graficas": graficas_disponibles(df),
        }
    )
    return paquete


def serie_zona(df: pd.DataFrame, max_puntos: int = 8000) -> dict:
    """
    Arma las series de la zona de strike a partir del dataframe ya filtrado.
    Ese dataframe debe ser el que D-Tale está mostrando.
    """
    columnas = resolver_columnas(df)
    if not columnas["x"] or not columnas["y"]:
        raise ValueError(
            "Faltan las columnas de ubicación (PlateLocSide y PlateLocHeight)."
        )

    datos = df.dropna(subset=[columnas["x"], columnas["y"]]).copy()
    total = int(len(datos))
    muestra = False
    if total > max_puntos:
        datos = datos.sample(max_puntos, random_state=0)
        muestra = True

    if columnas["tipo"] and columnas["tipo"] in datos.columns:
        datos["_tipo_grafico"] = datos[columnas["tipo"]].astype(str)
        grupos = datos.groupby("_tipo_grafico", sort=False)
    else:
        grupos = [("Lanzamientos", datos)]

    series = []
    for tipo, grupo in grupos:
        tipo = str(tipo)
        ids = (
            grupo[columnas["id"]].astype(str).tolist()
            if columnas["id"]
            else [str(i) for i in grupo.index]
        )
        series.append(
            {
                "tipo": tipo,
                "color": color_de(tipo),
                "x": [round(float(v), 4) for v in grupo[columnas["x"]]],
                "y": [round(float(v), 4) for v in grupo[columnas["y"]]],
                "id": ids,
            }
        )

    return {
        "series": series,
        "total": total,
        "mostrados": int(len(datos)),
        "muestra": muestra,
        "columnas": columnas,
    }


def plot_strike_zone_by_pitcher(df: pd.DataFrame, pitcher_id: str) -> go.Figure:
    """
    Zona de strike de un pitcher. Si el dataframe ya viene filtrado por D-Tale,
    pasa ese dataframe y el pitcher que quedó dentro del filtro.
    """
    columnas = resolver_columnas(df)
    if columnas["pitcher"]:
        pitcher_df = df[df[columnas["pitcher"]] == pitcher_id].copy()
    else:
        pitcher_df = df.copy()

    if pitcher_df.empty:
        raise ValueError(f"No se encontraron lanzamientos para el pitcher: {pitcher_id}")

    paquete = serie_zona(pitcher_df)
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=ZONA_STRIKE["x0"],
        x1=ZONA_STRIKE["x1"],
        y0=ZONA_STRIKE["y0"],
        y1=ZONA_STRIKE["y1"],
        line=dict(color="White", width=2, dash="dash"),
        fillcolor="rgba(255, 255, 255, 0.05)",
    )

    for serie in paquete["series"]:
        fig.add_trace(
            go.Scatter(
                x=serie["x"],
                y=serie["y"],
                mode="markers",
                name=serie["tipo"],
                marker=dict(size=9, opacity=0.8, color=serie["color"]),
                customdata=serie["id"],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "<b>ID:</b> %{customdata}<br>"
                    "<b>Ubicación X:</b> %{x:.2f} ft<br>"
                    "<b>Ubicación Z:</b> %{y:.2f} ft<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Mapa de pitcheos - Pitcher ID: {pitcher_id}",
        xaxis=dict(title="Ubicación horizontal (ft)", range=[-2.5, 2.5], zeroline=False),
        yaxis=dict(title="Ubicación vertical (ft)", range=[0, 5], zeroline=False),
        width=600,
        height=600,
        template="plotly_dark",
        legend=dict(title="Tipo de lanzamiento"),
    )
    return fig
