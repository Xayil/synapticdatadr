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
