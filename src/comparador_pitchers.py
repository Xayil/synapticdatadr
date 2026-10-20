from pathlib import Path
import re

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SynapticData | Comparador",
    page_icon="⚾",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
RUTA = ROOT / "resultados/stuff_plus_lanzamientos.parquet"

st.title("⚾ SynapticData")
st.subheader("Comparador de pitchers | Stuff+")
st.caption("Comparación de 1 a 4 pitchers con el modelo original.")

@st.cache_data
def cargar_datos():
    columnas = [
        "pitcher_anon_id", "AutoPitchType",
        "stuff_plus", "RelSpeed", "SpinRate",
        "InducedVertBreak", "HorzBreak"
    ]
    df = pd.read_parquet(RUTA, columns=columnas)
    return df.dropna(
        subset=["pitcher_anon_id", "stuff_plus"]
    )

if not RUTA.exists():
    st.error("No se encontró el archivo de resultados.")
    st.stop()

df = cargar_datos()

st.markdown("### Selecciona los pitchers")

cantidad = st.selectbox(
    "Número de pitchers a comparar",
    [1, 2, 3, 4],
    index=1
)

columnas = st.columns(cantidad)
seleccionados = []

for i, columna in enumerate(columnas):
    with columna:
        valor = st.text_input(
            f"Pitcher {i + 1}",
            value=["00778", "00417", "", ""][i],
            key=f"pitcher_{i}"
        )

        valor = valor.strip()

        if valor:
            if re.fullmatch(r"\d{1,5}", valor):
                identificador = f"pitcher_{int(valor):05d}"
            else:
                identificador = valor

            seleccionados.append(identificador)

if not seleccionados:
    st.info("Ingresa al menos un ID.")
    st.stop()

if len(seleccionados) != len(set(seleccionados)):
    st.warning("No repitas el mismo pitcher.")
    st.stop()

disponibles = set(df["pitcher_anon_id"].unique())
no_encontrados = [
    p for p in seleccionados if p not in disponibles
]

if no_encontrados:
    st.error(
        "Pitchers sin Stuff+ disponible: "
        + ", ".join(no_encontrados)
    )
    st.stop()

datos = df[df["pitcher_anon_id"].isin(seleccionados)].copy()

# Resumen por pitcher
resumen = (
    datos.groupby("pitcher_anon_id")
    .agg(
        lanzamientos=("stuff_plus", "count"),
        stuff_promedio=("stuff_plus", "mean"),
        stuff_mediana=("stuff_plus", "median"),
        stuff_maximo=("stuff_plus", "max"),
        velocidad=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"),
        tipos=("AutoPitchType", "nunique")
    )
    .reindex(seleccionados)
)

st.divider()
st.markdown("### Comparación general")

tarjetas = st.columns(len(seleccionados))

for i, pitcher in enumerate(seleccionados):
    fila = resumen.loc[pitcher]

    with tarjetas[i]:
        st.markdown(f"**{pitcher}**")
        st.metric("Stuff+ promedio", f"{fila['stuff_promedio']:.2f}")
        st.metric("Lanzamientos", f"{int(fila['lanzamientos']):,}")
        st.metric("Tipos de pitch", int(fila["tipos"]))

st.markdown("### Stuff+ promedio por pitcher")

st.bar_chart(
    resumen[["stuff_promedio"]],
    y="stuff_promedio",
    horizontal=True
)

st.markdown("### Estadísticas comparativas")

tabla = resumen.reset_index().rename(
    columns={"pitcher_anon_id": "Pitcher"}
)

st.dataframe(
    tabla.round(2),
    hide_index=True,
    use_container_width=True
)

# Repertorios
st.divider()
st.markdown("### Comparación de repertorios")

repertorio = (
    datos.groupby(
        ["pitcher_anon_id", "AutoPitchType"],
        observed=True
    )
    .agg(
        lanzamientos=("stuff_plus", "count"),
        stuff_promedio=("stuff_plus", "mean"),
        velocidad=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("HorzBreak", "mean")
    )
    .reset_index()
)

repertorio = repertorio[
    repertorio["lanzamientos"] > 0
].copy()

totales = repertorio.groupby(
    "pitcher_anon_id"
)["lanzamientos"].transform("sum")

repertorio["uso_%"] = (
    repertorio["lanzamientos"] / totales * 100
)

st.markdown("#### Stuff+ por tipo de lanzamiento")

grafica = repertorio.pivot(
    index="AutoPitchType",
    columns="pitcher_anon_id",
    values="stuff_promedio"
)

st.bar_chart(grafica)

st.markdown("#### Análisis individual del repertorio")

for pitcher in seleccionados:

    with st.expander(
        f"⚾ {pitcher} — Análisis completo",
        expanded=True
    ):
        rp = repertorio[
            repertorio["pitcher_anon_id"] == pitcher
        ].copy()

        rp = rp.sort_values(
            "lanzamientos", ascending=False
        )

        st.markdown("##### Composición del arsenal")

        st.bar_chart(
            rp.set_index("AutoPitchType")[["uso_%"]]
        )

        st.markdown("##### Calidad de cada lanzamiento")

        st.bar_chart(
            rp.set_index("AutoPitchType")[["stuff_promedio"]]
        )

        st.markdown("##### Características físicas")

        columnas_fisicas = [
            "AutoPitchType",
            "lanzamientos",
            "uso_%",
            "stuff_promedio",
            "velocidad",
            "spin",
            "ivb",
            "hb"
        ]

        st.dataframe(
            rp[columnas_fisicas].round(2),
            hide_index=True,
            use_container_width=True
        )

        # Identificar lanzamiento principal
        principal = rp.iloc[0]

        st.info(
            f"Lanzamiento principal: "
            f"{principal['AutoPitchType']} | "
            f"Uso: {principal['uso_%']:.1f}% | "
            f"Stuff+: {principal['stuff_promedio']:.2f}"
        )

        # Evaluar solamente tipos con muestra suficiente
        evaluables = rp[
            rp["lanzamientos"] >= 30
        ]

        if not evaluables.empty:
            mejor = evaluables.loc[
                evaluables["stuff_promedio"].idxmax()
            ]

            st.success(
                f"Mayor Stuff+ entre lanzamientos "
                f"con al menos 30 observaciones: "
                f"{mejor['AutoPitchType']} "
                f"({mejor['stuff_promedio']:.2f})"
            )

        pocos = rp[rp["lanzamientos"] < 30]

        if not pocos.empty:
            st.warning(
                "Algunos tipos tienen menos de "
                "30 lanzamientos. Sus promedios "
                "deben interpretarse con cautela."
            )

st.divider()

st.markdown("### Comparación directa por tipo de pitch")

comparacion = repertorio.pivot_table(
    index="AutoPitchType",
    columns="pitcher_anon_id",
    values="stuff_promedio",
    observed=True
)

comparacion = comparacion.reindex(
    columns=seleccionados
)

st.dataframe(
    comparacion.round(2),
    use_container_width=True
)

st.markdown("### Comparación de utilización")

uso = repertorio.pivot_table(
    index="AutoPitchType",
    columns="pitcher_anon_id",
    values="uso_%",
    observed=True
)

uso = uso.reindex(columns=seleccionados)

st.bar_chart(uso)

st.caption(
    "Las puntuaciones Stuff+ provienen del "
    "modelo original. Este comparador solo "
    "agrupa y visualiza resultados existentes. "
    "No modifica el modelo ni sus parámetros."
)