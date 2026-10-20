from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RUTA_DATOS = ROOT / "resultados/stuff_plus_lanzamientos.parquet"
RUTA_RANKING = ROOT / "resultados/ranking_pitchers.csv"
RUTA_SALIDA = ROOT / "resultados/repertorio_top20.csv"

print("\n=== ANÁLISIS DE REPERTORIOS STUFF+ ===")

# Cargar los resultados existentes
df = pd.read_parquet(RUTA_DATOS)
ranking = pd.read_csv(RUTA_RANKING)

# Seleccionar los 20 mejores pitchers
top20 = ranking.head(20)
ids = top20["pitcher_anon_id"].tolist()

# Filtrar lanzamientos de estos pitchers
datos = df[
    df["pitcher_anon_id"].isin(ids)
    & df["stuff_plus"].notna()
    & df["AutoPitchType"].notna()
].copy()

print(f"Pitchers seleccionados: {len(ids)}")
print(f"Lanzamientos analizados: {len(datos):,}")

# Estadísticas por pitcher y tipo de lanzamiento
repertorio = (
    datos.groupby(["pitcher_anon_id", "AutoPitchType"])
    .agg(
        lanzamientos=("stuff_plus", "count"),
        stuff_promedio=("stuff_plus", "mean"),
        stuff_mediana=("stuff_plus", "median"),
        velocidad_promedio=("RelSpeed", "mean"),
        spin_promedio=("SpinRate", "mean"),
        ivb_promedio=("InducedVertBreak", "mean"),
        hb_promedio=("HorzBreak", "mean"),
    )
    .reset_index()
)

# Porcentaje de utilización de cada lanzamiento
totales = repertorio.groupby("pitcher_anon_id")[
    "lanzamientos"
].transform("sum")

repertorio["uso_porcentaje"] = (
    repertorio["lanzamientos"] / totales * 100
)

# Incorporar posición del ranking general
repertorio = repertorio.merge(
    top20[["pitcher_anon_id", "posicion"]],
    on="pitcher_anon_id",
    how="left"
)

# Ordenar por pitcher y frecuencia de uso
repertorio = repertorio.sort_values(
    ["posicion", "lanzamientos"],
    ascending=[True, False]
)

# Redondear resultados
columnas = [
    "stuff_promedio",
    "stuff_mediana",
    "velocidad_promedio",
    "spin_promedio",
    "ivb_promedio",
    "hb_promedio",
    "uso_porcentaje",
]

repertorio[columnas] = repertorio[columnas].round(2)

# Guardar resultados
repertorio.to_csv(
    RUTA_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== REPERTORIO DE LOS 5 MEJORES PITCHERS ===")

for pitcher in ids[:5]:
    posicion = int(
        top20.loc[
            top20["pitcher_anon_id"] == pitcher,
            "posicion"
        ].iloc[0]
    )

    print(f"\n#{posicion} - {pitcher}")

    tabla = repertorio[
        repertorio["pitcher_anon_id"] == pitcher
    ][[
        "AutoPitchType",
        "lanzamientos",
        "uso_porcentaje",
        "stuff_promedio",
    ]]

    print(tabla.to_string(index=False))

print("\n=== TOP 15 LANZAMIENTOS INDIVIDUALES ===")

# Umbral provisional para evitar muestras diminutas
mejores = repertorio[
    repertorio["lanzamientos"] >= 30
].sort_values(
    "stuff_promedio",
    ascending=False
)

print(
    mejores[[
        "pitcher_anon_id",
        "AutoPitchType",
        "lanzamientos",
        "stuff_promedio",
        "uso_porcentaje",
    ]].head(15).to_string(index=False)
)

print(f"\nArchivo guardado: {RUTA_SALIDA}")
