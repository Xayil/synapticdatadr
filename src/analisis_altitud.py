from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RUTA = ROOT / "resultados/stuff_plus_lanzamientos.parquet"
SALIDA = ROOT / "resultados/analisis_altitud.csv"

print("\n=== ANÁLISIS DE STUFF+ POR ALTITUD ===")

df = pd.read_parquet(RUTA)

print("\nCategorías originales de altitud:")
print(
    df["altitude_category"]
    .value_counts(dropna=False)
    .to_string()
)

datos = df.dropna(
    subset=["stuff_plus", "altitude_category", "AutoPitchType"]
).copy()

# Resumen general por categoría
general = (
    datos.groupby("altitude_category", observed=True)
    .agg(
        lanzamientos=("stuff_plus", "count"),
        pitchers=("pitcher_anon_id", "nunique"),
        stuff_promedio=("stuff_plus", "mean"),
        stuff_mediana=("stuff_plus", "median"),
        velocidad=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("HorzBreak", "mean")
    )
    .reset_index()
)

print("\n=== RESULTADOS GENERALES ===")
print(general.round(2).to_string(index=False))

# Comparación por tipo de lanzamiento
tipos = (
    datos.groupby(
        ["altitude_category", "AutoPitchType"],
        observed=True
    )
    .agg(
        lanzamientos=("stuff_plus", "count"),
        pitchers=("pitcher_anon_id", "nunique"),
        stuff_promedio=("stuff_plus", "mean"),
        velocidad=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("HorzBreak", "mean")
    )
    .reset_index()
)

tipos = tipos.sort_values(
    ["AutoPitchType", "altitude_category"]
)

print("\n=== STUFF+ POR TIPO Y ALTITUD ===")
print(
    tipos[[
        "altitude_category",
        "AutoPitchType",
        "lanzamientos",
        "stuff_promedio",
        "ivb",
        "hb"
    ]].round(2).to_string(index=False)
)

SALIDA.parent.mkdir(exist_ok=True)
tipos.to_csv(SALIDA, index=False, encoding="utf-8-sig")

print(f"\nResultados guardados en: {SALIDA}")