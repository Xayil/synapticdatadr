from pathlib import Path
import pandas as pd

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[1]

ENTRADA = ROOT / "resultados/stuff_plus_lanzamientos.parquet"
SALIDA = ROOT / "resultados/ranking_pitchers.csv"

print("\n=== RANKING DE PITCHERS POR STUFF+ ===")

# Cargar resultados del modelo original
df = pd.read_parquet(ENTRADA)

# Solo lanzamientos con puntuación válida
df = df.dropna(subset=["pitcher_anon_id", "stuff_plus"])

print(f"Lanzamientos evaluados: {len(df):,}")
print(f"Pitchers identificados: {df['pitcher_anon_id'].nunique():,}")

# Calcular estadísticas por pitcher
ranking = (
    df.groupby("pitcher_anon_id")
    .agg(
        lanzamientos=("stuff_plus", "count"),
        stuff_promedio=("stuff_plus", "mean"),
        stuff_mediana=("stuff_plus", "median"),
        stuff_maximo=("stuff_plus", "max"),
        stuff_minimo=("stuff_plus", "min"),
        tipos_pitch=("AutoPitchType", "nunique"),
    )
    .reset_index()
)

# Mínimo de lanzamientos para comparar pitchers
MIN_LANZAMIENTOS = 100

ranking = ranking[
    ranking["lanzamientos"] >= MIN_LANZAMIENTOS
].copy()

# Ordenar por Stuff+ promedio
ranking = ranking.sort_values(
    ["stuff_promedio", "lanzamientos"],
    ascending=[False, False]
).reset_index(drop=True)

ranking.insert(0, "posicion", ranking.index + 1)

# Redondear estadísticas para presentación
columnas = [
    "stuff_promedio",
    "stuff_mediana",
    "stuff_maximo",
    "stuff_minimo",
]

ranking[columnas] = ranking[columnas].round(2)

# Guardar ranking
ranking.to_csv(SALIDA, index=False, encoding="utf-8-sig")

print(f"\nPitchers con al menos {MIN_LANZAMIENTOS} lanzamientos: {len(ranking):,}")

print("\n=== TOP 20 PITCHERS ===")
print(ranking.head(20).to_string(index=False))

print(f"\nRanking completo guardado en:\n{SALIDA}")