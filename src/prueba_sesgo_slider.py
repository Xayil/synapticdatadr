from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUTA = ROOT / "resultados/stuff_plus_lanzamientos.parquet"

df = pd.read_parquet(RUTA)

df = df.dropna(
    subset=["pitcher_anon_id", "AutoPitchType", "stuff_plus"]
)

# Calcular el lanzamiento más utilizado de cada pitcher
uso = (
    df.groupby(["pitcher_anon_id", "AutoPitchType"], observed=True)
    .size()
    .reset_index(name="lanzamientos_tipo")
)

principal = (
    uso.sort_values(
        ["pitcher_anon_id", "lanzamientos_tipo", "AutoPitchType"],
        ascending=[True, False, True]
    )
    .drop_duplicates("pitcher_anon_id")
    .rename(columns={"AutoPitchType": "pitch_principal"})
)

# Calcular Stuff+ general sin modificar puntuaciones
ranking = (
    df.groupby("pitcher_anon_id")
    .agg(
        lanzamientos=("stuff_plus", "count"),
        stuff_promedio=("stuff_plus", "mean")
    )
    .reset_index()
)

ranking = ranking.merge(
    principal[["pitcher_anon_id", "pitch_principal"]],
    on="pitcher_anon_id"
)

# Mantener mínimo de 100 lanzamientos
ranking = ranking[ranking["lanzamientos"] >= 100].copy()

ranking = ranking.sort_values(
    "stuff_promedio", ascending=False
).reset_index(drop=True)

ranking["ranking_original"] = ranking.index + 1

# Grupo A: todos
grupo_a = ranking.copy()

# Grupo B: sin slider como lanzamiento principal
grupo_b = ranking[
    ranking["pitch_principal"] != "Slider"
].copy()

grupo_b = grupo_b.reset_index(drop=True)
grupo_b["ranking_sin_slider_principal"] = grupo_b.index + 1

print("\n=== GRUPO A: TOP 20 GENERAL ===")
print(
    grupo_a.head(20).round(2).to_string(index=False)
)

print("\n=== GRUPO B: TOP 20 SIN SLIDER PRINCIPAL ===")
print(
    grupo_b.head(20).round(2).to_string(index=False)
)

print("\n=== RESUMEN ===")
print("Pitchers totales:", len(grupo_a))
print("Pitchers sin slider principal:", len(grupo_b))

print(
    "Stuff+ promedio entre pitchers del grupo A:",
    round(grupo_a["stuff_promedio"].mean(), 2)
)

print(
    "Stuff+ promedio entre pitchers del grupo B:",
    round(grupo_b["stuff_promedio"].mean(), 2)
)

# Guardar resultados
SALIDA = ROOT / "resultados/ranking_sin_slider_principal.csv"
grupo_b.round(2).to_csv(
    SALIDA, index=False, encoding="utf-8-sig"
)

print("\nArchivo generado:", SALIDA)
