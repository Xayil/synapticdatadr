from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reclutamiento.stuff_plus import (
    enriquecer,
    estimador_desde_artefacto,
    stuff_plus_desde_probabilidad,
    FEATURES_NUM,
    TIPOS_MODELO,
)

ROOT = Path(__file__).resolve().parents[1]
RUTA_BASE = ROOT / "database/stuff_model_df.parquet"
RUTA_MODELO = ROOT / "modelos/stuff_plus.joblib"
RUTA_SALIDA = ROOT / "resultados/stuff_plus_lanzamientos.parquet"

print("Cargando modelo Stuff+ original...")
artefacto = joblib.load(RUTA_MODELO)
modelo = estimador_desde_artefacto(artefacto)

print("Cargando base de lanzamientos...")
df = pd.read_parquet(RUTA_BASE)
print(f"Lanzamientos originales: {len(df):,}")

# Conservar todas las filas de la base original.
df["p_whiff"] = np.nan
df["stuff_plus"] = np.nan

# Identificar lanzamientos compatibles con el modelo.
columnas = [
    "RelSpeed", "SpinRate", "SpinAxis",
    "Extension", "RelHeight", "RelSide",
    "InducedVertBreak", "HorzBreak",
    "PitcherThrows", "AutoPitchType"
]

faltantes = [c for c in columnas if c not in df.columns]
if faltantes:
    raise ValueError(f"Columnas faltantes: {faltantes}")

validos = (
    df[columnas].notna().all(axis=1)
    & df["AutoPitchType"].isin(TIPOS_MODELO)
    & df["PitcherThrows"].isin(["Right", "Left"])
    & df["RelSpeed"].between(55, 105)
    & df["SpinRate"].between(200, 4000)
    & df["Extension"].between(3, 9)
    & df["RelHeight"].between(3, 8)
)

indices = df.index[validos]
print(f"Lanzamientos candidatos: {len(indices):,}")

# Aplicar la preparación física original.
datos = enriquecer(df.loc[indices].copy())

# Excluir entradas no finitas.
mascara = np.isfinite(
    datos[FEATURES_NUM].to_numpy(dtype=float)
).all(axis=1)

datos = datos.loc[mascara]
indices = indices[mascara]

print(f"Lanzamientos evaluables: {len(datos):,}")

# Predicción con parámetros ya entrenados.
probabilidades = modelo.probabilidades(datos)

puntuaciones = stuff_plus_desde_probabilidad(
    probabilidades,
    artefacto["mu_p"],
    artefacto["sigma_p"]
)

df.loc[indices, "p_whiff"] = probabilidades
df.loc[indices, "stuff_plus"] = puntuaciones

# Guardar resultados sin sobrescribir la base.
RUTA_SALIDA.parent.mkdir(exist_ok=True)
df.to_parquet(RUTA_SALIDA, index=False)

print("\n=== RESULTADOS STUFF+ ===")
print(f"Lanzamientos evaluados: {len(indices):,}")
print(f"Stuff+ promedio: {np.mean(puntuaciones):.2f}")
print(f"Stuff+ mínimo: {np.min(puntuaciones):.2f}")
print(f"Stuff+ máximo: {np.max(puntuaciones):.2f}")
print(f"\nArchivo guardado: {RUTA_SALIDA}")