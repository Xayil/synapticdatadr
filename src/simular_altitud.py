
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from reclutamiento.stuff_plus import (
    enriquecer,
    estimador_desde_artefacto,
    stuff_plus_desde_probabilidad,
    FEATURES_NUM,
    TIPOS_MODELO
)

ROOT = Path(__file__).resolve().parents[1]

RUTA_DATOS = ROOT / "resultados/stuff_plus_lanzamientos.parquet"
RUTA_MODELO = ROOT / "modelos/stuff_plus.joblib"
RUTA_SALIDA = ROOT / "resultados/simulacion_altitud_00778.csv"

PITCHER = "pitcher_00778"

print("\n=== SIMULACIÓN DE ALTITUD STUFF+ ===")

# Cargar modelo original, sin entrenar ni modificar.
artefacto = joblib.load(RUTA_MODELO)
modelo = estimador_desde_artefacto(artefacto)

# Seleccionar pitcher.
df = pd.read_parquet(RUTA_DATOS)
datos = df[
    (df["pitcher_anon_id"] == PITCHER)
    & df["stuff_plus"].notna()
    & df["AutoPitchType"].isin(TIPOS_MODELO)
].copy().reset_index(drop=True)

if datos.empty:
    raise ValueError("No hay lanzamientos evaluables.")

# Preparar escenarios atmosféricos.
mar = enriquecer(datos, altitud="No Altitude")
altura = enriquecer(datos, altitud="Extreme Altitude")

# Conservar el residuo físico del lanzamiento.
# Movimiento simulado = movimiento original
# + cambio en el movimiento físico esperado.
altura["InducedVertBreak"] = (
    datos["InducedVertBreak"].to_numpy()
    + altura["ivb_modelo"].to_numpy()
    - mar["ivb_modelo"].to_numpy()
)

altura["HorzBreak"] = (
    datos["HorzBreak"].to_numpy()
    + altura["hb_modelo"].to_numpy()
    - mar["hb_modelo"].to_numpy()
)

# Volver a calcular las variables derivadas del escenario.
altura = enriquecer(
    altura,
    altitud="Extreme Altitude"
)

# Verificar entradas.
validos = (
    np.isfinite(mar[FEATURES_NUM].to_numpy(dtype=float)).all(axis=1)
    & np.isfinite(altura[FEATURES_NUM].to_numpy(dtype=float)).all(axis=1)
)

datos = datos.loc[validos].reset_index(drop=True)
mar = mar.loc[validos].reset_index(drop=True)
altura = altura.loc[validos].reset_index(drop=True)

# Calcular Stuff+ usando exactamente el modelo guardado.
def calcular_stuff(frame):
    p = modelo.probabilidades(frame)
    return stuff_plus_desde_probabilidad(
        p,
        artefacto["mu_p"],
        artefacto["sigma_p"]
    )

stuff_mar = calcular_stuff(mar)
stuff_altura = calcular_stuff(altura)

# Crear tabla comparativa.
resultado = pd.DataFrame({
    "pitcher_anon_id": datos["pitcher_anon_id"],
    "AutoPitchType": datos["AutoPitchType"],
    "stuff_nivel_mar": stuff_mar,
    "stuff_harp_helu": stuff_altura,
    "diferencia": stuff_altura - stuff_mar,
    "ivb_mar": mar["InducedVertBreak"],
    "ivb_harp": altura["InducedVertBreak"],
    "hb_mar": mar["HorzBreak"],
    "hb_harp": altura["HorzBreak"]
})

resumen = (
    resultado.groupby("AutoPitchType", observed=True)
    .agg(
        lanzamientos=("diferencia", "count"),
        stuff_mar=("stuff_nivel_mar", "mean"),
        stuff_harp=("stuff_harp_helu", "mean"),
        diferencia=("diferencia", "mean"),
        ivb_mar=("ivb_mar", "mean"),
        ivb_harp=("ivb_harp", "mean"),
        hb_mar=("hb_mar", "mean"),
        hb_harp=("hb_harp", "mean")
    )
    .sort_values("lanzamientos", ascending=False)
)

print(f"\nPitcher: {PITCHER}")
print(f"Lanzamientos simulados: {len(resultado):,}")

print("\n=== COMPARACIÓN POR TIPO DE PITCH ===")
print(resumen.round(2).to_string())

print("\n=== RESULTADO GENERAL ===")
print(f"Stuff+ nivel del mar: {stuff_mar.mean():.2f}")
print(f"Stuff+ Harp Helú: {stuff_altura.mean():.2f}")
print(f"Diferencia: {(stuff_altura - stuff_mar).mean():+.2f}")

RUTA_SALIDA.parent.mkdir(exist_ok=True)
resultado.to_csv(RUTA_SALIDA, index=False)

print(f"\nArchivo guardado: {RUTA_SALIDA}")
