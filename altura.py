import pandas as pd


# Esto sirve para cargar la base de datos completa desde el archivo Parquet
df = pd.read_parquet("database/stuff_model_df.parquet")


# Esto sirve para mostrar todas las categorías de altura existentes en la base de datos
print("\n==========================================")
print("       CATEGORÍAS DE ALTURA")
print("==========================================")

print(df["altitude_category"].value_counts(dropna=False))


# Esto sirve para contar cuántas categorías de altura diferentes existen
print("\nCategorías diferentes:")
print(df["altitude_category"].nunique())


# Esto sirve para comprobar cuántas categorías de altura aparecen dentro de cada partido
altura_por_partido = (
    df.groupby("game_anon_id")["altitude_category"]
    .nunique()
)


# Esto sirve para comprobar si existen partidos asociados con más de una categoría de altura
print("\nMáximo de categorías de altura encontradas dentro de un mismo partido:")
print(altura_por_partido.max())

# Esto sirve para obtener una sola categoría de altura para cada partido
categoria_por_partido = (
    df.groupby("game_anon_id")["altitude_category"]
    .first()
)


# Esto sirve para contar cuántos partidos existen dentro de cada categoría de altura
partidos_por_altura = categoria_por_partido.value_counts(dropna=False)


# Esto sirve para mostrar la distribución de los partidos según su categoría de altura
print("\n==========================================")
print("       PARTIDOS POR ALTITUD")
print("==========================================")
print(partidos_por_altura)

# Esto sirve para obtener el año y la categoría de altura correspondiente a cada partido
contexto_partidos = (
    df.groupby("game_anon_id")
    .agg({
        "year": "first",
        "altitude_category": "first"
    })
)
# Esto sirve para contar los partidos de cada categoría de altura en cada temporada
partidos_altura_año = pd.crosstab(
    contexto_partidos["year"],
    contexto_partidos["altitude_category"],
    dropna=False
)
#hola 