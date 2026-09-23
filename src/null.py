import pandas as pd
import dtale


# Esto sirve para cargar la base de datos completa desde el archivo Parquet
df = pd.read_parquet("database/stuff_model_df.parquet")

# Esto sirve para contar cuántos valores nulos existen en cada columna
nulos_por_columna = df.isnull().sum()

# Esto sirve para conservar únicamente las columnas que tienen al menos un valor nulo
columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]

# Esto sirve para mostrar las columnas que contienen valores nulos y la cantidad encontrada
print("\n==========================================")
print("          VALORES NULOS")
print("==========================================")
print(columnas_con_nulos)

# Esto sirve para calcular qué porcentaje de cada columna corresponde a valores nulos
porcentaje_nulos = (
    columnas_con_nulos / len(df) * 100
).round(2)

# Esto sirve para organizar la cantidad y el porcentaje de valores nulos en una sola tabla
resumen_nulos = pd.DataFrame({
    "Valores nulos": columnas_con_nulos,
    "Porcentaje de nulos": porcentaje_nulos
})

# Esto sirve para mostrar un resumen más completo de los valores nulos encontrados
print("\n==========================================")
print("       RESUMEN DE VALORES NULOS")
print("==========================================")
print(resumen_nulos)

# Esto sirve para contar todos los valores nulos existentes en toda la base de datos
total_nulos = df.isnull().sum().sum()

# Esto sirve para identificar las filas que contienen al menos un valor nulo
filas_con_nulos = df[df.isnull().any(axis=1)]

# Esto sirve para contar cuántas filas tienen al menos un valor nulo
cantidad_filas_con_nulos = len(filas_con_nulos)

# Esto sirve para mostrar el resumen general de valores nulos de la base de datos
print("\n==========================================")
print("          RESUMEN GENERAL")
print("==========================================")
print("Total de valores nulos:", total_nulos)
print("Filas con al menos un valor nulo:", cantidad_filas_con_nulos)
print("Filas totales de la base:", len(df))

# Esto sirve para obtener los nombres de las columnas que contienen valores nulos
columnas_nulas = columnas_con_nulos.index.tolist()

# Esto sirve para conservar únicamente las filas con nulos y las columnas que contienen nulos
df_nulos = df.loc[
    df[columnas_nulas].isnull().any(axis=1),
    columnas_nulas
].copy()

# Esto sirve para reiniciar el índice de las filas que contienen valores nulos
df_nulos = df_nulos.reset_index(drop=True)

# Esto sirve para comprobar cuántas filas y columnas se enviarán a D-Tale
print("\nTamaño de df_nulos:", df_nulos.shape)

# Esto sirve para comprobar exactamente qué columnas se enviarán a D-Tale
print("\nColumnas de df_nulos:")
print(df_nulos.columns.tolist())

# Esto sirve para abrir en D-Tale las filas que contienen al menos un valor nulo
d = dtale.show(df_nulos, host="127.0.0.1")

# Esto sirve para obtener la dirección web de D-Tale
url_dtale = d._url

# Esto sirve para mostrar la dirección de D-Tale en la terminal
print("\nD-Tale está disponible en:")
print(url_dtale)

# Esto sirve para intentar abrir automáticamente D-Tale en el navegador
d.open_browser()

# Esto sirve para mantener activo D-Tale mientras se revisan los valores nulos
input("\nPresiona Enter cuando quieras cerrar D-Tale...")
