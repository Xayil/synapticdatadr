import time
import pandas as pd
import dtale

print("Cargando dataset...")
df = pd.read_parquet('database/stuff_model_df.parquet')

print("Iniciando servidor D-Tale...")
# Forzamos IP local explicita
d = dtale.show(df, host='127.0.0.1')

# Obtenemos la URL de forma directa
url = d._url
print(f"\n¡Servidor listo! Abre este enlace en tu navegador:\n{url}\n")

# Abre la pestaña automáticamente
dtale.instances()
d.open_browser()

# Mantiene el proceso activo en terminal
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nServidor detenido.")