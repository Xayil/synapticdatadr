from pathlib import Path
import re

ruta = Path("modelos/stuff_plus.joblib")

if not ruta.exists():
    raise FileNotFoundError(f"No existe: {ruta}")

contenido = ruta.read_bytes()

print("=== INSPECCIÓN STUFF+ ===")
print(f"Archivo: {ruta}")
print(f"Tamaño: {len(contenido):,} bytes")

print("\nTextos identificables:")

fragmentos = re.findall(
    rb"[\x20-\x7e]{4,}",
    contenido
)

for fragmento in fragmentos:
    print("-", fragmento.decode("ascii"))

print("\nInspección finalizada.")
print("No se ejecutó ni modificó el modelo.")
