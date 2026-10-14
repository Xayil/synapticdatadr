import sys
import time
import types
from pathlib import Path

import pandas as pd


def _preparar_scipy_para_dtale():
    """Windows bloquea dos extensiones de SciPy que D-Tale importa al arrancar.

    El visor de la tabla no las usa. Si el sistema deja cargar los binarios,
    se usan los reales.
    """
    import ctypes

    import scipy

    carpeta = Path(scipy.__file__).parent / "signal"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    kernel.LoadLibraryExW.restype = ctypes.c_void_p
    kernel.FreeLibrary.argtypes = [ctypes.c_void_p]

    def bloqueado(nombre):
        archivos = list(carpeta.glob(f"{nombre}*.pyd"))
        if not archivos:
            return False
        handle = kernel.LoadLibraryExW(str(archivos[0]), None, 0x00000008)
        if handle:
            kernel.FreeLibrary(handle)
            return False
        return ctypes.get_last_error() == 4551

    if not (bloqueado("_spline") or bloqueado("_upfirdn_apply")):
        return

    def no_disponible(*_args, **_kwargs):
        raise RuntimeError(
            "Esa función de SciPy está bloqueada por Control de aplicaciones de Windows."
        )

    exportados = (
        "_output_len",
        "_apply",
        "mode_enum",
        "sepfir2d",
        "symiirorder1_ic",
        "symiirorder2_ic_fwd",
        "symiirorder2_ic_bwd",
    )
    for nombre in ("scipy.signal._upfirdn_apply", "scipy.signal._spline"):
        modulo = types.ModuleType(nombre)
        for atributo in exportados:
            setattr(modulo, atributo, no_disponible)
        sys.modules[nombre] = modulo


_preparar_scipy_para_dtale()

import dtale

RUTA = Path(__file__).resolve().parents[1] / "database" / "stuff_model_df.parquet"

print("Cargando dataset...", flush=True)
df = pd.read_parquet(RUTA)

print("Iniciando servidor D-Tale...", flush=True)
# Forzamos IP local explicita
d = dtale.show(df, host="127.0.0.1", open_browser=False)

# Obtenemos la URL de forma directa
url = d._url
print(f"\n¡Servidor listo! Abre este enlace en tu navegador:\n{url}\n", flush=True)

# Abre la pestaña automáticamente
dtale.instances()
d.open_browser()

# Mantiene el proceso activo en terminal
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nServidor detenido.")
