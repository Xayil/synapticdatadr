"""Stuff+ de la organización, con la escala del notebook.

El notebook define

    p = P(whiff | X) = 1 / (1 + e^{-z})
    z = β · X*
    Stuff+ = 100 + 10 (p − μ_p) / σ_p

X lleva velocidad, spin, IVB, quiebre horizontal, el residuo contra el
modelo físico (ΔIVB, ΔHB), el factor de spin S, extensión, punto de
liberación y densidad del aire. El tipo de lanzamiento y la mano entran
como variables indicadoras.

La escala ya no mezcla todos los tipos. El notebook la fija dentro de
cada tipo T:

    Stuff+ = 100 + 10 (p − μ_{p,T}) / σ_{p,T}

100 es el lanzamiento promedio de ese tipo. μ y σ salen del entrenamiento.
La ubicación no entra aquí: eso es Location+, en reclutamiento/location_plus.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reclutamiento.fisica import (
    ALTITUD_DEFAULT,
    densidad_por_altitud,
    factor_spin,
    quiebre_modelo,
    signo_lado_brazo,
)

VERSION = "notebook-stuff-location-v2"

ROOT = Path(__file__).resolve().parents[1]
RUTA_DATOS = ROOT / "database" / "stuff_model_df.parquet"
RUTA_MODELO = ROOT / "modelos" / "stuff_plus.joblib"
RUTA_REFERENCIA = ROOT / "modelos" / "referencia_pitchers.parquet"

TIPOS_MODELO = [
    "Four-Seam",
    "Sinker",
    "Slider",
    "Curveball",
    "Changeup",
    "Cutter",
    "Splitter",
    "Sweeper",
]

FEATURES_NUM = [
    "RelSpeed",
    "SpinRate",
    "InducedVertBreak",
    "HB_mano",
    "delta_ivb",
    "delta_hb",
    "S",
    "Extension",
    "RelHeight",
    "RelSide_mano",
    "rho",
]
FEATURES_CAT = ["AutoPitchType", "PitcherThrows"]

COLUMNAS_BASE = [
    "pitcher_anon_id",
    "PitcherThrows",
    "altitude_category",
    "AutoPitchType",
    "RelSpeed",
    "EffectiveVelo",
    "ZoneSpeed",
    "SpinRate",
    "SpinAxis",
    "RelHeight",
    "RelSide",
    "Extension",
    "VertBreak",
    "InducedVertBreak",
    "HorzBreak",
    "VertApprAngle",
    "HorzApprAngle",
    "PlateLocSide",
    "PlateLocHeight",
    "Balls",
    "Strikes",
    "count",
    "BatterSide",
    "is_whiff",
]

METRICAS_PROMEDIO = [
    "RelSpeed",
    "EffectiveVelo",
    "ZoneSpeed",
    "SpinRate",
    "SpinAxis",
    "RelHeight",
    "RelSide",
    "Extension",
    "VertBreak",
    "InducedVertBreak",
    "HorzBreak",
    "VertApprAngle",
    "HorzApprAngle",
    "stuff_plus",
    "p_whiff",
    "location_plus",
    "PlateLocSide",
    "PlateLocHeight",
]


def stuff_plus_desde_probabilidad(probabilidad, mu_p: float, sigma_p: float):
    """Stuff+ = 100 + 10 (p − μ_p) / σ_p."""
    sigma = sigma_p if abs(sigma_p) > 1e-12 else 1.0
    return 100.0 + 10.0 * (np.asarray(probabilidad, dtype=float) - mu_p) / sigma


def stuff_plus_por_tipo(probabilidad, tipos, mu_por_tipo: dict, sigma_por_tipo: dict, mu_p: float, sigma_p: float):
    """Escala del notebook: cada tipo se compara contra su propio promedio."""
    mu = np.array([mu_por_tipo.get(tipo, mu_p) for tipo in tipos], dtype=float)
    sigma = np.array([sigma_por_tipo.get(tipo, sigma_p) for tipo in tipos], dtype=float)
    sigma = np.where(np.abs(sigma) < 1e-12, 1.0, sigma)
    return 100.0 + 10.0 * (np.asarray(probabilidad, dtype=float) - mu) / sigma


def enriquecer(df: pd.DataFrame, altitud: str | None = None) -> pd.DataFrame:
    """Agrega S, ρ, el quiebre del modelo físico y los residuos ΔIVB y ΔHB."""
    out = df.copy().reset_index(drop=True)
    if altitud is not None:
        rho = np.full(len(out), float(densidad_por_altitud(altitud)))
    elif "altitude_category" in out.columns:
        rho = np.asarray(densidad_por_altitud(out["altitude_category"]), dtype=float)
    else:
        rho = np.full(len(out), float(densidad_por_altitud(ALTITUD_DEFAULT)))

    zone = out["ZoneSpeed"] if "ZoneSpeed" in out.columns else None
    ivb_modelo, hb_modelo = quiebre_modelo(
        out["RelSpeed"].to_numpy(dtype=float),
        out["SpinRate"].to_numpy(dtype=float),
        out["SpinAxis"].to_numpy(dtype=float),
        out["Extension"].to_numpy(dtype=float),
        rho,
        None if zone is None else zone.to_numpy(dtype=float),
    )
    signo = signo_lado_brazo(out["PitcherThrows"])
    out["rho"] = rho
    out["S"] = np.atleast_1d(factor_spin(out["SpinRate"], out["RelSpeed"]))
    out["ivb_modelo"] = np.atleast_1d(np.asarray(ivb_modelo, dtype=float))
    out["hb_modelo"] = np.atleast_1d(np.asarray(hb_modelo, dtype=float))
    out["delta_ivb"] = out["InducedVertBreak"].to_numpy(dtype=float) - out["ivb_modelo"]
    out["HB_mano"] = out["HorzBreak"].to_numpy(dtype=float) * signo
    out["delta_hb"] = (out["HorzBreak"].to_numpy(dtype=float) - out["hb_modelo"]) * signo
    out["RelSide_mano"] = out["RelSide"].to_numpy(dtype=float) * signo
    return out


def _filtrar_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
    datos = df.copy()
    datos["EffectiveVelo"] = pd.to_numeric(datos["EffectiveVelo"], errors="coerce")
    datos["AutoPitchType"] = datos["AutoPitchType"].astype(str)
    datos = datos[datos["AutoPitchType"].isin(TIPOS_MODELO)]
    datos = datos[datos["PitcherThrows"].isin(["Right", "Left"])]
    obligatorias = [
        "RelSpeed",
        "SpinRate",
        "SpinAxis",
        "Extension",
        "RelHeight",
        "RelSide",
        "InducedVertBreak",
        "HorzBreak",
        "ZoneSpeed",
        "is_whiff",
        "pitcher_anon_id",
    ]
    datos = datos.dropna(subset=obligatorias)
    datos = datos[datos["RelSpeed"].between(55, 105)]
    datos = datos[datos["SpinRate"].between(200, 4000)]
    datos = datos[datos["Extension"].between(3, 9)]
    datos = datos[datos["RelHeight"].between(3, 8)]
    return datos.reset_index(drop=True)


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def _auc(y: np.ndarray, probabilidad: np.ndarray) -> float:
    """Área bajo la curva por rangos, sin depender de scikit-learn."""
    positivos = float(y.sum())
    negativos = float(len(y) - positivos)
    if positivos == 0 or negativos == 0:
        return float("nan")
    orden = np.argsort(probabilidad, kind="mergesort")
    rangos = np.empty(len(y), dtype=float)
    rangos[orden] = np.arange(1, len(y) + 1, dtype=float)
    suma = float(rangos[y == 1].sum())
    return (suma - positivos * (positivos + 1.0) / 2.0) / (positivos * negativos)


def _log_loss(y: np.ndarray, probabilidad: np.ndarray) -> float:
    p = np.clip(probabilidad, 1e-12, 1.0 - 1e-12)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _particion(y: np.ndarray, prueba: float = 0.2, semilla: int = 42):
    """80/20 estratificado por whiff. La escala de Stuff+ se fija solo con entrenamiento."""
    rng = np.random.default_rng(semilla)
    indices = np.arange(len(y))
    entrenamiento = []
    evaluacion = []
    for clase in (0, 1):
        del_clase = indices[y == clase]
        rng.shuffle(del_clase)
        corte = int(round(len(del_clase) * (1.0 - prueba)))
        entrenamiento.append(del_clase[:corte])
        evaluacion.append(del_clase[corte:])
    return np.sort(np.concatenate(entrenamiento)), np.sort(np.concatenate(evaluacion))


class ModeloStuffPlus:
    """Logística por máxima verosimilitud, con X* estandarizado y dummies de tipo y mano.

    La referencia de las indicadoras es Four-Seam y mano derecha, como en el
    esquema D_SL, D_CB, D_CH del notebook. El ajuste es IRLS.
    """

    def __init__(self):
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.nombres_: list[str] = []
        self.tipos_dummy_: list[str] = []
        self.manos_dummy_: list[str] = []
        self.tipo_base_: str = "Four-Seam"
        self.mano_base_: str = "Right"

    def matriz(self, datos: pd.DataFrame) -> np.ndarray:
        z = datos[FEATURES_NUM].to_numpy(dtype=float)
        z = (z - self.mu_) / self.sigma_
        columnas = [np.ones((len(datos), 1)), z]
        tipo = datos["AutoPitchType"].astype(str).to_numpy()
        mano = datos["PitcherThrows"].astype(str).to_numpy()
        for nivel in self.tipos_dummy_:
            columnas.append((tipo == nivel).astype(float)[:, None])
        for nivel in self.manos_dummy_:
            columnas.append((mano == nivel).astype(float)[:, None])
        return np.hstack(columnas)

    def fit(self, datos: pd.DataFrame, y: np.ndarray, max_iter: int = 40) -> "ModeloStuffPlus":
        valores = datos[FEATURES_NUM].to_numpy(dtype=float)
        self.mu_ = valores.mean(axis=0)
        sigma = valores.std(axis=0)
        self.sigma_ = np.where(sigma < 1e-8, 1.0, sigma)

        tipos = [tipo for tipo in TIPOS_MODELO if (datos["AutoPitchType"] == tipo).any()]
        self.tipo_base_ = "Four-Seam" if "Four-Seam" in tipos else tipos[0]
        self.tipos_dummy_ = [tipo for tipo in tipos if tipo != self.tipo_base_]
        manos = [mano for mano in ("Right", "Left") if (datos["PitcherThrows"] == mano).any()]
        self.mano_base_ = "Right" if "Right" in manos else manos[0]
        self.manos_dummy_ = [mano for mano in manos if mano != self.mano_base_]
        self.nombres_ = (
            ["intercepto"]
            + FEATURES_NUM
            + [f"AutoPitchType_{tipo}" for tipo in self.tipos_dummy_]
            + [f"PitcherThrows_{mano}" for mano in self.manos_dummy_]
        )

        X = self.matriz(datos)
        beta = np.zeros(X.shape[1])
        penalizacion = np.ones(X.shape[1])
        penalizacion[0] = 0.0
        for _ in range(max_iter):
            probabilidad = _sigmoid(X @ beta)
            peso = np.maximum(probabilidad * (1.0 - probabilidad), 1e-6)
            respuesta = X @ beta + (y - probabilidad) / peso
            hess = X.T @ (X * peso[:, None])
            hess[np.arange(len(beta)), np.arange(len(beta))] += penalizacion
            beta_nuevo = np.linalg.solve(hess, X.T @ (peso * respuesta))
            if np.max(np.abs(beta_nuevo - beta)) < 1e-6:
                beta = beta_nuevo
                break
            beta = beta_nuevo
        self.beta_ = beta
        return self

    def probabilidades(self, datos: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.matriz(datos) @ self.beta_)

    def coeficientes(self) -> list[dict]:
        return [
            {"variable": nombre, "beta": float(beta)}
            for nombre, beta in zip(self.nombres_, self.beta_)
        ]


def construir_artefactos(forzar: bool = False) -> dict:
    """Entrena el Stuff+ y guarda el perfil promedio de cada pitcher."""
    if (
        not forzar
        and RUTA_MODELO.exists()
        and RUTA_REFERENCIA.exists()
    ):
        return cargar_modelo()

    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No está la base de la organización en {RUTA_DATOS}")

    print("Cargando lanzamientos de la organización...")
    bruto = pd.read_parquet(RUTA_DATOS, columns=COLUMNAS_BASE)
    datos = _filtrar_entrenamiento(bruto)
    print(f"Lanzamientos válidos: {len(datos):,}")
    datos = enriquecer(datos)

    completos = np.isfinite(datos[FEATURES_NUM]).all(axis=1)
    datos = datos.loc[completos].reset_index(drop=True)
    y = datos["is_whiff"].to_numpy(dtype=float)
    indice_train, indice_test = _particion(y)
    print("Entrenando la regresión logística de whiff...")
    modelo = ModeloStuffPlus().fit(datos.iloc[indice_train].reset_index(drop=True), y[indice_train])

    p_train = modelo.probabilidades(datos.iloc[indice_train])
    tipos_train = datos.iloc[indice_train]["AutoPitchType"].astype(str).to_numpy()
    mu_p = float(p_train.mean())
    sigma_p = float(p_train.std(ddof=0))
    mu_por_tipo = {}
    sigma_por_tipo = {}
    for tipo in TIPOS_MODELO:
        muestra = p_train[tipos_train == tipo]
        if len(muestra) == 0:
            continue
        mu_por_tipo[tipo] = float(muestra.mean())
        sigma_tipo = float(muestra.std(ddof=0))
        sigma_por_tipo[tipo] = sigma_tipo if sigma_tipo > 1e-12 else 1.0
    p_test = modelo.probabilidades(datos.iloc[indice_test])
    auc = _auc(y[indice_test], p_test)
    perdida = _log_loss(y[indice_test], p_test)
    print(f"AUC de prueba: {auc:.3f}   log loss: {perdida:.3f}")
    print(f"Escala global de referencia: mu_p={mu_p:.4f}  sigma_p={sigma_p:.4f}")
    for tipo in TIPOS_MODELO:
        if tipo in mu_por_tipo:
            print(f"  {tipo}: mu_p={mu_por_tipo[tipo]:.4f}  sigma_p={sigma_por_tipo[tipo]:.4f}")
    for item in modelo.coeficientes():
        if item["variable"] in {"RelSpeed", "SpinRate", "InducedVertBreak", "HB_mano", "S", "intercepto"}:
            print(f"  beta {item['variable']}: {item['beta']:+.3f}")

    p_todo = modelo.probabilidades(datos)
    datos["p_whiff"] = p_todo
    datos["stuff_plus"] = stuff_plus_por_tipo(
        p_todo,
        datos["AutoPitchType"].astype(str).to_numpy(),
        mu_por_tipo,
        sigma_por_tipo,
        mu_p,
        sigma_p,
    )

    from reclutamiento.location_plus import entrenar_location

    print("Entrenando Location+...")
    estado_location, datos = entrenar_location(datos, indice_train, indice_test)

    agrupado = datos.groupby(
        ["pitcher_anon_id", "AutoPitchType", "PitcherThrows"],
        observed=True,
    )
    referencia = agrupado[METRICAS_PROMEDIO].mean()
    referencia["n_lanzamientos"] = agrupado.size()
    referencia = referencia.reset_index()

    artefacto = {
        "version": VERSION,
        "modelo": {
            "mu": modelo.mu_,
            "sigma": modelo.sigma_,
            "beta": modelo.beta_,
            "nombres": modelo.nombres_,
            "tipos_dummy": modelo.tipos_dummy_,
            "manos_dummy": modelo.manos_dummy_,
            "tipo_base": modelo.tipo_base_,
            "mano_base": modelo.mano_base_,
        },
        "mu_p": mu_p,
        "sigma_p": sigma_p,
        "mu_p_tipo": mu_por_tipo,
        "sigma_p_tipo": sigma_por_tipo,
        "location": estado_location,
        "auc": auc,
        "log_loss": perdida,
        "n_train": int(len(indice_train)),
        "n_test": int(len(indice_test)),
        "n_lanzamientos": int(len(datos)),
        "whiff_rate": float(y.mean()),
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
        "tipos": TIPOS_MODELO,
        "coeficientes": modelo.coeficientes(),
    }
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, RUTA_MODELO)
    referencia.to_parquet(RUTA_REFERENCIA, index=False)
    print(f"Pitchers en la referencia: {referencia['pitcher_anon_id'].nunique():,}")
    print(f"Guardado en {RUTA_MODELO.name} y {RUTA_REFERENCIA.name}")
    return artefacto


def cargar_modelo() -> dict:
    if not RUTA_MODELO.exists() or not RUTA_REFERENCIA.exists():
        return construir_artefactos(forzar=True)
    artefacto = joblib.load(RUTA_MODELO)
    if artefacto.get("version") != VERSION or "location" not in artefacto or "mu_p_tipo" not in artefacto:
        return construir_artefactos(forzar=True)
    return artefacto


_COLUMNAS_ESCENARIO = [
    "pitcher_anon_id",
    "AutoPitchType",
    "PitcherThrows",
    "RelSpeed",
    "SpinRate",
    "InducedVertBreak",
    "HorzBreak",
    "n_lanzamientos",
    "stuff_plus",
    "PlateLocSide",
    "PlateLocHeight",
]


def _vacio_escenario() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNAS_ESCENARIO)


@lru_cache(maxsize=16)
def _pitches_validos(ids: tuple[str, ...]) -> pd.DataFrame:
    """Lanzamientos ya filtrados de esos pitchers. El aire se aplica después."""
    if not ids or not RUTA_DATOS.exists():
        return _vacio_escenario()
    bruto = pd.read_parquet(
        RUTA_DATOS,
        columns=list(dict.fromkeys(COLUMNAS_BASE + ["Outs"])),
        filters=[("pitcher_anon_id", "in", list(ids))],
    )
    return _filtrar_entrenamiento(bruto)


def lanzamientos_en_escenario(
    ids: list[str],
    estadio: str,
    artefacto: dict | None = None,
    referencia: pd.DataFrame | None = None,
    pitching: dict | None = None,
) -> pd.DataFrame:
    """Stuff+ de los pitchers del equipo con el aire del estadio elegido.

    Cada lanzamiento conserva su velocidad, giro y quiebre. Cambia la densidad
    y, con ella, el quiebre que el modelo físico esperaba y el Stuff+.
    """
    ids_limpios = tuple(sorted({identificador for identificador in ids if identificador}))
    if not ids_limpios:
        return _vacio_escenario()
    if artefacto is None:
        artefacto = cargar_modelo()
    aire = estadio or ALTITUD_DEFAULT
    try:
        base = _pitches_validos(ids_limpios).copy()
    except (OSError, ValueError, KeyError):
        base = _vacio_escenario()

    partes = []
    if not base.empty and "pitcher_anon_id" in base.columns:
        puntuados = enriquecer(base, altitud=aire)
        completos = np.isfinite(puntuados[FEATURES_NUM]).all(axis=1)
        puntuados = puntuados.loc[completos].copy()
        if not puntuados.empty:
            probabilidad = estimador_desde_artefacto(artefacto).probabilidades(puntuados)
            puntuados["stuff_plus"] = stuff_plus_por_tipo(
                probabilidad,
                puntuados["AutoPitchType"].astype(str).to_numpy(),
                artefacto.get("mu_p_tipo", {}),
                artefacto.get("sigma_p_tipo", {}),
                artefacto["mu_p"],
                artefacto["sigma_p"],
            )
            puntuados["n_lanzamientos"] = 1
            if pitching is not None:
                from reclutamiento.pitching_plus import completar_capas

                puntuados = completar_capas(puntuados, artefacto, pitching)
            columnas = list(_COLUMNAS_ESCENARIO)
            if "pitching_plus" in puntuados.columns:
                columnas.append("pitching_plus")
            partes.append(puntuados[columnas])

    encontrados = set(partes[0]["pitcher_anon_id"]) if partes else set()
    faltan = [identificador for identificador in ids_limpios if identificador not in encontrados]
    if faltan and referencia is not None:
        partes.append(_perfil_en_escenario(referencia, faltan, aire, artefacto))
    if not partes:
        return _vacio_escenario()
    return pd.concat(partes, ignore_index=True)


def _perfil_en_escenario(
    referencia: pd.DataFrame,
    ids: list[str],
    estadio: str,
    artefacto: dict,
) -> pd.DataFrame:
    """Si no hay lanzamiento suelto, el promedio del tipo se califica en ese aire."""
    datos = referencia[referencia["pitcher_anon_id"].isin(ids)].copy()
    if datos.empty:
        return _vacio_escenario()
    stuff = []
    for _, fila in datos.iterrows():
        campos = {columna: fila[columna] for columna in fila.index if columna in {
            "AutoPitchType",
            "PitcherThrows",
            "RelSpeed",
            "EffectiveVelo",
            "ZoneSpeed",
            "SpinRate",
            "SpinAxis",
            "RelHeight",
            "RelSide",
            "Extension",
            "InducedVertBreak",
            "HorzBreak",
            "VertBreak",
            "PlateLocSide",
            "PlateLocHeight",
        }}
        campos["altitude_category"] = estadio
        try:
            stuff.append(puntuar_lanzamiento(campos, artefacto)["stuff_plus"])
        except ValueError:
            stuff.append(fila["stuff_plus"])
    datos["stuff_plus"] = stuff
    return datos[_COLUMNAS_ESCENARIO].reset_index(drop=True)


def cargar_referencia() -> pd.DataFrame:
    if not RUTA_REFERENCIA.exists() or not RUTA_MODELO.exists():
        construir_artefactos(forzar=True)
    return pd.read_parquet(RUTA_REFERENCIA)


def estimador_desde_artefacto(artefacto: dict) -> ModeloStuffPlus:
    estado = artefacto["modelo"]
    modelo = ModeloStuffPlus()
    modelo.mu_ = estado["mu"]
    modelo.sigma_ = estado["sigma"]
    modelo.beta_ = estado["beta"]
    modelo.nombres_ = estado["nombres"]
    modelo.tipos_dummy_ = estado["tipos_dummy"]
    modelo.manos_dummy_ = estado["manos_dummy"]
    modelo.tipo_base_ = estado["tipo_base"]
    modelo.mano_base_ = estado["mano_base"]
    return modelo


def puntuar_lanzamiento(datos: dict, artefacto: dict | None = None) -> dict:
    """Pasa un lanzamiento (o el promedio de un tipo) por el Stuff+."""
    if artefacto is None:
        artefacto = cargar_modelo()
    frame = pd.DataFrame([datos])
    altitud = datos.get("altitude_category") or ALTITUD_DEFAULT
    frame = enriquecer(frame, altitud=altitud)
    X = frame[FEATURES_NUM + FEATURES_CAT]
    if not np.isfinite(X[FEATURES_NUM]).all(axis=None):
        raise ValueError("Faltan métricas físicas para calcular el Stuff+.")
    probabilidad = float(estimador_desde_artefacto(artefacto).probabilidades(frame)[0])
    tipo = str(datos.get("AutoPitchType"))
    mu_tipo = artefacto.get("mu_p_tipo", {}).get(tipo, artefacto["mu_p"])
    sigma_tipo = artefacto.get("sigma_p_tipo", {}).get(tipo, artefacto["sigma_p"])
    stuff = float(stuff_plus_desde_probabilidad(probabilidad, mu_tipo, sigma_tipo))
    fila = frame.iloc[0]
    resultado = {
        "p_whiff": probabilidad,
        "stuff_plus": stuff,
        "delta_ivb": float(fila["delta_ivb"]),
        "delta_hb": float(fila["delta_hb"]),
        "ivb_modelo": float(fila["ivb_modelo"]),
        "hb_modelo": float(fila["hb_modelo"]),
        "S": float(fila["S"]),
        "rho": float(fila["rho"]),
    }
    if "location" in artefacto:
        from reclutamiento.location_plus import puntuar_location

        resultado.update(puntuar_location(datos, stuff, artefacto["location"]))
    return resultado


if __name__ == "__main__":
    construir_artefactos(forzar=True)
