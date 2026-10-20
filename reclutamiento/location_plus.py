"""Location+ de la capa 2 del notebook.

El Stuff+ ya midió el material. Aquí la pregunta es otra: dada esa calidad,
¿la ubicación sube o baja la probabilidad de whiff frente a la ubicación
típica de ese tipo de lanzamiento?

    P_actual = P(whiff | x, z, tipo, cuenta, manos, Stuff+)
    P_ref    = P(whiff | x_ref,T, z_ref,T, tipo, cuenta, manos, Stuff+)
    LE       = P_actual − P_ref
    Location+ = 100 + 10 (LE − μ_LE,T) / σ_LE,T

La superficie de cada tipo es cuadrática, como la primera forma del notebook:

    z = β0 + β1 x + β2 z + β3 x² + β4 z² + β5 x z
        + efectos de tipo, cuenta y mano + λ Stuff+

100 es la ubicación típica de ese tipo. La media y la desviación se fijan
con los lanzamientos de entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CUENTAS = [f"{bolas}-{strikes}" for bolas in range(4) for strikes in range(3)]
CUENTA_BASE = "0-0"
TIPO_BASE = "Four-Seam"
MANO_BASE = "Right"
FEATURES_LUGAR = ["plate_x", "plate_z", "plate_x2", "plate_z2", "plate_xz", "stuff_plus"]


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def _auc(y: np.ndarray, probabilidad: np.ndarray) -> float:
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


def cuenta_de(bolas, strikes) -> str:
    try:
        return f"{int(bolas)}-{int(strikes)}"
    except (TypeError, ValueError):
        return CUENTA_BASE


def preparar_ubicacion(df: pd.DataFrame) -> pd.DataFrame:
    """Deja plate, cuenta, mano del bateador y Stuff+ listos para el modelo."""
    out = df.copy()
    out["plate_x"] = pd.to_numeric(out.get("PlateLocSide"), errors="coerce")
    out["plate_z"] = pd.to_numeric(out.get("PlateLocHeight"), errors="coerce")
    if "count" in out.columns:
        cuenta = out["count"].astype(str).str.strip()
    else:
        cuenta = pd.Series(CUENTA_BASE, index=out.index)
    if "Balls" in out.columns and "Strikes" in out.columns:
        bolas = pd.to_numeric(out["Balls"], errors="coerce")
        strikes = pd.to_numeric(out["Strikes"], errors="coerce")
        armada = bolas.round().astype("Int64").astype(str) + "-" + strikes.round().astype("Int64").astype(str)
        cuenta = armada.where(bolas.notna() & strikes.notna(), cuenta)
    out["count"] = cuenta.where(cuenta.isin(CUENTAS), CUENTA_BASE)
    if "BatterSide" not in out.columns:
        out["BatterSide"] = MANO_BASE
    out["BatterSide"] = out["BatterSide"].where(out["BatterSide"].isin(["Right", "Left"]), other=np.nan)
    out["plate_x2"] = out["plate_x"] ** 2
    out["plate_z2"] = out["plate_z"] ** 2
    out["plate_xz"] = out["plate_x"] * out["plate_z"]
    return out


def filas_validas(df: pd.DataFrame) -> pd.Series:
    return (
        df["plate_x"].between(-3.0, 3.0)
        & df["plate_z"].between(0.5, 4.5)
        & df["BatterSide"].isin(["Right", "Left"])
        & df["PitcherThrows"].isin(["Right", "Left"])
        & np.isfinite(df["stuff_plus"])
        & df["count"].isin(CUENTAS)
    )


class ModeloLocation:
    """Logística de whiff con una superficie distinta por tipo de lanzamiento."""

    def __init__(self):
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.nombres_: list[str] = []
        self.tipos_dummy_: list[str] = []
        self.cuentas_dummy_: list[str] = []
        self.bate_dummy_: list[str] = []
        self.pitcher_dummy_: list[str] = []
        self.tipo_base_: str = TIPO_BASE
        self.referencia_xy_: dict[str, tuple[float, float]] = {}
        self.mu_le_: dict[str, float] = {}
        self.sigma_le_: dict[str, float] = {}

    def matriz(self, datos: pd.DataFrame) -> np.ndarray:
        valores = datos[FEATURES_LUGAR].to_numpy(dtype=float)
        z = (valores - self.mu_) / self.sigma_
        columnas = [np.ones((len(datos), 1)), z]
        tipo = datos["AutoPitchType"].astype(str).to_numpy()
        cuenta = datos["count"].astype(str).to_numpy()
        bate = datos["BatterSide"].astype(str).to_numpy()
        pitcher = datos["PitcherThrows"].astype(str).to_numpy()
        lugar = z[:, :5]
        for nivel in self.tipos_dummy_:
            indicador = (tipo == nivel).astype(float)
            columnas.append(indicador[:, None])
            columnas.append(lugar * indicador[:, None])
        for nivel in self.cuentas_dummy_:
            columnas.append((cuenta == nivel).astype(float)[:, None])
        for nivel in self.bate_dummy_:
            columnas.append((bate == nivel).astype(float)[:, None])
        for nivel in self.pitcher_dummy_:
            columnas.append((pitcher == nivel).astype(float)[:, None])
        return np.hstack(columnas)

    def fit(self, datos: pd.DataFrame, y: np.ndarray, max_iter: int = 30) -> "ModeloLocation":
        valores = datos[FEATURES_LUGAR].to_numpy(dtype=float)
        self.mu_ = valores.mean(axis=0)
        sigma = valores.std(axis=0)
        self.sigma_ = np.where(sigma < 1e-8, 1.0, sigma)

        tipos = sorted(datos["AutoPitchType"].astype(str).unique())
        self.tipo_base_ = TIPO_BASE if TIPO_BASE in tipos else tipos[0]
        self.tipos_dummy_ = [tipo for tipo in tipos if tipo != self.tipo_base_]
        self.cuentas_dummy_ = [cuenta for cuenta in CUENTAS if cuenta != CUENTA_BASE and (datos["count"] == cuenta).any()]
        self.bate_dummy_ = [mano for mano in ("Left",) if (datos["BatterSide"] == mano).any()]
        self.pitcher_dummy_ = [mano for mano in ("Left",) if (datos["PitcherThrows"] == mano).any()]

        nombres = ["intercepto"] + FEATURES_LUGAR
        for tipo in self.tipos_dummy_:
            nombres.append(f"tipo_{tipo}")
            nombres.extend(f"tipo_{tipo}_{feature}" for feature in FEATURES_LUGAR[:5])
        nombres.extend(f"cuenta_{cuenta}" for cuenta in self.cuentas_dummy_)
        nombres.extend(f"bate_{mano}" for mano in self.bate_dummy_)
        nombres.extend(f"pitcher_{mano}" for mano in self.pitcher_dummy_)
        self.nombres_ = nombres

        X = self.matriz(datos)
        beta = np.zeros(X.shape[1])
        penalizacion = np.ones(X.shape[1])
        penalizacion[0] = 0.0
        for _ in range(max_iter):
            probabilidad = _sigmoid(X @ beta)
            peso = np.maximum(probabilidad * (1.0 - probabilidad), 1e-6)
            respuesta = X @ beta + (y - probabilidad) / peso
            hess = X.T @ (X * peso[:, None])
            hess[np.diag_indices(len(beta))] += penalizacion
            beta_nuevo = np.linalg.solve(hess, X.T @ (peso * respuesta))
            if np.max(np.abs(beta_nuevo - beta)) < 1e-6:
                beta = beta_nuevo
                break
            beta = beta_nuevo
        self.beta_ = beta
        return self

    def probabilidades(self, datos: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.matriz(datos) @ self.beta_)

    def a_dict(self) -> dict:
        return {
            "mu": self.mu_,
            "sigma": self.sigma_,
            "beta": self.beta_,
            "nombres": self.nombres_,
            "tipos_dummy": self.tipos_dummy_,
            "cuentas_dummy": self.cuentas_dummy_,
            "bate_dummy": self.bate_dummy_,
            "pitcher_dummy": self.pitcher_dummy_,
            "tipo_base": self.tipo_base_,
            "referencia_xy": {tipo: list(par) for tipo, par in self.referencia_xy_.items()},
            "mu_le": self.mu_le_,
            "sigma_le": self.sigma_le_,
        }


def modelo_desde_dict(estado: dict) -> ModeloLocation:
    modelo = ModeloLocation()
    modelo.mu_ = estado["mu"]
    modelo.sigma_ = estado["sigma"]
    modelo.beta_ = estado["beta"]
    modelo.nombres_ = estado["nombres"]
    modelo.tipos_dummy_ = estado["tipos_dummy"]
    modelo.cuentas_dummy_ = estado["cuentas_dummy"]
    modelo.bate_dummy_ = estado["bate_dummy"]
    modelo.pitcher_dummy_ = estado["pitcher_dummy"]
    modelo.tipo_base_ = estado["tipo_base"]
    modelo.referencia_xy_ = {tipo: (float(par[0]), float(par[1])) for tipo, par in estado["referencia_xy"].items()}
    modelo.mu_le_ = {tipo: float(valor) for tipo, valor in estado["mu_le"].items()}
    modelo.sigma_le_ = {tipo: float(valor) for tipo, valor in estado["sigma_le"].items()}
    return modelo


def _con_referencia(datos: pd.DataFrame, referencia_xy: dict[str, tuple[float, float]]) -> pd.DataFrame:
    ref = datos.copy()
    for tipo, (x_ref, z_ref) in referencia_xy.items():
        mascara = ref["AutoPitchType"].astype(str) == tipo
        ref.loc[mascara, "plate_x"] = x_ref
        ref.loc[mascara, "plate_z"] = z_ref
    ref["plate_x2"] = ref["plate_x"] ** 2
    ref["plate_z2"] = ref["plate_z"] ** 2
    ref["plate_xz"] = ref["plate_x"] * ref["plate_z"]
    return ref


def _escala_efecto(efecto: np.ndarray, tipos: np.ndarray, mu_le: dict, sigma_le: dict) -> np.ndarray:
    mu = np.array([mu_le.get(tipo, 0.0) for tipo in tipos], dtype=float)
    sigma = np.array([sigma_le.get(tipo, 1.0) for tipo in tipos], dtype=float)
    sigma = np.where(np.abs(sigma) < 1e-12, 1.0, sigma)
    return 100.0 + 10.0 * (efecto - mu) / sigma


def entrenar_location(datos: pd.DataFrame, indice_train: np.ndarray, indice_test: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Ajusta Location+ y devuelve el estado junto con el frame puntuado."""
    frame = preparar_ubicacion(datos)
    validos = filas_validas(frame).to_numpy()
    train = np.intersect1d(indice_train, np.flatnonzero(validos), assume_unique=False)
    test = np.intersect1d(indice_test, np.flatnonzero(validos), assume_unique=False)
    print(f"Location+: {len(train):,} lanzamientos de entrenamiento con ubicacion")

    entrenamiento = frame.iloc[train].reset_index(drop=True)
    y_train = frame.iloc[train]["is_whiff"].to_numpy(dtype=float)
    modelo = ModeloLocation().fit(entrenamiento, y_train)

    referencia = {}
    for tipo, bloque in entrenamiento.groupby("AutoPitchType"):
        referencia[str(tipo)] = (float(bloque["plate_x"].mean()), float(bloque["plate_z"].mean()))
    modelo.referencia_xy_ = referencia

    efecto_train = (
        modelo.probabilidades(entrenamiento)
        - modelo.probabilidades(_con_referencia(entrenamiento, referencia))
    )
    tipos_train = entrenamiento["AutoPitchType"].astype(str).to_numpy()
    mu_le = {}
    sigma_le = {}
    for tipo in sorted(set(tipos_train)):
        muestra = efecto_train[tipos_train == tipo]
        mu_le[tipo] = float(muestra.mean())
        sigma = float(muestra.std(ddof=0))
        sigma_le[tipo] = sigma if sigma > 1e-12 else 1.0
    modelo.mu_le_ = mu_le
    modelo.sigma_le_ = sigma_le

    evaluacion = frame.iloc[test].reset_index(drop=True)
    p_test = modelo.probabilidades(evaluacion)
    y_test = frame.iloc[test]["is_whiff"].to_numpy(dtype=float)
    auc = _auc(y_test, p_test)
    perdida = _log_loss(y_test, p_test)
    print(f"Location+ AUC de prueba: {auc:.3f}   log loss: {perdida:.3f}")

    usable = frame.loc[validos].copy()
    p_actual = modelo.probabilidades(usable)
    efecto = p_actual - modelo.probabilidades(_con_referencia(usable, referencia))
    tipos = usable["AutoPitchType"].astype(str).to_numpy()
    frame["p_location"] = np.nan
    frame["location_effect"] = np.nan
    frame["location_plus"] = np.nan
    frame.loc[validos, "p_location"] = p_actual
    frame.loc[validos, "location_effect"] = efecto
    frame.loc[validos, "location_plus"] = _escala_efecto(efecto, tipos, mu_le, sigma_le)

    estado = modelo.a_dict()
    estado["auc"] = auc
    estado["log_loss"] = perdida
    estado["n_train"] = int(len(train))
    estado["n_test"] = int(len(test))
    return estado, frame


def puntuar_location(datos: dict, stuff_plus: float, estado: dict) -> dict:
    """Location+ de un lanzamiento. Si falta el plato, no inventa una ubicación."""
    lado = datos.get("PlateLocSide")
    altura = datos.get("PlateLocHeight")
    if lado is None or altura is None:
        return {}
    try:
        lado_f = float(lado)
        altura_f = float(altura)
    except (TypeError, ValueError):
        return {}
    if not np.isfinite(lado_f) or not np.isfinite(altura_f):
        return {}

    bolas = datos.get("Balls", 0)
    strikes = datos.get("Strikes", 0)
    if datos.get("count"):
        cuenta = str(datos["count"])
    else:
        cuenta = cuenta_de(0 if bolas is None else bolas, 0 if strikes is None else strikes)
    if cuenta not in CUENTAS:
        cuenta = CUENTA_BASE
    bate = datos.get("BatterSide") or MANO_BASE
    if bate not in ("Right", "Left"):
        bate = MANO_BASE

    fila = preparar_ubicacion(
        pd.DataFrame(
            [
                {
                    "PlateLocSide": lado_f,
                    "PlateLocHeight": altura_f,
                    "Balls": bolas,
                    "Strikes": strikes,
                    "count": cuenta,
                    "BatterSide": bate,
                    "PitcherThrows": datos.get("PitcherThrows"),
                    "AutoPitchType": datos.get("AutoPitchType"),
                    "stuff_plus": stuff_plus,
                }
            ]
        )
    )
    modelo = modelo_desde_dict(estado)
    if not bool(filas_validas(fila).iloc[0]):
        return {}
    p_actual = float(modelo.probabilidades(fila)[0])
    p_ref = float(modelo.probabilidades(_con_referencia(fila, modelo.referencia_xy_))[0])
    efecto = p_actual - p_ref
    tipo = str(datos.get("AutoPitchType"))
    location = float(_escala_efecto(np.array([efecto]), np.array([tipo]), modelo.mu_le_, modelo.sigma_le_)[0])
    return {
        "p_location": p_actual,
        "p_location_ref": p_ref,
        "location_effect": efecto,
        "location_plus": location,
        "PlateLocSide": lado_f,
        "PlateLocHeight": altura_f,
        "Balls": None if bolas is None else int(bolas),
        "Strikes": None if strikes is None else int(strikes),
        "BatterSide": bate,
        "count": cuenta,
    }


def grilla_location(
    estado: dict,
    tipo: str,
    stuff_plus: float,
    pitcher: str,
    bate: str = "Right",
    cuenta: str = "0-0",
    n: int = 35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mapa de Location+ sobre el plato, con el Stuff+ y el contexto fijos."""
    xs = np.linspace(-1.6, 1.6, n)
    zs = np.linspace(1.0, 4.0, n)
    lado, altura = np.meshgrid(xs, zs)
    filas = preparar_ubicacion(
        pd.DataFrame(
            {
                "PlateLocSide": lado.ravel(),
                "PlateLocHeight": altura.ravel(),
                "count": cuenta if cuenta in CUENTAS else CUENTA_BASE,
                "BatterSide": bate if bate in ("Right", "Left") else MANO_BASE,
                "PitcherThrows": pitcher,
                "AutoPitchType": tipo,
                "stuff_plus": stuff_plus,
            }
        )
    )
    modelo = modelo_desde_dict(estado)
    p_actual = modelo.probabilidades(filas)
    p_ref = modelo.probabilidades(_con_referencia(filas, modelo.referencia_xy_))
    efecto = p_actual - p_ref
    tipos = np.full(len(filas), str(tipo))
    valores = _escala_efecto(efecto, tipos, modelo.mu_le_, modelo.sigma_le_)
    return xs, zs, valores.reshape(lado.shape)
