"""Pitching+ de la capa 3 del notebook.

El valor realizado de un lanzamiento, desde el lado del pitcher, es

    RV = V(S_antes) − carreras de la jugada − V(S_después)

S = (outs, corredores, cuenta). V(S) es la expectativa de carreras que
faltan en la entrada, suavizada como en el notebook:

    V(S) = (n_S * promedio_S + k * μ) / (n_S + k)

Con eso se entrena

    RV_esperado = Ridge(Stuff+, Location+, bolas, strikes, outs,
                        diferencia de velo, IVB y HB contra la recta, ρ, tipo)

y la escala queda

    Pitching+ = 100 + 10 (RV_esperado − μ_ref) / σ_ref

100 es el lanzamiento promedio del entrenamiento. Más alto favorece al pitcher.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reclutamiento.fisica import densidad_por_altitud
from reclutamiento.location_plus import (
    _con_referencia,
    _escala_efecto,
    filas_validas,
    modelo_desde_dict,
    preparar_ubicacion,
)
from reclutamiento.stuff_plus import (
    COLUMNAS_BASE,
    FEATURES_NUM,
    RUTA_DATOS,
    TIPOS_MODELO,
    _filtrar_entrenamiento,
    enriquecer,
    estimador_desde_artefacto,
    stuff_plus_por_tipo,
)

VERSION = "notebook-pitching-v1"
ALPHA = 10.0
K_SUAVIZADO = 50
RUTA_MODELO = Path(__file__).resolve().parents[1] / "modelos" / "pitching_plus.joblib"

NUMERICAS = [
    "stuff_plus",
    "location_plus",
    "balls_before",
    "strikes_before",
    "outs_before",
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "rho",
]
TERMINALES = {
    "Strikeout",
    "Walk",
    "Single",
    "Double",
    "Triple",
    "HomeRun",
    "Out",
    "HitByPitch",
    "Sacrifice",
    "Error",
    "FieldersChoice",
}
BOLAS = {"BallCalled", "BallinDirt", "BallIntentional"}
STRIKES = {"StrikeCalled", "StrikeSwinging"}


def _empaquetar(primera: int, segunda: int, tercera: int) -> int:
    return int(primera) + 2 * int(segunda) + 4 * int(tercera)


def _desempaquetar(estado: int) -> tuple[int, int, int]:
    return estado & 1, (estado >> 1) & 1, (estado >> 2) & 1


def _avanzar(bases: int, evento: str, outs_en_jugada: int, carreras: int) -> tuple[int, int]:
    """Devuelve (bases posteriores, outs que suma la jugada)."""
    primera, segunda, tercera = _desempaquetar(bases)
    carreras = int(carreras)
    outs_en_jugada = int(outs_en_jugada)

    if evento in {"Walk", "HitByPitch"}:
        anota = int(primera and segunda and tercera)
        nueva_primera = 1
        nueva_segunda = 1 if primera else segunda
        nueva_tercera = 1 if primera and segunda else tercera
        return _empaquetar(nueva_primera, nueva_segunda, nueva_tercera), 0 if anota or True else 0

    if evento == "HomeRun":
        return 0, 0

    if evento == "Triple":
        return _empaquetar(0, 0, 1), 0

    if evento == "Double":
        if carreras > segunda + tercera:
            return _empaquetar(0, 1, 0), 0
        return _empaquetar(0, 1, primera), 0

    if evento == "Single":
        opciones = [
            (_empaquetar(1, primera, segunda), tercera),
            (_empaquetar(1, 0, primera), segunda + tercera),
            (_empaquetar(1, 0, 0), primera + segunda + tercera),
        ]
        return min(opciones, key=lambda opcion: abs(opcion[1] - carreras))[0], 0

    if evento == "Strikeout":
        return bases, 1

    if evento == "Sacrifice":
        return _empaquetar(0, primera, segunda), 1

    if evento == "FieldersChoice":
        if tercera:
            return _empaquetar(1, primera, segunda), 1
        if segunda:
            return _empaquetar(1, primera, 0), 1
        return _empaquetar(1, 0, 0), 1

    if evento == "Error":
        return _empaquetar(1, primera, segunda), 0

    outs = outs_en_jugada if outs_en_jugada > 0 else 1
    if outs >= 3:
        return 0, 3
    if outs == 2:
        if primera:
            return _empaquetar(0, segunda, tercera), 2
        if segunda:
            return _empaquetar(primera, 0, tercera), 2
        if tercera:
            return _empaquetar(primera, segunda, 0), 2
        return bases, 2
    if carreras > 0 and tercera:
        return _empaquetar(primera, segunda, 0), 1
    return bases, outs


def _marca_tiempo(identificador: str) -> int:
    try:
        return uuid.UUID(str(identificador)).time
    except (ValueError, AttributeError, TypeError):
        return 0


def reconstruir_estados(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena cada media entrada y arma el estado anterior y posterior."""
    datos = df.copy()
    datos["_orden"] = datos["PitchUID"].map(_marca_tiempo)
    datos = datos.sort_values(
        ["game_anon_id", "Inning", "Top/Bottom", "_orden"],
        kind="mergesort",
    )
    n = len(datos)
    base_antes = np.zeros(n, dtype=np.int16)
    base_despues = np.zeros(n, dtype=np.int16)
    outs_despues = np.zeros(n, dtype=np.int16)
    bolas_despues = np.zeros(n, dtype=np.int16)
    strikes_despues = np.zeros(n, dtype=np.int16)
    entrada_termina = np.zeros(n, dtype=bool)
    carreras_restantes = np.zeros(n, dtype=float)

    juegos = datos["game_anon_id"].to_numpy()
    innings = datos["Inning"].to_numpy()
    mitades = datos["Top/Bottom"].astype(str).to_numpy()
    outs = datos["Outs"].to_numpy(dtype=float)
    bolas = datos["Balls"].to_numpy(dtype=float)
    strikes = datos["Strikes"].to_numpy(dtype=float)
    eventos = datos["play_result"].astype(str).to_numpy()
    llamados = datos["PitchCall"].astype(str).to_numpy()
    outs_jugada = datos["OutsOnPlay"].fillna(0).to_numpy(dtype=float)
    carreras = datos["RunsScored"].fillna(0).to_numpy(dtype=float)

    inicio = 0
    while inicio < n:
        fin = inicio + 1
        while (
            fin < n
            and juegos[fin] == juegos[inicio]
            and innings[fin] == innings[inicio]
            and mitades[fin] == mitades[inicio]
        ):
            fin += 1
        bases = 0
        for i in range(inicio, fin):
            if not np.isfinite(outs[i]):
                outs[i] = 0
            base_antes[i] = bases
            evento = eventos[i]
            terminal = evento in TERMINALES or outs_jugada[i] > 0
            if terminal:
                bases_nuevas, outs_suma = _avanzar(bases, evento, outs_jugada[i], carreras[i])
                outs_nuevo = min(3, int(outs[i]) + int(outs_suma))
                bolas_nuevas, strikes_nuevos = 0, 0
            else:
                bases_nuevas = bases
                outs_nuevo = int(outs[i])
                bolas_nuevas = int(bolas[i]) if np.isfinite(bolas[i]) else 0
                strikes_nuevos = int(strikes[i]) if np.isfinite(strikes[i]) else 0
                if llamados[i] in BOLAS or evento in BOLAS:
                    bolas_nuevas = min(3, bolas_nuevas + 1)
                elif llamados[i] in STRIKES or evento == "StrikeSwinging":
                    strikes_nuevos = min(2, strikes_nuevos + 1)
                elif "Foul" in llamados[i] or evento == "FoulBallFieldable":
                    if strikes_nuevos < 2:
                        strikes_nuevos += 1
            base_despues[i] = 0 if outs_nuevo >= 3 else bases_nuevas
            outs_despues[i] = outs_nuevo
            bolas_despues[i] = bolas_nuevas
            strikes_despues[i] = strikes_nuevos
            entrada_termina[i] = outs_nuevo >= 3
            bases = 0 if outs_nuevo >= 3 else bases_nuevas
            if i + 1 < fin and np.isfinite(outs[i + 1]) and int(outs[i + 1]) != outs_nuevo and outs_nuevo < 3:
                bases = 0
        restantes = np.cumsum(carreras[inicio:fin][::-1])[::-1]
        carreras_restantes[inicio:fin] = restantes
        inicio = fin

    datos["base_state_before"] = base_antes
    datos["base_state_after"] = base_despues
    datos["outs_after"] = outs_despues
    datos["balls_after"] = bolas_despues
    datos["strikes_after"] = strikes_despues
    datos["inning_ended"] = entrada_termina
    datos["runs_remaining"] = carreras_restantes
    return datos.reset_index(drop=True)


def auditar_transiciones(datos: pd.DataFrame) -> float:
    """Fracción de lanzamientos cuyo out posterior coincide con el siguiente."""
    mismo = (
        (datos["game_anon_id"] == datos["game_anon_id"].shift(-1))
        & (datos["Inning"] == datos["Inning"].shift(-1))
        & (datos["Top/Bottom"] == datos["Top/Bottom"].shift(-1))
        & ~datos["inning_ended"]
    )
    if not mismo.any():
        return float("nan")
    coincide = datos.loc[mismo, "outs_after"].to_numpy() == datos["Outs"].shift(-1)[mismo].to_numpy()
    return float(np.mean(coincide))


def _tabla_valor(datos: pd.DataFrame, k: float = K_SUAVIZADO) -> tuple[dict, float]:
    previo = float(datos["runs_remaining"].mean())
    grupo = (
        datos.groupby(["Outs", "base_state_before", "Balls", "Strikes"], observed=True)["runs_remaining"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grupo["RE"] = (grupo["count"] * grupo["mean"] + k * previo) / (grupo["count"] + k)
    tabla = {
        (int(fila.Outs), int(fila.base_state_before), int(fila.Balls), int(fila.Strikes)): float(fila.RE)
        for fila in grupo.itertuples(index=False)
    }
    return tabla, previo


def _consultar(tabla: dict, previo: float, outs, bases, bolas, strikes) -> float:
    try:
        llave = (int(outs), int(bases), int(bolas), int(strikes))
    except (TypeError, ValueError):
        return previo
    return tabla.get(llave, previo)


def asignar_run_value(datos: pd.DataFrame, tabla: dict, previo: float) -> pd.DataFrame:
    """RV_pitcher = V(antes) − carreras − V(después). Si cierra la entrada, V(después) = 0."""
    out = datos.copy()
    antes = [
        _consultar(tabla, previo, fila.Outs, fila.base_state_before, fila.Balls, fila.Strikes)
        for fila in out.itertuples(index=False)
    ]
    despues = []
    for fila in out.itertuples(index=False):
        if fila.inning_ended:
            despues.append(0.0)
        else:
            despues.append(
                _consultar(tabla, previo, fila.outs_after, fila.base_state_after, fila.balls_after, fila.strikes_after)
            )
    out["RE_before"] = antes
    out["RE_after"] = despues
    out["rv_pitcher"] = out["RE_before"] - out["RunsScored"].fillna(0) - out["RE_after"]
    return out


def _rectas(datos: pd.DataFrame) -> pd.DataFrame:
    """Promedio de la recta de cada pitcher. Si no hay four-seam, usa el sinker."""
    filas = []
    for pitcher, bloque in datos.groupby("pitcher_anon_id", observed=True):
        recta = bloque[bloque["AutoPitchType"] == "Four-Seam"]
        if recta.empty:
            recta = bloque[bloque["AutoPitchType"] == "Sinker"]
        if recta.empty:
            continue
        filas.append(
            {
                "pitcher_anon_id": pitcher,
                "fb_velo": float(recta["RelSpeed"].mean()),
                "fb_ivb": float(recta["InducedVertBreak"].mean()),
                "fb_hb": float(recta["HorzBreak"].mean()),
            }
        )
    return pd.DataFrame(filas)


def _puntuar_stuff_location(datos: pd.DataFrame, artefacto_stuff: dict) -> pd.DataFrame:
    frame = enriquecer(datos)
    completos = np.isfinite(frame[FEATURES_NUM]).all(axis=1)
    frame = frame.loc[completos].copy()
    modelo = estimador_desde_artefacto(artefacto_stuff)
    probabilidad = modelo.probabilidades(frame)
    frame["stuff_plus"] = stuff_plus_por_tipo(
        probabilidad,
        frame["AutoPitchType"].astype(str).to_numpy(),
        artefacto_stuff.get("mu_p_tipo", {}),
        artefacto_stuff.get("sigma_p_tipo", {}),
        artefacto_stuff["mu_p"],
        artefacto_stuff["sigma_p"],
    )
    preparado = preparar_ubicacion(frame)
    preparado["stuff_plus"] = frame["stuff_plus"].to_numpy()
    location = np.full(len(preparado), np.nan)
    validos = filas_validas(preparado).to_numpy()
    if validos.any() and "location" in artefacto_stuff:
        local = modelo_desde_dict(artefacto_stuff["location"])
        usable = preparado.loc[validos]
        efecto = local.probabilidades(usable) - local.probabilidades(_con_referencia(usable, local.referencia_xy_))
        location[validos] = _escala_efecto(
            efecto,
            usable["AutoPitchType"].astype(str).to_numpy(),
            local.mu_le_,
            local.sigma_le_,
        )
    frame["location_plus"] = location
    frame["rho"] = frame["rho"].to_numpy(dtype=float)
    return frame


def _matriz(datos: pd.DataFrame, estado: dict) -> np.ndarray:
    numericas = []
    for nombre in NUMERICAS:
        columna = pd.to_numeric(datos[nombre], errors="coerce").to_numpy(dtype=float)
        mediana = estado["mediana"][nombre]
        columna = np.where(np.isfinite(columna), columna, mediana)
        numericas.append(columna)
    X = np.column_stack(numericas)
    X = (X - estado["mu"]) / estado["sigma"]
    tipo = datos["pitch_type"].astype(str).to_numpy()
    dummies = [(tipo == nivel).astype(float) for nivel in estado["tipos"]]
    if dummies:
        X = np.column_stack([X, *dummies])
    return np.column_stack([np.ones(len(X)), X])


def _ajustar_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = X.T @ X
    penalizacion = np.eye(X.shape[1]) * alpha
    penalizacion[0, 0] = 0.0
    return np.linalg.solve(xtx + penalizacion, X.T @ y)


def entrenar(artefacto_stuff: dict | None = None, forzar: bool = False) -> dict:
    """Ajusta Pitching+ con 2024-2025 y deja 2026 como temporada de prueba."""
    if not forzar and RUTA_MODELO.exists():
        guardado = joblib.load(RUTA_MODELO)
        if guardado.get("version") == VERSION:
            return guardado
    if artefacto_stuff is None:
        from reclutamiento.stuff_plus import cargar_modelo

        artefacto_stuff = cargar_modelo()
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No está la base de la organización en {RUTA_DATOS}")

    print("Cargando lanzamientos para Pitching+...")
    columnas = list(dict.fromkeys(
        COLUMNAS_BASE
        + ["game_anon_id", "year", "PitchCall", "play_result", "RunsScored", "OutsOnPlay", "Outs", "Inning", "Top/Bottom", "PitchUID"]
    ))
    bruto = pd.read_parquet(RUTA_DATOS, columns=columnas)
    bruto = bruto[bruto["AutoPitchType"].isin(TIPOS_MODELO)]
    bruto = bruto[bruto["PitchCall"].astype(str) != "Undefined"]
    bruto = _filtrar_entrenamiento(bruto)
    bruto = bruto[bruto["Outs"].isin([0, 1, 2])]
    bruto = bruto[bruto["Balls"].between(0, 3)]
    bruto = bruto[bruto["Strikes"].between(0, 2)]
    print(f"Lanzamientos para el estado: {len(bruto):,}")
    estados = reconstruir_estados(bruto)
    acuerdo = auditar_transiciones(estados)
    print(f"Outs posteriores que coinciden con el siguiente lanzamiento: {acuerdo:.1%}")

    puntuados = _puntuar_stuff_location(estados, artefacto_stuff)
    # El filtro de Stuff+ puede quitar filas. El run value se calcula sobre las que quedan.
    estados_idx = estados.set_index(estados.index)
    comun = puntuados.index.intersection(estados.index)
    # _puntuar copia y filtra, así que el índice puede no coincidir.
    # Volvemos a unir por las columnas de identidad del lanzamiento.
    llave = ["PitchUID"]
    estados = estados.drop_duplicates(llave)
    puntuados = puntuados.drop_duplicates(llave)
    junto = puntuados.merge(
        estados[llave + ["Outs", "Balls", "Strikes", "base_state_before", "base_state_after", "outs_after", "balls_after", "strikes_after", "inning_ended", "runs_remaining", "RunsScored", "year", "play_result"]],
        on="PitchUID",
        how="inner",
        suffixes=("", "_estado"),
    )
    print(f"Lanzamientos con Stuff+ y estado: {len(junto):,}")
    tabla, previo = _tabla_valor(junto[junto["year"] < 2026])
    junto = asignar_run_value(junto, tabla, previo)
    rectas = _rectas(junto)
    junto = junto.merge(rectas, on="pitcher_anon_id", how="left")
    junto["velo_diff_vs_fb"] = junto["RelSpeed"] - junto["fb_velo"]
    junto["ivb_diff_vs_fb"] = junto["InducedVertBreak"] - junto["fb_ivb"]
    junto["hb_diff_vs_fb"] = junto["HorzBreak"] - junto["fb_hb"]
    junto["balls_before"] = junto["Balls"]
    junto["strikes_before"] = junto["Strikes"]
    junto["outs_before"] = junto["Outs"]
    junto["pitch_type"] = junto["AutoPitchType"].astype(str)
    junto = junto.replace([np.inf, -np.inf], np.nan).dropna(subset=["rv_pitcher", "stuff_plus", "rho"])

    train = junto[junto["year"] < 2026].copy()
    test = junto[junto["year"] == 2026].copy()
    mediana = {nombre: float(train[nombre].median()) for nombre in NUMERICAS}
    for nombre, valor in mediana.items():
        train[nombre] = train[nombre].fillna(valor)
        test[nombre] = test[nombre].fillna(valor)
    mu = train[NUMERICAS].mean().to_numpy(dtype=float)
    sigma = train[NUMERICAS].std(ddof=0).to_numpy(dtype=float)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    tipos = [tipo for tipo in TIPOS_MODELO if (train["pitch_type"] == tipo).any()]
    estado = {"mediana": mediana, "mu": mu, "sigma": sigma, "tipos": tipos}
    X_train = _matriz(train, estado)
    y_train = train["rv_pitcher"].to_numpy(dtype=float)
    beta = _ajustar_ridge(X_train, y_train, ALPHA)
    pred_train = X_train @ beta
    pred_test = _matriz(test, estado) @ beta
    y_test = test["rv_pitcher"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y_test - pred_test)))
    rmse = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))
    mu_rv = float(pred_train.mean())
    sigma_rv = float(pred_train.std(ddof=0))
    print(f"Prueba 2026  MAE {mae:.4f}  RMSE {rmse:.4f}  mu_rv {mu_rv:.5f}  sigma_rv {sigma_rv:.5f}")
    artefacto = {
        "version": VERSION,
        "alpha": ALPHA,
        "k": K_SUAVIZADO,
        "beta": beta,
        "mu": mu,
        "sigma": sigma,
        "mediana": mediana,
        "tipos": tipos,
        "numericas": NUMERICAS,
        "mu_rv": mu_rv,
        "sigma_rv": sigma_rv if sigma_rv > 1e-12 else 1.0,
        "mae": mae,
        "rmse": rmse,
        "acuerdo_outs": acuerdo,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "re_prior": previo,
    }
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, RUTA_MODELO)
    print(f"Guardado en {RUTA_MODELO.name}")
    return artefacto


def cargar_pitching(entrenar_si_falta: bool = True) -> dict:
    if RUTA_MODELO.exists():
        guardado = joblib.load(RUTA_MODELO)
        if guardado.get("version") == VERSION:
            return guardado
    if not entrenar_si_falta:
        raise FileNotFoundError(f"No está el Pitching+ en {RUTA_MODELO}")
    return entrenar(forzar=True)


def predecir_rv(datos: pd.DataFrame, artefacto: dict) -> np.ndarray:
    return _matriz(datos, artefacto) @ artefacto["beta"]


def escala_pitching(rv_esperado, artefacto: dict):
    return 100.0 + 10.0 * (np.asarray(rv_esperado, dtype=float) - artefacto["mu_rv"]) / artefacto["sigma_rv"]


def referencia_recta(repertorio: list[dict]) -> dict | None:
    """La recta del repertorio contra la que se miden las diferencias del arsenal."""
    def _peso(fila: dict) -> float:
        if fila.get("uso_pct"):
            return float(fila["uso_pct"])
        if fila.get("n_lanzamientos"):
            return float(fila["n_lanzamientos"])
        return 1.0

    for tipo in ("Four-Seam", "Sinker"):
        grupo = [fila for fila in repertorio if fila.get("AutoPitchType") == tipo and np.isfinite(fila.get("RelSpeed", np.nan))]
        if not grupo:
            continue
        pesos = np.array([_peso(fila) for fila in grupo], dtype=float)
        def _media(columna: str) -> float:
            valores = np.array([float(fila[columna]) for fila in grupo], dtype=float)
            validos = np.isfinite(valores) & np.isfinite(pesos) & (pesos > 0)
            if not validos.any():
                return float("nan")
            return float(np.average(valores[validos], weights=pesos[validos]))

        return {
            "RelSpeed": _media("RelSpeed"),
            "InducedVertBreak": _media("InducedVertBreak"),
            "HorzBreak": _media("HorzBreak"),
        }
    return None


def puntuar_pitching(
    datos: dict,
    artefacto: dict,
    recta: dict | None = None,
    outs: float | None = None,
) -> dict:
    """Pitching+ de un lanzamiento ya calificado en Stuff+ y, si existe, Location+."""
    if recta is None:
        velo_diff = ivb_diff = hb_diff = 0.0
    else:
        velo_diff = float(datos.get("RelSpeed") or np.nan) - float(recta["RelSpeed"])
        ivb_diff = float(datos.get("InducedVertBreak") or np.nan) - float(recta["InducedVertBreak"])
        hb_diff = float(datos.get("HorzBreak") or np.nan) - float(recta["HorzBreak"])
    outs_valor = datos.get("Outs") if outs is None else outs
    if outs_valor is None or not np.isfinite(float(outs_valor)):
        outs_valor = 0
    fila = pd.DataFrame(
        [
            {
                "stuff_plus": datos.get("stuff_plus"),
                "location_plus": datos.get("location_plus"),
                "balls_before": datos.get("Balls", 0),
                "strikes_before": datos.get("Strikes", 0),
                "outs_before": outs_valor,
                "velo_diff_vs_fb": velo_diff,
                "ivb_diff_vs_fb": ivb_diff,
                "hb_diff_vs_fb": hb_diff,
                "rho": datos.get("rho"),
                "pitch_type": datos.get("AutoPitchType"),
            }
        ]
    )
    if not np.isfinite(fila[["stuff_plus", "rho"]].to_numpy(dtype=float)).all():
        return {}
    esperado = float(predecir_rv(fila, artefacto)[0])
    return {
        "rv_esperado": esperado,
        "pitching_plus": float(escala_pitching(esperado, artefacto)),
    }


def completar_capas(datos: pd.DataFrame, artefacto_stuff: dict, artefacto_pitching: dict) -> pd.DataFrame:
    """Location+ y Pitching+ para lanzamientos que ya traen Stuff+ y el aire del escenario."""
    frame = datos.copy()
    preparado = preparar_ubicacion(frame)
    preparado["stuff_plus"] = frame["stuff_plus"].to_numpy()
    location = np.full(len(frame), np.nan)
    if "location" in artefacto_stuff:
        validos = filas_validas(preparado).to_numpy()
        if validos.any():
            local = modelo_desde_dict(artefacto_stuff["location"])
            usable = preparado.loc[validos]
            efecto = local.probabilidades(usable) - local.probabilidades(
                _con_referencia(usable, local.referencia_xy_)
            )
            location[validos] = _escala_efecto(
                efecto,
                usable["AutoPitchType"].astype(str).to_numpy(),
                local.mu_le_,
                local.sigma_le_,
            )
    frame["location_plus"] = location
    return anotar_organizacion(frame, artefacto_pitching)


def anotar_organizacion(datos: pd.DataFrame, artefacto: dict) -> pd.DataFrame:
    """Añade Pitching+ a lanzamientos del equipo que ya traen Stuff+ y el aire del escenario."""
    if datos.empty:
        return datos
    frame = datos.copy()
    rectas = _rectas(frame)
    if rectas.empty:
        frame["velo_diff_vs_fb"] = 0.0
        frame["ivb_diff_vs_fb"] = 0.0
        frame["hb_diff_vs_fb"] = 0.0
    else:
        frame = frame.merge(rectas, on="pitcher_anon_id", how="left")
        frame["velo_diff_vs_fb"] = frame["RelSpeed"] - frame["fb_velo"]
        frame["ivb_diff_vs_fb"] = frame["InducedVertBreak"] - frame["fb_ivb"]
        frame["hb_diff_vs_fb"] = frame["HorzBreak"] - frame["fb_hb"]
    frame["balls_before"] = frame["Balls"] if "Balls" in frame.columns else 0
    frame["strikes_before"] = frame["Strikes"] if "Strikes" in frame.columns else 0
    frame["outs_before"] = frame["Outs"] if "Outs" in frame.columns else 0
    frame["pitch_type"] = frame["AutoPitchType"].astype(str)
    if "location_plus" not in frame.columns:
        frame["location_plus"] = np.nan
    esperado = predecir_rv(frame, artefacto)
    frame["rv_esperado"] = esperado
    frame["pitching_plus"] = escala_pitching(esperado, artefacto)
    return frame


if __name__ == "__main__":
    entrenar(forzar=True)
