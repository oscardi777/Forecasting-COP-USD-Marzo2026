"""
mlr_howa_rolling.py
===================
Benchmark / modelo de pronóstico MLR-HOWA con rolling window calendario.

Implementa el modelo propuesto en:
  Flores-Sosa et al. (2022). "Forecasting the exchange rate with multiple
  linear regression and heavy ordered weighted average operators."
  Knowledge-Based Systems, 248, 108863.

CAMBIO RESPECTO A LA VERSIÓN ANTERIOR:
  El loop de predicción ahora es completamente out-of-sample (recursivo real):
  para predecir el paso t+k, usa únicamente las predicciones anteriores ŷ
  como lags — nunca valores reales del período de predicción.
  Esto hace la comparación con DFGCN metodológicamente justa: ambos modelos
  predicen el horizonte completo sin ver datos futuros.

Uso:
    python mlr.py --data_path .\\datos\\data_wordProf_2019-2025.csv --results_dir ./resultados_mlrowa_Profe_2019-2025/ --save_predictions

    python mlr.py --data_path .\\datos\\tasas_BRL_COP_CHI_1993-2025.csv --results_dir ./resultados_mlrowa_OnlyCOP_1993-2025/ --save_predictions

    python mlr.py --data_path datos/data_rates_DifInterbank.csv --results_dir resultados_mlrhowa_DIFRates --save_predictions

    python mlr.py --data_path .\\datos\\data_comodities_interbank_2019-2025.csv --results_dir ./resultados_mlrowa_ComoditiesIB_2019-2025/ --save_predictions

    python mlr.py --data_path .//datos//tasas_BRL_COP_CHI_1993-2025.csv --results_dir resultados_mlrowa_ORates_1lags_CH --save_predictions

Opcional:
    --root_path   Carpeta raiz del CSV (default: .)
    --data_path   Nombre o ruta completa del CSV
    --results_dir Carpeta donde guardar CSVs de resultados
                  (default: ./resultados_mlrhowa/)
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
import random

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

fix_seed = 42
random.seed(fix_seed)
np.random.seed(fix_seed)

from utils.metrics import metric


# ===========================================================================
# GRANULARIDADES (idéntico a dfgcn.py)
# ===========================================================================
GRANULARIDADES = {
    "1": ("1 semana",  relativedelta(weeks=1)),
    "2": ("2 semanas", relativedelta(weeks=2)),
    "3": ("1 mes",     relativedelta(months=1)),
    "4": ("2 meses",   relativedelta(months=2)),
    "5": ("3 meses",   relativedelta(months=3)),
    "6": ("6 meses",   relativedelta(months=6)),
    "7": ("1 ano",     relativedelta(years=1)),
    "8": ("2 anos",    relativedelta(years=2)),
}

MIN_PERIODOS_RECOMENDADOS = {
    "1": 52, "2": 26, "3": 12, "4": 6,
    "5": 4,  "6": 2,  "7": 2,  "8": 2,
}
MIN_PERIODOS_ABSOLUTO = 2


# ===========================================================================
# OPERADORES OWA / HOWA
# ===========================================================================

def _gen_weights_owa(n: int, orness: float) -> np.ndarray:
    if n == 1:
        return np.array([1.0])
    orness = float(np.clip(orness, 1e-6, 1 - 1e-6))
    if abs(orness - 0.5) < 1e-6:
        return np.ones(n) / n
    j_arr = np.arange(1, n + 1, dtype=float)

    def _orness_from_k(k):
        raw = np.exp(-k * (j_arr - 1))
        if not np.all(np.isfinite(raw)) or raw.sum() == 0:
            return 0.5
        w = raw / raw.sum()
        return float(np.sum(w * (n - j_arr) / (n - 1)))

    if orness > 0.5:
        lo, hi = 0.0, 200.0 / max(n, 2)
        for _ in range(30):
            if _orness_from_k(hi) >= orness:
                break
            hi *= 2
    else:
        lo, hi = -200.0 / max(n, 2), 0.0
        for _ in range(30):
            if _orness_from_k(lo) <= orness:
                break
            lo *= 2

    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _orness_from_k(mid) > orness:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-10:
            break

    k_opt = (lo + hi) / 2.0
    raw = np.exp(-k_opt * (j_arr - 1))
    w = raw / raw.sum()
    return np.clip(w, 0, 1)


def _gen_weights_howa(n: int, beta: float) -> np.ndarray:
    if n == 1:
        return np.array([1.0])
    beta = float(np.clip(beta, 0.0, 1.0))
    target_sum = 1.0 + beta * (n - 1)
    j_arr = np.arange(1, n + 1, dtype=float)
    raw = np.exp(-0.5 * (j_arr - 1))
    w = raw / raw.sum() * target_sum
    return w


def _sort_descending(values: np.ndarray) -> np.ndarray:
    return np.sort(values)[::-1]


def _sort_by_inducer(values: np.ndarray, inducer: np.ndarray) -> np.ndarray:
    order = np.argsort(inducer)[::-1]
    return values[order]


def _apply_lambda(values: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam - 1.0) < 1e-9:
        return values
    sign = np.sign(values)
    safe = np.abs(values) ** lam * sign
    return safe


# ===========================================================================
# CLASE PRINCIPAL: MLR-HOWA
# ===========================================================================

class MLR_HOWA:
    OPERATORS = ("owa", "howa", "iowa", "igowa", "ihowa", "ighowa")

    def __init__(self, operator: str = "howa", orness: float = 0.7,
                 beta: float = 0.5, lam: float = 2.0,
                 weights: np.ndarray = None):
        assert operator.lower() in self.OPERATORS, \
            f"Operador '{operator}' no reconocido. Opciones: {self.OPERATORS}"
        self.operator = operator.lower()
        self.orness   = orness
        self.beta     = beta
        self.lam      = lam
        self.custom_w = weights
        self.coef_    = None

    def _build_weights(self, n: int) -> np.ndarray:
        if self.custom_w is not None:
            w = np.asarray(self.custom_w, dtype=float)
            assert len(w) == n
            return w
        if self.operator in ("howa", "ihowa", "ighowa"):
            return _gen_weights_howa(n, self.beta)
        else:
            return _gen_weights_owa(n, self.orness)

    def _owa_mean(self, values: np.ndarray, w: np.ndarray,
                  inducer: np.ndarray = None) -> float:
        vals = values.astype(float).copy()
        if inducer is not None:
            b = _sort_by_inducer(vals, inducer)
        else:
            b = _sort_descending(vals)
        if self.operator in ("igowa", "ighowa"):
            b_transformed = _apply_lambda(b, self.lam)
            result = np.sum(w * b_transformed)
            sign = np.sign(result)
            return float(sign * (abs(result) ** (1.0 / self.lam)))
        else:
            return float(np.sum(w * b))

    def _owa_var(self, values: np.ndarray, w: np.ndarray,
                 mean: float, inducer: np.ndarray = None) -> float:
        vals = values.astype(float).copy()
        if inducer is not None:
            b = _sort_by_inducer(vals, inducer)
        else:
            b = _sort_descending(vals)
        return float(np.sum(w * (b - mean) ** 2))

    def _owa_cov(self, x: np.ndarray, y: np.ndarray, w: np.ndarray,
                 mean_x: float, mean_y: float,
                 inducer: np.ndarray = None) -> float:
        x = x.astype(float).copy()
        y = y.astype(float).copy()
        if inducer is not None:
            order = np.argsort(inducer)[::-1]
        else:
            order = np.argsort(y)[::-1]
        bx = x[order]
        by = y[order]
        return float(np.sum(w * (bx - mean_x) * (by - mean_y)))

    @staticmethod
    def _default_inducer(n: int) -> np.ndarray:
        return np.arange(1, n + 1, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        w    = self._build_weights(n)
        use_inducer = self.operator in ("iowa", "igowa", "ihowa", "ighowa")
        inducer = self._default_inducer(n) if use_inducer else None

        mu_y = self._owa_mean(y, w, inducer)
        mu_X = np.array([self._owa_mean(X[:, k], w, inducer) for k in range(p)])

        A = np.zeros((p, p))
        c = np.zeros(p)
        for k in range(p):
            c[k] = self._owa_cov(X[:, k], y, w, mu_X[k], mu_y, inducer)
            for l in range(p):
                A[k, l] = self._owa_cov(X[:, k], X[:, l], w,
                                        mu_X[k], mu_X[l], inducer)
        try:
            betas = np.linalg.solve(A, c)
        except np.linalg.LinAlgError:
            betas = np.linalg.lstsq(A, c, rcond=None)[0]

        alpha = mu_y - np.dot(betas, mu_X)
        self.coef_ = np.concatenate([[alpha], betas])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "Modelo no ajustado. Llama fit() primero."
        alpha = self.coef_[0]
        betas = self.coef_[1:]
        return alpha + X @ betas


# ===========================================================================
# FUNCIONES DE CALENDARIO
# ===========================================================================

def inicio_periodo_calendario(fecha, gran_delta):
    ts = pd.Timestamp(fecha)
    if gran_delta.weeks:
        return ts - pd.Timedelta(days=ts.weekday())
    if gran_delta.years >= 1:
        return ts.replace(month=1, day=1)
    if gran_delta.months == 6:
        mes_ini = 1 if ts.month <= 6 else 7
        return ts.replace(month=mes_ini, day=1)
    if gran_delta.months == 3:
        mes_ini = ((ts.month - 1) // 3) * 3 + 1
        return ts.replace(month=mes_ini, day=1)
    if gran_delta.months == 2:
        mes_ini = ((ts.month - 1) // 2) * 2 + 1
        return ts.replace(month=mes_ini, day=1)
    return ts.replace(day=1)


def etiqueta_periodo(fecha_ini, gran_delta):
    ts = pd.Timestamp(fecha_ini)
    if gran_delta.weeks:
        return ts.strftime("%Y-W%V")
    if gran_delta.years >= 1:
        return str(ts.year)
    if gran_delta.months == 6:
        semestre = 1 if ts.month <= 6 else 2
        return f"{ts.year}-S{semestre}"
    if gran_delta.months == 3:
        trimestre = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{trimestre}"
    if gran_delta.months >= 2:
        return ts.strftime("%Y-%m")
    return ts.strftime("%Y-%m")


def generar_ventanas_calendario(dates_all, n_entren, gran_delta, modo):
    fecha_min = pd.Timestamp(dates_all.min())
    fecha_max = pd.Timestamp(dates_all.max())
    inicio_base = inicio_periodo_calendario(fecha_min, gran_delta)

    ventanas = []
    ptr = inicio_base

    while True:
        train_ini_rolling = ptr
        train_fin = ptr
        for _ in range(n_entren):
            train_fin = train_fin + gran_delta
        train_fin = train_fin - pd.Timedelta(days=1)

        pred_ini = inicio_periodo_calendario(train_fin + pd.Timedelta(days=1), gran_delta)
        pred_fin = pred_ini + gran_delta - pd.Timedelta(days=1)

        if pred_fin > fecha_max:
            break

        n_obs_train = ((dates_all >= ptr) & (dates_all <= train_fin)).sum()
        n_obs_pred  = ((dates_all >= pred_ini) & (dates_all <= pred_fin)).sum()

        if n_obs_train < 10 or n_obs_pred < 1:
            ptr = ptr + gran_delta
            continue

        train_ini_efectivo = inicio_base if modo == "expanding" else train_ini_rolling

        ventanas.append({
            "train_ini_efectivo": train_ini_efectivo,
            "train_fin":          train_fin,
            "pred_ini":           pred_ini,
            "pred_fin":           pred_fin,
            "etiqueta":           etiqueta_periodo(pred_ini, gran_delta),
        })

        ptr = ptr + gran_delta

    return ventanas


# ===========================================================================
# CONSTRUCCIÓN DE FEATURES (lags)
# ===========================================================================

def build_lag_features(data: np.ndarray, num_lags: int):
    """
    Construye matriz de features (lags) y vector objetivo.

    Para cada fila t (t >= num_lags):
        X[t] = [data[t-1, :], data[t-2, :], ..., data[t-num_lags, :]]
        y[t] = data[t, -1]  ← última columna = target

    Returns
    -------
    X : ndarray, shape (T-num_lags, N_vars * num_lags)
    y : ndarray, shape (T-num_lags,)
    """
    T, N = data.shape
    rows_X, rows_y = [], []
    for t in range(num_lags, T):
        lag_feats = []
        for lag in range(1, num_lags + 1):
            lag_feats.append(data[t - lag, :])
        rows_X.append(np.concatenate(lag_feats))
        rows_y.append(data[t, -1])

    if not rows_X:
        return np.empty((0, N * num_lags)), np.empty((0,))

    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


def build_predict_input(rolling_window: np.ndarray, num_lags: int) -> np.ndarray:
    """
    Construye el vector de features para predecir el siguiente paso
    usando las últimas num_lags filas de rolling_window.

    rolling_window tiene shape (num_lags, N_vars).
    Los lags se ordenan de más reciente a más antiguo:
        lag1 = rolling_window[-1], lag2 = rolling_window[-2], ...
    """
    lag_feats = []
    for lag in range(1, num_lags + 1):
        lag_feats.append(rolling_window[-lag, :])
    return np.concatenate(lag_feats).reshape(1, -1)


# ===========================================================================
# UTILIDADES DE CONSOLA
# ===========================================================================

def pedir_entero(prompt, minimo=1, maximo=None):
    while True:
        try:
            v = int(input(prompt).strip())
            if v < minimo:
                print(f"  Debe ser >= {minimo}.")
            elif maximo is not None and v > maximo:
                print(f"  Debe ser <= {maximo}.")
            else:
                return v
        except ValueError:
            print("  Ingresa un entero valido.")


def pedir_float(prompt, minimo=0.0, maximo=1.0):
    while True:
        try:
            v = float(input(prompt).strip())
            if v < minimo or v > maximo:
                print(f"  Debe estar entre {minimo} y {maximo}.")
            else:
                return v
        except ValueError:
            print("  Ingresa un numero decimal valido.")


def pedir_opcion(prompt, validas):
    while True:
        r = input(prompt).strip()
        if r in validas:
            return r
        print(f"  Opciones validas: {', '.join(validas)}")


# ===========================================================================
# INTERFAZ INTERACTIVA
# ===========================================================================

def interfaz_interactiva(dates_all, N_vars):
    f_min = pd.Timestamp(dates_all.min())
    f_max = pd.Timestamp(dates_all.max())
    anios_datos = (f_max - f_min).days / 365.25

    print("\n" + "=" * 65)
    print("  ROLLING WINDOW FORECAST  ---  COP/USD  con  MLR-HOWA")
    print("  (prediccion recursiva out-of-sample - version corregida)")
    print("=" * 65)
    print(f"  Dataset : {f_min.date()} -> {f_max.date()}  (~{anios_datos:.1f} anos)")
    print(f"  Variables: {N_vars}  ({'multivariado' if N_vars > 1 else 'univariado'})")
    print()

    print("Granularidad del periodo (horizonte = 1 periodo completo):")
    print()
    for k, (nom, _) in GRANULARIDADES.items():
        print(f"  [{k}]  {nom}")
    print()
    clave = pedir_opcion(">>> Elige la granularidad [3]: ", list(GRANULARIDADES.keys()))
    etiq_gran, gran_delta = GRANULARIDADES[clave]
    print(f"  Granularidad: {etiq_gran}")

    if gran_delta.years:
        ppa = 1.0 / gran_delta.years
    elif gran_delta.months:
        ppa = 12.0 / gran_delta.months
    else:
        ppa = 52.0 / (gran_delta.weeks or 1)

    total_periodos  = int(anios_datos * ppa)
    min_recomendado = MIN_PERIODOS_RECOMENDADOS.get(clave, 12)
    min_absoluto    = MIN_PERIODOS_ABSOLUTO
    max_entren      = max(min_absoluto, total_periodos - 2)

    sugerencias = []
    for anios_sug, desc in [(1, "1 ano"), (2, "2 anos"), (3, "3 anos"),
                             (5, "5 anos"), (10, "10 anos")]:
        n_sug = round(anios_sug * ppa)
        if min_absoluto <= n_sug <= max_entren:
            sugerencias.append((n_sug, f"{n_sug} periodos ~ {desc}"))

    print()
    print(f"Cuantos periodos de '{etiq_gran}' usar para entrenar?")
    print(f"  Minimo recomendado : {min_recomendado} periodos (~{min_recomendado/ppa:.1f} anos)")
    print(f"  Minimo absoluto    : {min_absoluto} periodos (NO recomendado)")
    print()
    for n_sug, desc in sugerencias:
        marker = "  <-- recomendado" if n_sug == min_recomendado else ""
        print(f"    {desc}{marker}")

    default_entren = min_recomendado if min_recomendado <= max_entren else max_entren

    while True:
        n_entren = pedir_entero(
            f">>> Numero de periodos de entrenamiento [{default_entren}]: ",
            minimo=min_absoluto, maximo=max_entren,
        )
        if n_entren < min_recomendado:
            print(f"\n  ADVERTENCIA: {n_entren} periodos es menor al minimo recomendado.")
            confirmar = pedir_opcion("  Continuar de todas formas? [s/n]: ", ["s", "n"])
            if confirmar == "s":
                break
            else:
                print()
                continue
        else:
            break

    print(f"  Se ejecutaran aproximadamente {total_periodos - n_entren} ventana(s).")

    print()
    print("Modo de ventana:")
    print("  [1] Rolling   -> siempre los mismos N periodos, deslizandose.")
    print("  [2] Expanding -> desde el inicio del dataset, ventana creciente.")
    modo_r = pedir_opcion(">>> Modo [1]: ", ["1", "2"])
    modo   = "rolling" if modo_r == "1" else "expanding"
    print(f"  Modo: {modo}")

    print()
    print("Numero de lags a usar como features:")
    print("  NOTA: con prediccion recursiva real, mas lags = mas estabilidad")
    print("  en horizontes largos. Sugerencias: 5 (1 sem), 22 (1 mes), 66 (3 meses)")
    num_lags = pedir_entero(">>> num_lags [22]: ", minimo=1, maximo=500)

    print()
    print("Tipo de operador MLR-OWA:")
    ops_desc = {
        "owa":    "OWA   - pesos suma=1, reordenamiento por valor",
        "howa":   "HOWA  - pesos suma>1 (heavy), reordenamiento por valor",
        "iowa":   "IOWA  - OWA con reordenamiento inducido por tiempo",
        "igowa":  "IGOWA - IOWA generalizado (con lambda)",
        "ihowa":  "IHOWA - HOWA con reordenamiento inducido por tiempo",
        "ighowa": "IGHOWA- IHOWA generalizado (con lambda)",
    }
    print()
    for k, desc in ops_desc.items():
        print(f"  [{k:6s}]  {desc}")
    op_keys = list(ops_desc.keys())
    operator = pedir_opcion("\n>>> Operador [owa]: ", op_keys + [""]).strip()
    if operator == "":
        operator = "owa"
    print(f"  Operador: {operator.upper()}")

    orness, beta, lam = 0.7, 0.5, 2.0

    if operator in ("owa", "iowa", "igowa"):
        print()
        print("  orness (0=pesimista/min, 1=optimista/max, 0.5=media aritmetica):")
        orness = pedir_float(">>> orness [0.7]: ", 0.0, 1.0) if \
            input("  Cambiar orness? [s/N]: ").strip().lower() == "s" else 0.7

    if operator in ("howa", "ihowa", "ighowa"):
        print()
        print("  beta (0=media OWA normal, 1=operador total):")
        beta = pedir_float(">>> beta [0.5]: ", 0.0, 1.0) if \
            input("  Cambiar beta? [s/N]: ").strip().lower() == "s" else 0.5

    if operator in ("igowa", "ighowa"):
        print()
        lam = float(input(">>> lambda [2.0]: ").strip() or "2.0")

    custom_w = None
    print()
    if input("  Usar pesos personalizados? [s/N]: ").strip().lower() == "s":
        print(f"  Ingrese {num_lags} pesos separados por coma:")
        try:
            raw = input("  Pesos: ").strip()
            custom_w = np.array([float(x) for x in raw.split(",")])
            if len(custom_w) != num_lags:
                print(f"  ERROR: se esperaban {num_lags} pesos. Usando automaticos.")
                custom_w = None
            else:
                print(f"  Pesos aceptados (suma={custom_w.sum():.4f})")
        except Exception:
            print("  Error al parsear. Usando automaticos.")
            custom_w = None

    hp = dict(num_lags=num_lags, operator=operator, orness=orness,
              beta=beta, lam=lam, custom_w=custom_w)

    return gran_delta, n_entren, modo, hp, etiq_gran, clave


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",        default=".")
    parser.add_argument("--data_path",        default="datos/tasa_cop_usd.csv")
    parser.add_argument("--results_dir",      default="./resultados_mlrhowa/")
    parser.add_argument("--save_predictions", action="store_true", default=False)
    cli = parser.parse_args()

    os.makedirs(cli.results_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. CARGA DE DATOS
    # -------------------------------------------------------------------
    csv_path = os.path.join(cli.root_path, cli.data_path)
    if not os.path.exists(csv_path):
        csv_path = cli.data_path

    print(f"\nCargando: {csv_path}")
    df = pd.read_csv(csv_path)

    date_col = next((c for c in df.columns if c.lower() in ("date", "fecha")), None)
    if date_col is None:
        raise ValueError("El CSV necesita una columna 'date' o 'fecha'.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    num_cols = [c for c in df.columns if c != date_col]
    N        = len(num_cols)
    features = "S" if N == 1 else "MS"

    print(f"  Variables: {N}  ->  features='{features}'")
    print(f"  Periodo  : {df[date_col].iloc[0].date()} -> {df[date_col].iloc[-1].date()}")
    print(f"  Target   : '{num_cols[-1]}' (ultima columna)")

    data_values = df[num_cols].values.astype(np.float32)
    dates_all   = pd.DatetimeIndex(df[date_col])
    y_ini       = df[date_col].iloc[0].year
    y_fin       = df[date_col].iloc[-1].year

    # -------------------------------------------------------------------
    # 2. INTERFAZ INTERACTIVA
    # -------------------------------------------------------------------
    gran_delta, n_entren, modo, hp, etiq_gran, clave_gran = \
        interfaz_interactiva(dates_all, N)

    num_lags = hp["num_lags"]
    operator = hp["operator"]
    orness   = hp["orness"]
    beta     = hp["beta"]
    lam      = hp["lam"]
    custom_w = hp["custom_w"]

    # -------------------------------------------------------------------
    # 3. GENERAR VENTANAS CALENDARIO
    # -------------------------------------------------------------------
    ventanas = generar_ventanas_calendario(dates_all, n_entren, gran_delta, modo)

    if not ventanas:
        print("\nNo se genero ninguna ventana.")
        return

    print(f"\n  Ventanas generadas  : {len(ventanas)}")
    print(f"  Primera prediccion  : {ventanas[0]['pred_ini'].date()} -> "
          f"{ventanas[0]['pred_fin'].date()}  [{ventanas[0]['etiqueta']}]")
    print(f"  Ultima prediccion   : {ventanas[-1]['pred_ini'].date()} -> "
          f"{ventanas[-1]['pred_fin'].date()}  [{ventanas[-1]['etiqueta']}]")

    # -------------------------------------------------------------------
    # 4. NOMBRES DE ARCHIVOS
    # -------------------------------------------------------------------
    etiq_s = etiq_gran.replace(" ", "_")
    modo_s = "rol" if modo == "rolling" else "exp"
    op_s   = operator

    out_csv = os.path.join(
        cli.results_dir,
        f"mlrhowa_{y_ini}-{y_fin}_{etiq_s}_{n_entren}p_{modo_s}_{op_s}.csv",
    )
    pred_csv_path = os.path.join(
        cli.results_dir,
        f"predictions_{y_ini}-{y_fin}_{etiq_s}_{n_entren}p_{modo_s}_{op_s}.csv",
    )

    if cli.save_predictions and os.path.exists(pred_csv_path):
        os.remove(pred_csv_path)

    # -------------------------------------------------------------------
    # 5. LOOP PRINCIPAL
    # -------------------------------------------------------------------
    resultados   = []
    t_ini_global = time.time()

    print("\n" + "=" * 65)
    print(f"  Granularidad  : {etiq_gran}")
    print(f"  Entrenamiento : {n_entren} periodo(s) por ventana")
    print(f"  Modo          : {modo}")
    print(f"  Operador      : {operator.upper()}")
    print(f"  num_lags      : {num_lags}")
    print(f"  Prediccion    : RECURSIVA OUT-OF-SAMPLE (sin datos futuros)")
    if operator in ("owa", "iowa", "igowa"):
        print(f"  orness        : {orness}")
    if operator in ("howa", "ihowa", "ighowa"):
        print(f"  beta          : {beta}")
    if operator in ("igowa", "ighowa"):
        print(f"  lambda        : {lam}")
    print(f"  Ventanas      : {len(ventanas)}")
    print("=" * 65)

    for i, v in enumerate(ventanas, start=1):

        t_ini_ef = v["train_ini_efectivo"]
        t_fin    = v["train_fin"]
        p_ini    = v["pred_ini"]
        p_fin    = v["pred_fin"]
        etiq     = v["etiqueta"]

        print(f"\n[{i}/{len(ventanas)}]  Entrena: {t_ini_ef.date()} -> {t_fin.date()}"
              f"   Predice: {p_ini.date()} -> {p_fin.date()}  [{etiq}]")

        mask_tr = (dates_all >= t_ini_ef) & (dates_all <= t_fin)
        mask_pr = (dates_all >= p_ini)    & (dates_all <= p_fin)

        idx_tr = np.where(mask_tr)[0]
        idx_pr = np.where(mask_pr)[0]

        n_tr_obs = len(idx_tr)
        n_pr_obs = len(idx_pr)

        min_tr_needed = num_lags + 5
        if n_tr_obs < min_tr_needed:
            print(f"  Solo {n_tr_obs} obs en entrenamiento (minimo {min_tr_needed}). Saltando.")
            continue

        if n_pr_obs < 1:
            print("  Sin datos reales en el periodo de prediccion. Saltando.")
            continue

        # --- Escalado ---
        scaler      = StandardScaler()
        data_tr_raw = data_values[idx_tr]
        scaler.fit(data_tr_raw)
        data_tr_sc  = scaler.transform(data_tr_raw)

        # --- Construccion de features con lags (sobre datos de TRAIN) ---
        X_tr, y_tr = build_lag_features(data_tr_sc, num_lags)

        if len(X_tr) < 5:
            print(f"  Muestras de entrenamiento insuficientes ({len(X_tr)}). Saltando.")
            continue

        # --- Ajuste MLR-OWA ---
        t0 = time.time()
        model = MLR_HOWA(
            operator=operator, orness=orness, beta=beta, lam=lam,
            weights=custom_w if (custom_w is not None and len(custom_w) == len(y_tr)) else None,
        )
        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            print(f"  ERROR en fit: {e}. Saltando.")
            continue

        print(f"  Ajuste MLR-OWA: {time.time()-t0:.2f}s  |  "
              f"alpha={model.coef_[0]:.4f}  |  "
              f"betas[0:3]: [{', '.join(f'{b:.3f}' for b in model.coef_[1:4])}...]")

        # -------------------------------------------------------------------
        # PREDICCION RECURSIVA COMPLETAMENTE OUT-OF-SAMPLE
        #
        # La ventana deslizante se inicializa con los ultimos num_lags dias
        # del periodo de ENTRENAMIENTO (en escala original).
        # Cada paso nuevo agrega la PREDICCION (no el dato real) a la ventana.
        # Nunca se miran datos del periodo de prediccion.
        # -------------------------------------------------------------------

        # Verificar que hay suficientes observaciones antes del periodo de prediccion
        primer_idx_pr = idx_pr[0]
        if primer_idx_pr < num_lags:
            print(f"  No hay {num_lags} observaciones previas al periodo predicho. Saltando.")
            continue

        # Ventana inicial: ultimos num_lags dias del entrenamiento (escala real)
        # Shape: (num_lags, N_vars)
        rolling_window = data_values[primer_idx_pr - num_lags: primer_idx_pr].copy()

        pred_real_list = []

        for step in range(n_pr_obs):
            # 1. Normalizar la ventana actual con el scaler de train
            window_sc = scaler.transform(rolling_window)

            # 2. Construir vector de features [lag1, lag2, ..., lag_k] (normalizado)
            x_pred = build_predict_input(window_sc, num_lags)

            # 3. Predecir en escala normalizada
            y_pred_sc = float(model.predict(x_pred)[0])

            # 4. Des-normalizar: construir fila completa para inverse_transform
            dummy        = np.zeros((1, N), dtype=np.float32)
            dummy[0, -1] = y_pred_sc
            y_pred_real  = float(scaler.inverse_transform(dummy)[0, -1])
            pred_real_list.append(y_pred_real)

            # 5. ACTUALIZAR VENTANA CON LA PREDICCION (nunca con dato real)
            #    Se actualiza SOLO la columna target (-1); las demas variables
            #    se mantienen con el ultimo valor conocido (del train o de
            #    la ultima fila de la ventana).
            next_row        = rolling_window[-1].copy()   # copia la ultima fila
            next_row[-1]    = y_pred_real                 # reemplaza solo el target
            rolling_window  = np.vstack([rolling_window[1:], next_row])
            # rolling_window ahora tiene los lags actualizados con predicciones

        pred_real_arr = np.array(pred_real_list, dtype=float).reshape(-1, 1)
        true_real_arr = data_values[idx_pr, -1:].astype(float)

        # --- Metricas en escala real ---
        mae_r, mse_r, rmse_r, mape_r, mspe_r, rse_r = metric(pred_real_arr, true_real_arr)

        # --- Metricas en escala normalizada ---
        # Transformar pred y true a escala norm para comparacion con DFGCN
        tmp_pred = np.zeros((n_pr_obs, N), dtype=np.float32)
        tmp_true = np.zeros((n_pr_obs, N), dtype=np.float32)
        tmp_pred[:, -1] = pred_real_arr[:, 0]
        tmp_true[:, -1] = true_real_arr[:, 0]
        pred_norm_arr = scaler.transform(tmp_pred)[:, -1:]
        true_norm_arr = scaler.transform(tmp_true)[:, -1:]
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(pred_norm_arr, true_norm_arr)

        print(f"  MAE={mae_r:.2f} COP  |  RMSE={rmse_r:.2f} COP  |  "
              f"MAPE={mape_r*100:.2f}%  |  MAE_norm={mae_n:.4f}")

        # --- Guardar predicciones puntuales ---
        if cli.save_predictions:
            rows = []
            for step in range(n_pr_obs):
                rows.append({
                    "periodo":        etiq,
                    "fecha_pred_ini": p_ini.strftime("%Y-%m-%d"),
                    "fecha_pred_fin": p_fin.strftime("%Y-%m-%d"),
                    "date":           dates_all[idx_pr[step]].strftime("%Y-%m-%d"),
                    "pred_cop_usd":   float(pred_real_arr[step, 0]),
                    "true_cop_usd":   float(true_real_arr[step, 0]),
                    "error_abs":      float(abs(pred_real_arr[step, 0] - true_real_arr[step, 0])),
                })
            write_header = not os.path.exists(pred_csv_path)
            pd.DataFrame(rows).to_csv(pred_csv_path, index=False,
                                      mode="a", header=write_header)

        resultados.append({
            "periodo_predicho":   etiq,
            "fecha_pred_ini":     p_ini.strftime("%Y-%m-%d"),
            "fecha_pred_fin":     p_fin.strftime("%Y-%m-%d"),
            "dias_habiles_pred":  n_pr_obs,
            "fecha_train_ini":    t_ini_ef.strftime("%Y-%m-%d"),
            "fecha_train_fin":    t_fin.strftime("%Y-%m-%d"),
            "operador":           operator.upper(),
            "num_lags":           num_lags,
            "mae_normalizado":    round(float(mae_n),  6),
            "mse_normalizado":    round(float(mse_n),  6),
            "rmse_normalizado":   round(float(rmse_n), 6),
            "mape_normalizado":   round(float(mape_n), 6),
            "mae_escala_pesos":   round(float(mae_r),  4),
            "mse_escala_pesos":   round(float(mse_r),  4),
            "rmse_escala_pesos":  round(float(rmse_r), 4),
            "mape_escala_pesos":  round(float(mape_r), 6),
        })

        pd.DataFrame(resultados).to_csv(out_csv, index=False)

    # -------------------------------------------------------------------
    # 6. RESUMEN FINAL
    # -------------------------------------------------------------------
    t_total = time.time() - t_ini_global
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL  ---  MLR-HOWA (recursivo out-of-sample)")
    print("=" * 65)

    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res.to_csv(out_csv, index=False)

        for col, nom in [
            ("mae_escala_pesos",  "MAE (pesos COP)"),
            ("rmse_escala_pesos", "RMSE (pesos COP)"),
            ("mape_escala_pesos", "MAPE real"),
            ("mae_normalizado",   "MAE normalizado"),
            ("rmse_normalizado",  "RMSE normalizado"),
        ]:
            vals = df_res[col].astype(float)
            if "mape" in col:
                print(f"  {nom:25s}: media={vals.mean()*100:.2f}%  std={vals.std()*100:.2f}%")
            else:
                print(f"  {nom:25s}: media={vals.mean():.4f}  std={vals.std():.4f}")

        print(f"\n  Ventanas ejecutadas : {len(resultados)}")
        print(f"  Tiempo total        : {t_total/60:.1f} min ({t_total:.0f} s)")
        print(f"  Resultados guardados: {out_csv}")
        if cli.save_predictions:
            print(f"  Predicciones        : {pred_csv_path}")
    else:
        print("  No se completo ninguna ventana.")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()