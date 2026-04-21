"""
random_walk_rolling.py
======================
Benchmark de Random Walk sin drift con rolling window calendario.
Usa exactamente las mismas ventanas de tiempo que rolling_forecast_window_PRECISE.py
para permitir comparación directa de métricas.

El modelo predice: y_hat(t+h) = y(t) para todo h >= 1
(último valor observado antes del período predicho, sin drift).

Uso:
    python random_walk_rolling.py --data_path datos/tasa_cop_usd.csv --save_predictions
    python random_walk_rolling.py --data_path datos/tasas_BRL_COP_CHI_1993-2025.csv --save_predictions
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

# ---------------------------------------------------------------------------
# Raíz del repositorio en el path de Python
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ====================== REPRODUCIBILIDAD ======================
fix_seed = 42
random.seed(fix_seed)
np.random.seed(fix_seed)
# ==============================================================

# ---------------------------------------------------------------------------
# Módulos del repositorio
# ---------------------------------------------------------------------------
from utils.metrics import metric   # metric(pred, true) → mae, mse, rmse, mape, mspe, rse


# ===========================================================================
# GRANULARIDADES DISPONIBLES
# ===========================================================================
GRANULARIDADES = {
    "1": ("1 semana",    relativedelta(weeks=1)),
    "2": ("2 semanas",   relativedelta(weeks=2)),
    "3": ("1 mes",       relativedelta(months=1)),
    "4": ("2 meses",     relativedelta(months=2)),
    "5": ("3 meses",     relativedelta(months=3)),
    "6": ("6 meses",     relativedelta(months=6)),
    "7": ("1 año",       relativedelta(years=1)),
    "8": ("2 años",      relativedelta(years=2)),
}


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
# CONSOLA — utilidades
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


def pedir_opcion(prompt, validas):
    while True:
        r = input(prompt).strip()
        if r in validas:
            return r
        print(f"  Opciones validas: {', '.join(validas)}")


# ===========================================================================
# INTERFAZ INTERACTIVA
# ===========================================================================
def interfaz_interactiva(dates_all):
    f_min = pd.Timestamp(dates_all.min())
    f_max = pd.Timestamp(dates_all.max())
    anios_datos = (f_max - f_min).days / 365.25

    print("\n" + "=" * 65)
    print("  RANDOM WALK SIN DRIFT  —  Rolling Window  —  COP/USD")
    print("=" * 65)
    print(f"  Dataset: {f_min.date()} -> {f_max.date()}  (~{anios_datos:.1f} anos)")
    print()

    # -------------------------------------------------------------------
    # 1. GRANULARIDAD
    # -------------------------------------------------------------------
    print("Granularidad del periodo (horizonte = paso = 1 periodo completo):")
    print()
    for k, (nom, _) in GRANULARIDADES.items():
        print(f"  [{k}]  {nom}")
    print()

    clave = pedir_opcion(">>> Elige la granularidad [3]: ", list(GRANULARIDADES.keys()))
    etiq_gran, gran_delta = GRANULARIDADES[clave]
    print(f"  Granularidad: {etiq_gran}")

    # -------------------------------------------------------------------
    # 2. NÚMERO DE PERÍODOS DE ENTRENAMIENTO
    # -------------------------------------------------------------------
    if gran_delta.years:
        ppa = 1.0 / gran_delta.years
    elif gran_delta.months:
        ppa = 12.0 / gran_delta.months
    else:
        ppa = 52.0 / (gran_delta.weeks or 1)

    total_periodos = int(anios_datos * ppa)
    min_entren     = 2
    max_entren     = max(2, total_periodos - 2)

    sugerencias = []
    for anios_sug, desc in [(1, "1 año"), (2, "2 años"), (3, "3 años"),
                             (5, "5 años"), (10, "10 años")]:
        n_sug = round(anios_sug * ppa)
        if min_entren <= n_sug <= max_entren:
            sugerencias.append((n_sug, f"{n_sug} periodos ~ {desc}"))

    print()
    print(f"Cuantos periodos completos de '{etiq_gran}' usar como ventana de referencia?")
    print("(El random walk no entrena, pero esto define desde donde se calcula el ultimo")
    print(" valor observado, igual que en rolling_forecast_window_PRECISE.py)")
    for n_sug, desc in sugerencias:
        print(f"    {desc}")

    default_entren = sugerencias[-2][0] if len(sugerencias) >= 2 else min_entren

    n_entren = pedir_entero(
        f">>> Numero de periodos de referencia [{default_entren}]: ",
        minimo=min_entren, maximo=max_entren,
    )

    n_ventanas_est = total_periodos - n_entren
    print(f"  Se ejecutaran aproximadamente {n_ventanas_est} ventana(s).")

    # -------------------------------------------------------------------
    # 3. MODO: rolling vs expanding
    # -------------------------------------------------------------------
    print()
    print("Modo de ventana:")
    print("  [1] Rolling   → siempre los mismos N periodos, deslizandose.")
    print("  [2] Expanding → desde el inicio del dataset, ventana creciente.")
    modo_r = pedir_opcion(">>> Modo [1]: ", ["1", "2"])
    modo   = "rolling" if modo_r == "1" else "expanding"
    print(f"  Modo: {modo}")

    return gran_delta, n_entren, modo, etiq_gran


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Random Walk sin drift — Rolling Window")
    parser.add_argument("--root_path",        default=".")
    parser.add_argument("--data_path",        default="datos/tasa_cop_usd.csv")
    parser.add_argument("--results_dir",      default="./resultados_randomwalk/")
    parser.add_argument("--save_predictions", action="store_true", default=False,
                        help="Guardar CSV con las predicciones de cada ventana")
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

    data_values = df[num_cols].values.astype(np.float32)   # (T, N)
    dates_all   = pd.DatetimeIndex(df[date_col])
    y_ini       = df[date_col].iloc[0].year
    y_fin       = df[date_col].iloc[-1].year

    # -------------------------------------------------------------------
    # 2. INTERFAZ INTERACTIVA
    # -------------------------------------------------------------------
    gran_delta, n_entren, modo, etiq_gran = interfaz_interactiva(dates_all)

    # -------------------------------------------------------------------
    # 3. GENERAR VENTANAS CALENDARIO
    # -------------------------------------------------------------------
    ventanas = generar_ventanas_calendario(dates_all, n_entren, gran_delta, modo)

    if not ventanas:
        print("\nNo se genero ninguna ventana. Reduce n_entren o elige otra granularidad.")
        return

    print(f"\n  Ventanas generadas  : {len(ventanas)}")
    print(f"  Primera prediccion  : {ventanas[0]['pred_ini'].date()} -> "
          f"{ventanas[0]['pred_fin'].date()}  [{ventanas[0]['etiqueta']}]")
    print(f"  Ultima prediccion   : {ventanas[-1]['pred_ini'].date()} -> "
          f"{ventanas[-1]['pred_fin'].date()}  [{ventanas[-1]['etiqueta']}]")

    # -------------------------------------------------------------------
    # 4. NOMBRES DE ARCHIVOS DE SALIDA
    # -------------------------------------------------------------------
    etiq_s  = etiq_gran.replace(" ", "_")
    modo_s  = "rol" if modo == "rolling" else "exp"

    out_csv = os.path.join(
        cli.results_dir,
        f"randomwalk_{y_ini}-{y_fin}_{etiq_s}_{n_entren}p_{modo_s}.csv",
    )

    pred_csv_path = os.path.join(
        cli.results_dir,
        f"predictions_randomwalk_{y_ini}-{y_fin}_{etiq_s}_{n_entren}p_{modo_s}.csv",
    )

    # -------------------------------------------------------------------
    # 5. LOOP PRINCIPAL
    # -------------------------------------------------------------------
    resultados   = []
    t_ini_global = time.time()

    print("\n" + "=" * 65)
    print(f"  Modelo       : Random Walk sin drift")
    print(f"  Granularidad : {etiq_gran}")
    print(f"  Ventana ref. : {n_entren} periodo(s)")
    print(f"  Modo         : {modo}")
    print(f"  Ventanas     : {len(ventanas)}")
    print("=" * 65)

    for i, v in enumerate(ventanas, start=1):

        t_ini_ef = v["train_ini_efectivo"]
        t_fin    = v["train_fin"]
        p_ini    = v["pred_ini"]
        p_fin    = v["pred_fin"]
        etiq     = v["etiqueta"]

        print(f"\n[{i}/{len(ventanas)}]  Ref: {t_ini_ef.date()} -> {t_fin.date()}"
              f"   Predice: {p_ini.date()} -> {p_fin.date()}  [{etiq}]")

        # Máscaras de índices
        mask_tr = (dates_all >= t_ini_ef) & (dates_all <= t_fin)
        mask_pr = (dates_all >= p_ini)    & (dates_all <= p_fin)

        idx_tr = np.where(mask_tr)[0]
        idx_pr = np.where(mask_pr)[0]

        n_pr_obs = len(idx_pr)

        if len(idx_tr) < 1 or n_pr_obs < 1:
            print(f"  Sin datos suficientes. Saltando.")
            continue

        # Último valor observado inmediatamente antes del período predicho
        primer_idx_pr = idx_pr[0]
        if primer_idx_pr < 1:
            print(f"  Sin valor previo al periodo predicho. Saltando.")
            continue

        ultimo_valor_obs = data_values[primer_idx_pr - 1]  # shape (N,)

        # ---------------------------------------------------------------
        # PREDICCIÓN: repetir el último valor para cada paso del período
        # y_hat(t+1) = y_hat(t+2) = ... = y_hat(t+h) = y(t)
        # ---------------------------------------------------------------
        pred_raw = np.tile(ultimo_valor_obs, (n_pr_obs, 1)).astype(np.float32)  # (n_pr_obs, N)

        # Escalar usando los datos de la ventana de referencia
        scaler = StandardScaler()
        data_tr_raw = data_values[idx_tr]
        scaler.fit(data_tr_raw)

        pred_norm = scaler.transform(pred_raw)
        true_raw  = data_values[idx_pr]
        true_norm = scaler.transform(true_raw)

        # Columna objetivo: la última variable (igual que en rolling_forecast_window_PRECISE.py)
        true_norm_obj = true_norm[:, -1:]
        true_real_obj = true_raw[:,  -1:]
        pred_norm_obj = pred_norm[:, -1:]

        # Desnormalizar predicción para escala real COP
        dummy = np.zeros((n_pr_obs, N), dtype=np.float32)
        dummy[:, -1] = pred_norm[:, -1]
        pred_real_obj = scaler.inverse_transform(dummy)[:, -1:]

        # ---------------------------------------------------------------
        # MÉTRICAS
        # ---------------------------------------------------------------
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(pred_norm_obj, true_norm_obj)
        mae_r, mse_r, rmse_r, mape_r, mspe_r, rse_r = metric(pred_real_obj, true_real_obj)

        print(f"  MAE={mae_r:.2f} COP  |  RMSE={rmse_r:.2f} COP  |  "
              f"MAPE={mape_r*100:.2f}%  |  MAE_norm={mae_n:.4f}")

        # ---------------------------------------------------------------
        # GUARDAR PREDICCIONES EN CSV (si se solicitó)
        # ---------------------------------------------------------------
        if cli.save_predictions:
            rows = []
            for step in range(n_pr_obs):
                rows.append({
                    "periodo":            etiq,
                    "fecha_pred_ini":     p_ini.strftime("%Y-%m-%d"),
                    "fecha_pred_fin":     p_fin.strftime("%Y-%m-%d"),
                    "date":               dates_all[idx_pr[step]].strftime("%Y-%m-%d"),
                    "pred_cop_usd":       float(pred_real_obj[step, 0]),
                    "true_cop_usd":       float(true_real_obj[step, 0]),
                    "error_abs":          float(abs(pred_real_obj[step, 0] - true_real_obj[step, 0])),
                })

            window_pred_df = pd.DataFrame(rows)

            if not os.path.exists(pred_csv_path):
                window_pred_df.to_csv(pred_csv_path, index=False, mode="w")
                print(f"  Archivo de predicciones creado: {pred_csv_path}")
            else:
                window_pred_df.to_csv(pred_csv_path, index=False, mode="a", header=False)
                print(f"  Predicciones anadidas al archivo: {pred_csv_path}")

        # ---------------------------------------------------------------
        # ACUMULAR RESULTADOS
        # ---------------------------------------------------------------
        resultados.append({
            "periodo_predicho":   etiq,
            "fecha_pred_ini":     p_ini.strftime("%Y-%m-%d"),
            "fecha_pred_fin":     p_fin.strftime("%Y-%m-%d"),
            "dias_habiles_pred":  n_pr_obs,
            "fecha_ref_ini":      t_ini_ef.strftime("%Y-%m-%d"),
            "fecha_ref_fin":      t_fin.strftime("%Y-%m-%d"),
            "mae_normalizado":    round(float(mae_n),  6),
            "mse_normalizado":    round(float(mse_n),  6),
            "rmse_normalizado":   round(float(rmse_n), 6),
            "mape_normalizado":   round(float(mape_n), 6),
            "mae_escala_pesos":   round(float(mae_r),  4),
            "mse_escala_pesos":   round(float(mse_r),  4),
            "rmse_escala_pesos":  round(float(rmse_r), 4),
            "mape_escala_pesos":  round(float(mape_r), 6),
        })

        # Guardado incremental tras cada ventana
        pd.DataFrame(resultados).to_csv(out_csv, index=False)

    # -------------------------------------------------------------------
    # 6. RESUMEN FINAL
    # -------------------------------------------------------------------
    t_total = time.time() - t_ini_global
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL  —  Random Walk sin drift")
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