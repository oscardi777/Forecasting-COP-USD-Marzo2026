"""
random_walk_rolling.py
======================
Benchmark de Random Walk sin drift con rolling window calendario.

El modelo predice: y_hat(t+h) = y(t) para todo h >= 1
(ultimo valor real observado antes del periodo predicho, sin drift).

No necesita entrenamiento. Solo requiere seleccionar el horizonte de prediccion.

Para cada ventana del calendario:
  - Toma el ultimo valor real antes del inicio del periodo predicho.
  - Repite ese valor para cada dia habil dentro del periodo predicho.

Ejemplo con horizonte 1 semana:
  Ultimo valor conocido = cierre del viernes 1993-01-01
  Predice 1993-01-04 hasta 1993-01-08 (todos con el mismo valor)
  Para la siguiente semana toma el valor real de 1993-01-08 y predice 1993-01-11 - 1993-01-15

Uso:
    python random_walk.py --data_path datos/tasa_cop_usd.csv --save_predictions
    python random_walk.py --data_path datos/tasas_BRL_COP_CHI_1993-2025.csv --save_predictions
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


GRANULARIDADES = {
    "1": ("1 semana",    relativedelta(weeks=1)),
    "2": ("2 semanas",   relativedelta(weeks=2)),
    "3": ("1 mes",       relativedelta(months=1)),
    "4": ("2 meses",     relativedelta(months=2)),
    "5": ("3 meses",     relativedelta(months=3)),
    "6": ("6 meses",     relativedelta(months=6)),
    "7": ("1 ano",       relativedelta(years=1)),
    "8": ("2 anos",      relativedelta(years=2)),
}


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


def generar_ventanas_rw(dates_all, gran_delta):
    """
    Genera ventanas cubri todo el dataset desde el inicio.
    No hay ventana de entrenamiento: el ultimo dato antes del periodo es la prediccion.
    """
    fecha_min = pd.Timestamp(dates_all.min())
    fecha_max = pd.Timestamp(dates_all.max())
    inicio_base = inicio_periodo_calendario(fecha_min, gran_delta)

    ventanas = []
    ptr = inicio_base

    while True:
        pred_ini = ptr
        pred_fin = pred_ini + gran_delta - pd.Timedelta(days=1)

        if pred_fin > fecha_max:
            break

        n_obs_pred = ((dates_all >= pred_ini) & (dates_all <= pred_fin)).sum()
        if n_obs_pred >= 1:
            ventanas.append({
                "pred_ini": pred_ini,
                "pred_fin": pred_fin,
                "etiqueta": etiqueta_periodo(pred_ini, gran_delta),
            })

        ptr = ptr + gran_delta

    return ventanas


def pedir_opcion(prompt, validas):
    while True:
        r = input(prompt).strip()
        if r in validas:
            return r
        print(f"  Opciones validas: {', '.join(validas)}")


def interfaz_interactiva(dates_all):
    f_min = pd.Timestamp(dates_all.min())
    f_max = pd.Timestamp(dates_all.max())
    anios_datos = (f_max - f_min).days / 365.25

    print("\n" + "=" * 65)
    print("  RANDOM WALK SIN DRIFT  ---  Benchmark")
    print("=" * 65)
    print(f"  Dataset: {f_min.date()} -> {f_max.date()}  (~{anios_datos:.1f} anos)")
    print()
    print("  Logica: para cada periodo, se toma el ultimo valor real")
    print("  observado antes del inicio y se repite para cada dia")
    print("  dentro del horizonte seleccionado.")
    print()
    print("Horizonte de prediccion:")
    print()
    for k, (nom, _) in GRANULARIDADES.items():
        print(f"  [{k}]  {nom}")
    print()

    clave = pedir_opcion(">>> Elige el horizonte [3]: ", list(GRANULARIDADES.keys()))
    etiq_gran, gran_delta = GRANULARIDADES[clave]
    print(f"  Horizonte seleccionado: {etiq_gran}")

    return gran_delta, etiq_gran


def main():
    parser = argparse.ArgumentParser(description="Random Walk sin drift")
    parser.add_argument("--root_path",        default=".")
    parser.add_argument("--data_path",        default="datos/tasa_cop_usd.csv")
    parser.add_argument("--results_dir",      default="./resultados_randomwalk/")
    parser.add_argument("--save_predictions", action="store_true", default=False,
                        help="Guardar CSV con predicciones puntuales de cada periodo")
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
    N = len(num_cols)
    features = "S" if N == 1 else "MS"

    print(f"  Variables: {N}  ->  features='{features}'")
    print(f"  Periodo  : {df[date_col].iloc[0].date()} -> {df[date_col].iloc[-1].date()}")

    data_values = df[num_cols].values.astype(np.float32)
    dates_all   = pd.DatetimeIndex(df[date_col])
    y_ini       = df[date_col].iloc[0].year
    y_fin       = df[date_col].iloc[-1].year

    # -------------------------------------------------------------------
    # 2. INTERFAZ
    # -------------------------------------------------------------------
    gran_delta, etiq_gran = interfaz_interactiva(dates_all)

    # -------------------------------------------------------------------
    # 3. GENERAR VENTANAS
    # -------------------------------------------------------------------
    ventanas = generar_ventanas_rw(dates_all, gran_delta)

    if not ventanas:
        print("\nNo se genero ninguna ventana. Elige otro horizonte.")
        return

    print(f"\n  Ventanas generadas  : {len(ventanas)}")
    print(f"  Primera prediccion  : {ventanas[0]['pred_ini'].date()} -> "
          f"{ventanas[0]['pred_fin'].date()}  [{ventanas[0]['etiqueta']}]")
    print(f"  Ultima prediccion   : {ventanas[-1]['pred_ini'].date()} -> "
          f"{ventanas[-1]['pred_fin'].date()}  [{ventanas[-1]['etiqueta']}]")

    # -------------------------------------------------------------------
    # 4. NOMBRES DE ARCHIVOS DE SALIDA
    # -------------------------------------------------------------------
    etiq_s = etiq_gran.replace(" ", "_")

    out_csv = os.path.join(cli.results_dir,
                           f"randomwalk_{y_ini}-{y_fin}_{etiq_s}.csv")
    pred_csv_path = os.path.join(cli.results_dir,
                                 f"predictions_randomwalk_{y_ini}-{y_fin}_{etiq_s}.csv")

    # Borrar archivo de predicciones anterior si existe para empezar limpio
    if cli.save_predictions and os.path.exists(pred_csv_path):
        os.remove(pred_csv_path)

    # -------------------------------------------------------------------
    # 5. LOOP PRINCIPAL
    # -------------------------------------------------------------------
    resultados   = []
    t_ini_global = time.time()

    print("\n" + "=" * 65)
    print(f"  Modelo    : Random Walk sin drift")
    print(f"  Horizonte : {etiq_gran}")
    print(f"  Ventanas  : {len(ventanas)}")
    print("=" * 65)

    for i, v in enumerate(ventanas, start=1):

        p_ini = v["pred_ini"]
        p_fin = v["pred_fin"]
        etiq  = v["etiqueta"]

        print(f"\n[{i}/{len(ventanas)}]  Predice: {p_ini.date()} -> {p_fin.date()}  [{etiq}]")

        mask_pr  = (dates_all >= p_ini) & (dates_all <= p_fin)
        idx_pr   = np.where(mask_pr)[0]
        n_pr_obs = len(idx_pr)

        if n_pr_obs < 1:
            print("  Sin datos en el periodo. Saltando.")
            continue

        primer_idx_pr = idx_pr[0]
        if primer_idx_pr < 1:
            print("  Sin valor previo disponible. Saltando.")
            continue

        # Ultimo valor real observado antes del inicio del periodo
        ultimo_valor_obs = data_values[primer_idx_pr - 1]

        # Prediccion: mismo valor para cada dia del periodo
        pred_raw = np.tile(ultimo_valor_obs, (n_pr_obs, 1)).astype(np.float32)
        true_raw = data_values[idx_pr]

        # Escalar con toda la historia anterior al periodo predicho
        history_raw = data_values[:primer_idx_pr]
        scaler = StandardScaler()
        scaler.fit(history_raw)

        pred_norm     = scaler.transform(pred_raw)
        true_norm     = scaler.transform(true_raw)
        true_norm_obj = true_norm[:, -1:]
        true_real_obj = true_raw[:,  -1:]
        pred_norm_obj = pred_norm[:, -1:]

        dummy = np.zeros((n_pr_obs, N), dtype=np.float32)
        dummy[:, -1] = pred_norm[:, -1]
        pred_real_obj = scaler.inverse_transform(dummy)[:, -1:]

        # Metricas
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(pred_norm_obj, true_norm_obj)
        mae_r, mse_r, rmse_r, mape_r, mspe_r, rse_r = metric(pred_real_obj, true_real_obj)

        print(f"  MAE={mae_r:.2f} COP  |  RMSE={rmse_r:.2f} COP  |  "
              f"MAPE={mape_r*100:.2f}%  |  MAE_norm={mae_n:.4f}")

        # Guardar predicciones puntuales
        if cli.save_predictions:
            rows = []
            for step in range(n_pr_obs):
                rows.append({
                    "periodo":        etiq,
                    "fecha_pred_ini": p_ini.strftime("%Y-%m-%d"),
                    "fecha_pred_fin": p_fin.strftime("%Y-%m-%d"),
                    "date":           dates_all[idx_pr[step]].strftime("%Y-%m-%d"),
                    "pred_cop_usd":   float(pred_real_obj[step, 0]),
                    "true_cop_usd":   float(true_real_obj[step, 0]),
                    "error_abs":      float(abs(pred_real_obj[step, 0] - true_real_obj[step, 0])),
                })

            write_header = not os.path.exists(pred_csv_path)
            pd.DataFrame(rows).to_csv(pred_csv_path, index=False,
                                      mode="a", header=write_header)

        # Acumular resultados por periodo
        resultados.append({
            "periodo_predicho":  etiq,
            "fecha_pred_ini":    p_ini.strftime("%Y-%m-%d"),
            "fecha_pred_fin":    p_fin.strftime("%Y-%m-%d"),
            "dias_habiles_pred": n_pr_obs,
            "mae_normalizado":   round(float(mae_n),  6),
            "mse_normalizado":   round(float(mse_n),  6),
            "rmse_normalizado":  round(float(rmse_n), 6),
            "mape_normalizado":  round(float(mape_n), 6),
            "mae_escala_pesos":  round(float(mae_r),  4),
            "mse_escala_pesos":  round(float(mse_r),  4),
            "rmse_escala_pesos": round(float(rmse_r), 4),
            "mape_escala_pesos": round(float(mape_r), 6),
        })

        pd.DataFrame(resultados).to_csv(out_csv, index=False)

    # -------------------------------------------------------------------
    # 6. RESUMEN FINAL
    # -------------------------------------------------------------------
    t_total = time.time() - t_ini_global
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL  ---  Random Walk sin drift")
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