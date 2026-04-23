"""
dm_test.py
==========
Calcula el test de Diebold-Mariano (DM) para todos los archivos de predicciones
disponibles en las carpetas de resultados de DFGCN y Random Walk.

Estructura esperada de archivos:
  ./resultados_rolling/
      predictions_1993-2025_1_mes_12p_rol.csv
      predictions_1993-2025_1_mes_24p_rol.csv
      predictions_1993-2025_2_meses_12p_rol.csv
      ...
  ./resultados_randomwalk/
      predictions_randomwalk_1993-2025_1_mes.csv
      predictions_randomwalk_1993-2025_2_meses.csv
      ...

Para cada archivo de DFGCN detecta el horizonte (1_mes, 2_meses, etc.)
y lo compara contra el archivo de Random Walk con el mismo horizonte.

Columnas requeridas en cada CSV:
  date, pred_cop_usd, true_cop_usd

Test DM (Diebold & Mariano, 1995):
  - Diferencial de perdidas: d_t = e1_t^2 - e2_t^2
    donde e1 = error del modelo 1 (DFGCN), e2 = error del modelo 2 (RW)
  - H0: E[d_t] = 0  (igual precision de pronostico)
  - H1: E[d_t] != 0 (diferente precision)
  - Estadistico: DM = d_bar / sqrt(var_LRV / T)
    donde var_LRV incluye autocovarianzas hasta lag h-1 (h = horizonte en dias)
  - DM < 0 significa DFGCN es mejor (menor error cuadratico)
  - DM > 0 significa Random Walk es mejor

Uso:
    python dm_test.py
    python dm_test.py --path_dfgcn ./resultados_rolling/ --path_rw ./resultados_randomwalk/
    python dm_test.py --output dm_resultados.csv
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

warnings.filterwarnings("ignore")


# ===========================================================================
# MAPEO DE HORIZONTES
# Detecta el horizonte en el nombre del archivo y devuelve (h_dias, clave)
# h_dias: numero aproximado de dias habiles en el horizonte
#   (usado para la correccion de autocovarianza en el DM test)
# clave: texto que identifica el horizonte en los nombres de archivo
# ===========================================================================
HORIZONTE_MAP = [
    ("1_semana",  5,   "1_semana"),
    ("2_semanas", 10,  "2_semanas"),
    ("1_mes",     20,  "1_mes"),
    ("2_meses",   40,  "2_meses"),
    ("3_meses",   60,  "3_meses"),
    ("6_meses",   120, "6_meses"),
    ("1_ano",     252, "1_ano"),
    ("2_anos",    504, "2_anos"),
    # variantes con tilde (por si acaso)
    ("1_año",     252, "1_ano"),
    ("2_años",    504, "2_anos"),
]


def detectar_horizonte(filename):
    """Devuelve (h_dias, clave_horizonte) o (None, None) si no se reconoce."""
    fn = filename.lower()
    for patron, h, clave in HORIZONTE_MAP:
        if patron in fn:
            return h, clave
    return None, None


# ===========================================================================
# CARGA ROBUSTA DE CSV
# ===========================================================================
def cargar_predicciones(path):
    """
    Carga un CSV de predicciones y devuelve DataFrame con columnas estandar:
      date (datetime), pred_cop_usd (float), true_cop_usd (float)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detectar columna de fecha
    date_col = None
    for c in df.columns:
        if any(x in c for x in ["date", "fecha"]):
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No se encontro columna de fecha en {os.path.basename(path)}")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Detectar columnas de prediccion y verdad
    pred_col = None
    true_col = None
    for c in df.columns:
        if "pred" in c and ("cop" in c or "usd" in c):
            pred_col = c
        if ("true" in c or "real" in c or "actual" in c) and ("cop" in c or "usd" in c):
            true_col = c

    if pred_col is None or true_col is None:
        raise ValueError(
            f"No se encontraron columnas pred/true en {os.path.basename(path)}.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df.rename(columns={pred_col: "pred_cop_usd", true_col: "true_cop_usd"})
    df = df[["date", "pred_cop_usd", "true_cop_usd"]].dropna()
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ===========================================================================
# TEST DIEBOLD-MARIANO
# ===========================================================================
def autocovarianza(x, lag):
    """Autocovarianza de la serie x al lag dado."""
    T = len(x)
    x_mean = np.mean(x)
    return np.sum((x[lag:] - x_mean) * (x[:-lag] - x_mean)) / T


def dm_test(df_model, df_rw, h):
    """
    Calcula el estadistico DM entre dos modelos usando perdida cuadratica.

    Parametros
    ----------
    df_model : DataFrame con columnas date, pred_cop_usd, true_cop_usd  (DFGCN)
    df_rw    : DataFrame con columnas date, pred_cop_usd, true_cop_usd  (RW)
    h        : int — horizonte en pasos (dias habiles), para la varianza LRV

    Devuelve
    --------
    dict con: dm_stat, p_value, N_obs, mae_model, mae_rw, rmse_model, rmse_rw,
              mejor_modelo
    """
    # Alinear por fecha
    merged = pd.merge(df_model, df_rw, on="date", suffixes=("_m", "_rw"))

    if len(merged) < 10:
        raise ValueError(f"Solo {len(merged)} observaciones en comun. Minimo 10.")

    e1 = merged["pred_cop_usd_m"].values  - merged["true_cop_usd_m"].values
    e2 = merged["pred_cop_usd_rw"].values - merged["true_cop_usd_rw"].values
    true_vals = merged["true_cop_usd_m"].values

    loss1 = e1 ** 2
    loss2 = e2 ** 2
    d = loss1 - loss2
    T = len(d)

    d_mean = np.mean(d)

    # Varianza de largo plazo (Harvey, Leybourne & Newbold, 1997)
    # Incluye autocovarianzas hasta lag h-1
    gamma0 = np.var(d, ddof=0)
    var_lrv = gamma0
    for lag in range(1, h):
        if lag < T:
            gamma = autocovarianza(d, lag)
            var_lrv += 2 * gamma

    if var_lrv <= 0:
        var_lrv = gamma0  # fallback si la varianza LRV es negativa

    # Estadistico DM con correccion de muestra pequena (Harvey et al. 1997)
    dm_stat = d_mean / np.sqrt(var_lrv / T)

    # p-value bilateral con distribucion t(T-1)
    p_value = 2 * (1 - t_dist.cdf(abs(dm_stat), df=T - 1))

    # Metricas adicionales
    mae_model = float(np.mean(np.abs(e1)))
    mae_rw    = float(np.mean(np.abs(e2)))
    rmse_model = float(np.sqrt(np.mean(e1 ** 2)))
    rmse_rw    = float(np.sqrt(np.mean(e2 ** 2)))

    # MAPE
    mask = true_vals != 0
    mape_model = float(np.mean(np.abs(e1[mask] / true_vals[mask]))) * 100
    mape_rw    = float(np.mean(np.abs(e2[mask] / true_vals[mask]))) * 100

    mejor = "DFGCN" if dm_stat < 0 else "RandomWalk"
    if p_value >= 0.05:
        mejor = "Sin diferencia significativa"

    return {
        "dm_stat":    round(dm_stat,   6),
        "p_value":    round(p_value,  10),
        "sig_5pct":   p_value < 0.05,
        "sig_1pct":   p_value < 0.01,
        "mejor_modelo": mejor,
        "N_obs":      T,
        "mae_dfgcn":  round(mae_model,  4),
        "mae_rw":     round(mae_rw,     4),
        "rmse_dfgcn": round(rmse_model, 4),
        "rmse_rw":    round(rmse_rw,    4),
        "mape_dfgcn": round(mape_model, 4),
        "mape_rw":    round(mape_rw,    4),
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Test DM: DFGCN vs Random Walk")
    parser.add_argument("--path_dfgcn", default="./resultados_rolling/",
                        help="Carpeta con archivos predictions_*.csv del DFGCN")
    parser.add_argument("--path_rw",    default="./resultados_randomwalk/",
                        help="Carpeta con archivos predictions_randomwalk_*.csv")
    parser.add_argument("--output",     default="dm_results_completo.csv",
                        help="Nombre del CSV de salida con los resultados")
    cli = parser.parse_args()

    # -------------------------------------------------------------------
    # 1. INDEXAR ARCHIVOS DE RANDOM WALK (uno por horizonte)
    # -------------------------------------------------------------------
    rw_por_horizonte = {}
    if os.path.isdir(cli.path_rw):
        for fname in sorted(os.listdir(cli.path_rw)):
            if not (fname.endswith(".csv") and "predictions" in fname.lower()):
                continue
            _, clave = detectar_horizonte(fname)
            if clave:
                rw_por_horizonte[clave] = os.path.join(cli.path_rw, fname)

    if not rw_por_horizonte:
        print(f"\nNo se encontraron archivos de Random Walk en: {cli.path_rw}")
        print("Asegurate de haber corrido random_walk_rolling.py con --save_predictions")
        return

    print(f"\nArchivos Random Walk encontrados:")
    for k, p in sorted(rw_por_horizonte.items()):
        print(f"  [{k}]  {os.path.basename(p)}")

    # -------------------------------------------------------------------
    # 2. INDEXAR ARCHIVOS DE DFGCN (varios por horizonte)
    # -------------------------------------------------------------------
    dfgcn_files = []
    if os.path.isdir(cli.path_dfgcn):
        for fname in sorted(os.listdir(cli.path_dfgcn)):
            if not (fname.endswith(".csv") and "predictions" in fname.lower()):
                continue
            # Excluir archivos del random walk si por error estan en la misma carpeta
            if "randomwalk" in fname.lower():
                continue
            h, clave = detectar_horizonte(fname)
            if clave:
                dfgcn_files.append({
                    "fname":  fname,
                    "path":   os.path.join(cli.path_dfgcn, fname),
                    "h":      h,
                    "clave":  clave,
                })

    if not dfgcn_files:
        print(f"\nNo se encontraron archivos de DFGCN en: {cli.path_dfgcn}")
        print("Asegurate de haber corrido rolling_forecast_window_PRECISE.py con --save_predictions")
        return

    print(f"\nArchivos DFGCN encontrados: {len(dfgcn_files)}")
    for f in dfgcn_files:
        rw_disponible = "OK" if f["clave"] in rw_por_horizonte else "SIN RW"
        print(f"  [{f['clave']:12s}]  {f['fname']}  [{rw_disponible}]")

    # -------------------------------------------------------------------
    # 3. CALCULAR DM PARA CADA PAR
    # -------------------------------------------------------------------
    resultados = []
    errores    = []

    print("\n" + "=" * 80)
    print("  CALCULANDO TEST DIEBOLD-MARIANO")
    print("=" * 80)

    for f in dfgcn_files:
        clave = f["clave"]
        h     = f["h"]

        if clave not in rw_por_horizonte:
            msg = (f"Sin archivo RW para horizonte '{clave}'. "
                   f"Corre random_walk_rolling.py con ese horizonte primero.")
            print(f"\n  OMITIDO: {f['fname']}")
            print(f"    {msg}")
            errores.append({"archivo": f["fname"], "error": msg})
            continue

        print(f"\n  Comparando: {f['fname']}")
        print(f"  vs          {os.path.basename(rw_por_horizonte[clave])}")
        print(f"  Horizonte   {clave}  (h={h} dias habiles)")

        try:
            df_model = cargar_predicciones(f["path"])
            df_rw    = cargar_predicciones(rw_por_horizonte[clave])

            res = dm_test(df_model, df_rw, h)

            fila = {
                "archivo_dfgcn":    f["fname"],
                "archivo_rw":       os.path.basename(rw_por_horizonte[clave]),
                "horizonte":        clave,
                "h_dias":           h,
                "N_obs":            res["N_obs"],
                "dm_stat":          res["dm_stat"],
                "p_value":          res["p_value"],
                "significativo_5%": res["sig_5pct"],
                "significativo_1%": res["sig_1pct"],
                "mejor_modelo":     res["mejor_modelo"],
                "mae_dfgcn":        res["mae_dfgcn"],
                "mae_rw":           res["mae_rw"],
                "rmse_dfgcn":       res["rmse_dfgcn"],
                "rmse_rw":          res["rmse_rw"],
                "mape_dfgcn_%":     res["mape_dfgcn"],
                "mape_rw_%":        res["mape_rw"],
            }

            resultados.append(fila)

            signo = "<-- DFGCN mejor" if res["dm_stat"] < 0 else "<-- RW mejor"
            sig   = " *** p<0.01" if res["sig_1pct"] else (" ** p<0.05" if res["sig_5pct"] else "")
            print(f"    DM = {res['dm_stat']:8.4f}  p = {res['p_value']:.6f}{sig}  "
                  f"N={res['N_obs']}  {signo}")
            print(f"    MAE  DFGCN={res['mae_dfgcn']:.2f}  RW={res['mae_rw']:.2f}  "
                  f"RMSE  DFGCN={res['rmse_dfgcn']:.2f}  RW={res['rmse_rw']:.2f}")
            print(f"    MAPE DFGCN={res['mape_dfgcn']:.2f}%  RW={res['mape_rw']:.2f}%")

        except Exception as e:
            msg = str(e)
            print(f"    ERROR: {msg}")
            errores.append({"archivo": f["fname"], "error": msg})

    # -------------------------------------------------------------------
    # 4. TABLA RESUMEN EN CONSOLA
    # -------------------------------------------------------------------
    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res = df_res.sort_values(["horizonte", "archivo_dfgcn"]).reset_index(drop=True)

        print("\n" + "=" * 80)
        print("  TABLA RESUMEN — TEST DIEBOLD-MARIANO (DFGCN vs Random Walk)")
        print("=" * 80)
        print()

        # Cabecera
        header = (f"{'Archivo DFGCN':<45} {'Horizonte':<12} {'DM':>8} {'p-value':>10} "
                  f"{'Sig':>5} {'Mejor':>18} {'N':>6}")
        print(header)
        print("-" * len(header))

        for _, row in df_res.iterrows():
            sig = "***" if row["significativo_1%"] else ("**" if row["significativo_5%"] else "")
            print(f"  {row['archivo_dfgcn']:<43} {row['horizonte']:<12} "
                  f"{row['dm_stat']:>8.4f} {row['p_value']:>10.6f} "
                  f"{sig:>5} {row['mejor_modelo']:>18} {row['N_obs']:>6}")

        print()
        print("  Nota: DM < 0 => DFGCN tiene menor error. DM > 0 => Random Walk tiene menor error.")
        print("  *** p<0.01  ** p<0.05")
        print()

        # Guardar CSV
        df_res.to_csv(cli.output, index=False)
        print(f"  Resultados guardados en: {cli.output}")
        print(f"  Total comparaciones    : {len(resultados)}")

        if errores:
            print(f"  Omitidos por error     : {len(errores)}")
            for e in errores:
                print(f"    {e['archivo']}: {e['error']}")

    else:
        print("\nNo se genero ningun resultado.")
        if errores:
            for e in errores:
                print(f"  ERROR en {e['archivo']}: {e['error']}")

    print()


if __name__ == "__main__":
    main()