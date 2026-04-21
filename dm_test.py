import pandas as pd
import numpy as np
import os
from scipy.stats import norm

PATH_DFGCN = "./resultados_rolling"
PATH_RW = "./resultados_randomwalk"

# =========================
# DETECTAR HORIZONTE
# =========================

def get_horizon_and_key(filename):
    if "1_semana" in filename:
        return 5, "1_semana"
    elif "2_semanas" in filename:
        return 10, "2_semanas"
    elif "1_mes" in filename:
        return 20, "1_mes"
    elif "2_meses" in filename:
        return 40, "2_meses"
    elif "3_meses" in filename:
        return 60, "3_meses"
    elif "6_meses" in filename:
        return 120, "6_meses"
    elif "1_año" in filename:
        return 252, "1_año"
    elif "2_años" in filename:
        return 504, "2_años"
    else:
        return 1, None

# =========================
# CARGA ROBUSTA
# =========================

def load_csv_safe(path):
    df = pd.read_csv(path)

    # limpiar nombres
    df.columns = df.columns.str.strip().str.lower()

    # posibles nombres de fecha
    for col in ["date", "fecha", "time"]:
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col])
            break
    else:
        # intentar usar índice
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

# =========================
# PREPARAR
# =========================

def prepare_df(df):
    df = df.copy()

    if "pred_cop_usd" not in df.columns or "true_cop_usd" not in df.columns:
        raise ValueError("Columnas faltantes en el CSV")

    df["error"] = df["pred_cop_usd"] - df["true_cop_usd"]
    df["loss"] = df["error"]**2

    return df[["date", "loss"]].dropna()

# =========================
# AUTOCOV
# =========================

def autocovariance(x, lag):
    T = len(x)
    x_mean = np.mean(x)
    return np.sum((x[lag:] - x_mean)*(x[:-lag] - x_mean)) / T

# =========================
# DM TEST
# =========================

def dm_test(loss_model, loss_rw, h):
    d = loss_model - loss_rw
    T = len(d)

    d_mean = np.mean(d)

    gamma0 = np.var(d, ddof=0)
    var_d = gamma0

    for lag in range(1, h):
        gamma = autocovariance(d, lag)
        var_d += 2 * gamma

    dm_stat = d_mean / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return dm_stat, p_value

# =========================
# MATCH RW POR HORIZONTE
# =========================

def find_rw_file(horizon_key):
    for file in os.listdir(PATH_RW):
        if horizon_key and horizon_key in file:
            return file
    return None

# =========================
# LOOP PRINCIPAL
# =========================

results = []

for file in os.listdir(PATH_DFGCN):
    if not file.endswith(".csv"):
        continue

    print(f"\nProcesando: {file}")

    h, key = get_horizon_and_key(file)

    if key is None:
        print("⚠️ No se pudo detectar horizonte")
        continue

    rw_file = find_rw_file(key)

    if rw_file is None:
        print("⚠️ No se encontró RW correspondiente")
        continue

    try:
        df_model = prepare_df(load_csv_safe(os.path.join(PATH_DFGCN, file)))
        df_rw = prepare_df(load_csv_safe(os.path.join(PATH_RW, rw_file)))

        df = pd.merge(df_model, df_rw, on="date", suffixes=("_model", "_rw"))

        if len(df) < 10:
            print("⚠️ Muy pocos datos tras merge")
            continue

        dm_stat, p_value = dm_test(
            df["loss_model"].values,
            df["loss_rw"].values,
            h
        )

        results.append({
            "Modelo": file,
            "RW": rw_file,
            "h": h,
            "DM": dm_stat,
            "p_value": p_value
        })

    except Exception as e:
        print(f"❌ Error en {file}: {e}")

# =========================
# RESULTADOS
# =========================

df_results = pd.DataFrame(results)

if not df_results.empty:
    df_results["significativo_5%"] = df_results["p_value"] < 0.05
    df_results = df_results.sort_values("p_value")

    print("\n===== RESULTADOS FINALES =====\n")
    print(df_results.to_string(index=False))

    df_results.to_csv("dm_results.csv", index=False)
else:
    print("❌ No se generaron resultados")