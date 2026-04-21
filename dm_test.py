import pandas as pd
import numpy as np
import os
from scipy.stats import norm

# =========================
# PATHS
# =========================

PATH_DFGCN = "./resultados_rolling"
PATH_RW = "./resultados_randomwalk"

# =========================
# MAPEO DE HORIZONTES
# =========================

def get_horizon_from_name(filename):
    if "1_semana" in filename:
        return 5
    elif "2_semanas" in filename:
        return 10
    elif "1_mes" in filename:
        return 20
    elif "2_meses" in filename:
        return 40
    elif "3_meses" in filename:
        return 60
    elif "6_meses" in filename:
        return 120
    elif "1_año" in filename:
        return 252
    elif "2_años" in filename:
        return 504
    else:
        return 1  # fallback

# =========================
# FUNCIONES
# =========================

def load_csv(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def prepare_df(df):
    df = df.copy()
    df["error"] = df["pred_cop_usd"] - df["true_cop_usd"]
    df["loss"] = df["error"]**2
    return df[["date", "loss"]]

def autocovariance(x, lag):
    T = len(x)
    x_mean = np.mean(x)
    return np.sum((x[lag:] - x_mean)*(x[:-lag] - x_mean)) / T

def dm_test(loss_model, loss_rw, h):
    d = loss_model - loss_rw
    T = len(d)
    
    d_mean = np.mean(d)
    
    # varianza HAC
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    
    for lag in range(1, h):
        gamma = autocovariance(d, lag)
        var_d += 2 * gamma
    
    dm_stat = d_mean / np.sqrt(var_d / T)
    
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

# =========================
# MATCH RW FILE
# =========================

def find_matching_rw(df_file):
    for rw_file in os.listdir(PATH_RW):
        if any(key in df_file for key in [
            "1_mes", "1_semana", "2_meses", "2_semanas",
            "3_meses", "6_meses", "1_año", "2_años"
        ]):
            if key in rw_file:
                return rw_file
    return None

# =========================
# LOOP PRINCIPAL
# =========================

results = []

for file in os.listdir(PATH_DFGCN):
    if not file.endswith(".csv"):
        continue
    
    print(f"Procesando: {file}")
    
    # horizonte dinámico
    h = get_horizon_from_name(file)
    
    # encontrar RW correspondiente
    rw_file = None
    for candidate in os.listdir(PATH_RW):
        if file.split("_rol")[0].replace("predictions_", "") in candidate:
            rw_file = candidate
            break
    
    if rw_file is None:
        print(f"No se encontró RW para {file}")
        continue
    
    # cargar datos
    df_model = prepare_df(load_csv(os.path.join(PATH_DFGCN, file)))
    df_rw = prepare_df(load_csv(os.path.join(PATH_RW, rw_file)))
    
    # merge
    df = pd.merge(df_model, df_rw, on="date", suffixes=("_model", "_rw"))
    
    # DM test
    dm_stat, p_value = dm_test(
        df["loss_model"].values,
        df["loss_rw"].values,
        h
    )
    
    results.append({
        "Modelo": file,
        "RW_base": rw_file,
        "Horizonte(h)": h,
        "DM_stat": dm_stat,
        "p_value": p_value
    })

# =========================
# TABLA FINAL
# =========================

df_results = pd.DataFrame(results)

df_results["Significativo_5%"] = df_results["p_value"] < 0.05

df_results = df_results.sort_values("p_value")

print("\n===== RESULTADOS DM TEST =====\n")
print(df_results.to_string(index=False))

df_results.to_csv("dm_results_all_models.csv", index=False)