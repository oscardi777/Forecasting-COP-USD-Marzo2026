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
# DETECTAR HORIZONTE (robusto a prefijos "predictions_" o "rolling_")
# =========================
def get_horizon_and_key(filename):
    filename = filename.lower()
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
    elif "1_año" in filename or "1_ano" in filename:
        return 252, "1_año"
    elif "2_años" in filename or "2_anos" in filename:
        return 504, "2_años"
    else:
        return None, None

# =========================
# CARGA ROBUSTA DE CSV (maneja diferentes nombres de columnas)
# =========================
def load_csv_safe(path):
    try:
        df = pd.read_csv(path)
        
        # Limpiar nombres de columnas
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Detectar columna de fecha
        date_col = None
        for col in df.columns:
            if any(x in col for x in ["date", "fecha", "time", "index"]):
                date_col = col
                break
        if date_col:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            # Si no hay columna obvia, usar índice o primera columna
            df = df.reset_index()
            df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        
        # Detectar columnas de predicción y valor real (robusto)
        pred_col = None
        true_col = None
        for col in df.columns:
            if ("pred" in col and "cop" in col) or col == "pred_cop_usd":
                pred_col = col
            if ("true" in col and "cop" in col) or col == "true_cop_usd":
                true_col = col
        
        if pred_col is None or true_col is None:
            print(f"⚠️ Columnas de pred/true no encontradas en {os.path.basename(path)}")
            print(f"   Columnas disponibles: {list(df.columns)}")
            raise ValueError(f"Columnas faltantes en {os.path.basename(path)}")
        
        # Renombrar a nombres estándar
        df = df.rename(columns={pred_col: "pred_cop_usd", true_col: "true_cop_usd"})
        
        return df
    except Exception as e:
        print(f"❌ Error al leer {os.path.basename(path)}: {e}")
        raise

# =========================
# PREPARAR DATAFRAME (calcular loss = error²)
# =========================
def prepare_df(df):
    df = df.copy()
    if "pred_cop_usd" not in df.columns or "true_cop_usd" not in df.columns:
        raise ValueError("Columnas pred_cop_usd / true_cop_usd no encontradas después de carga")
    
    df["error"] = df["pred_cop_usd"] - df["true_cop_usd"]
    df["loss"] = df["error"] ** 2
    
    return df[["date", "loss"]].dropna()

# =========================
# AUTOCOVARIANZA (para DM test)
# =========================
def autocovariance(x, lag):
    T = len(x)
    x_mean = np.mean(x)
    return np.sum((x[lag:] - x_mean) * (x[:-lag] - x_mean)) / T

# =========================
# DIEBOLD-MARIANO TEST (versión estándar para pronósticos h-pasos)
# =========================
def dm_test(loss_model, loss_rw, h):
    d = loss_model - loss_rw
    T = len(d)
    
    if T < 10:
        raise ValueError("Muy pocos datos para el test DM")
    
    d_mean = np.mean(d)
    
    # Varianza robusta (autocorrelación hasta lag h-1)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        gamma = autocovariance(d, lag)
        var_d += 2 * gamma
    
    dm_stat = d_mean / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

# =========================
# ENCONTRAR RW CORRESPONDIENTE POR HORIZONTE
# =========================
def find_rw_file(horizon_key):
    for file in os.listdir(PATH_RW):
        if file.endswith(".csv") and horizon_key in file.lower():
            return file
    return None

# =========================
# LOOP PRINCIPAL
# =========================
results = []

print("🔍 Buscando archivos en:", PATH_DFGCN)
for file in sorted(os.listdir(PATH_DFGCN)):
    if not file.endswith(".csv"):
        continue
    
    h, key = get_horizon_and_key(file)
    if key is None:
        continue  # Solo procesar archivos con horizonte conocido
    
    rw_file = find_rw_file(key)
    if rw_file is None:
        print(f"⚠️ No se encontró RW para horizonte '{key}' en archivo: {file}")
        continue
    
    print(f"Procesando: {file}  (horizonte: {key}, h={h})  → RW: {rw_file}")
    
    try:
        df_model = prepare_df(load_csv_safe(os.path.join(PATH_DFGCN, file)))
        df_rw = prepare_df(load_csv_safe(os.path.join(PATH_RW, rw_file)))
        
        # Merge por fecha
        df = pd.merge(df_model, df_rw, on="date", suffixes=("_model", "_rw"))
        
        if len(df) < 10:
            print(f"⚠️ Muy pocos datos tras merge en {file}")
            continue
        
        dm_stat, p_value = dm_test(
            df["loss_model"].values,
            df["loss_rw"].values,
            h
        )
        
        results.append({
            "Modelo_DFGCN": file,
            "RW_correspondiente": rw_file,
            "Horizonte": key,
            "h": h,
            "DM_stat": round(dm_stat, 6),
            "p_value": round(p_value, 10),
            "significativo_5%": p_value < 0.05,
            "N_obs": len(df)
        })
        
    except Exception as e:
        print(f"❌ Error procesando {file}: {e}")

# =========================
# RESULTADOS FINALES
# =========================
if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["p_value", "Horizonte", "Modelo_DFGCN"])
    
    print("\n" + "="*120)
    print("RESULTADOS DIEBOLD-MARIANO TEST (DFGCN vs Random Walk)")
    print("="*120)
    print(df_results.to_string(index=False))
    print("="*120)
    
    # Guardar CSV
    output_path = "dm_results_completo.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Resultados guardados en: {output_path}")
    print(f"   Total de comparaciones: {len(df_results)}")
else:
    print("❌ No se generaron resultados. Revisa los mensajes de error arriba.")