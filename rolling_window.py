import os
import sys
import subprocess
import pandas as pd
import numpy as np
import shutil
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import metric

TRAINING_YEARS = 10

DATA_PATH = "datos/tasa_cop_usd_1993-2025.csv"
RESULTS_CSV = "results_rolling_window_2005-2010.csv"
HORIZONS = {
    "short": 30,
    "medium": 128,
    "long": 252,
}
SEQ_LEN = 48
LABEL_LEN = 30
D_MODEL = 128
N_HEADS = 1
E_LAYERS = 1
D_LAYERS = 1
D_FF = 128
FACTOR = 3
EMBED = "timeF"
USE_NORM = 1
K = 1
TRAIN_EPOCHS = 30
PATIENCE = 5
FEATURES = "S"
TARGET = "OT"
FREQ = "b"
ACTIVATION = "sigmoid"
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
LRADJ = "type7"
PCT_START = 0.3
NUM_WORKERS = 0


def load_full_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "COP_USD" in df.columns and TARGET not in df.columns:
        df = df.rename(columns={"COP_USD": TARGET})
    return df


def save_temp_csv(df_slice, filepath):
    out = df_slice[["date", TARGET]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(filepath, index=False)


def run_training(model_id, train_csv, pred_len, des):
    cmd = [
        sys.executable, "run.py",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "DFGCN",
        "--data", "custom",
        "--root_path", ".",
        "--data_path", train_csv,
        "--features", FEATURES,
        "--target", TARGET,
        "--freq", FREQ,
        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(pred_len),
        "--enc_in", "1",
        "--dec_in", "1",
        "--c_out", "1",
        "--d_model", str(D_MODEL),
        "--n_heads", str(N_HEADS),
        "--e_layers", str(E_LAYERS),
        "--d_layers", str(D_LAYERS),
        "--d_ff", str(D_FF),
        "--factor", str(FACTOR),
        "--embed", EMBED,
        "--use_norm", str(USE_NORM),
        "--k", str(K),
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", str(LEARNING_RATE),
        "--lradj", LRADJ,
        "--pct_start", str(PCT_START),
        "--dropout", str(DROPOUT),
        "--activation", ACTIVATION,
        "--num_workers", str(NUM_WORKERS),
        "--des", des,
        "--use_gpu", "False",
    ]
    subprocess.run(cmd, check=True)


def extract_metrics_from_test(setting):
    """
    NUEVO: Extrae métricas del archivo de resultados que genera run.py
    en lugar de usar predict_future.py.
    
    run.py crea: ./test_results/{setting}/predictions_vs_actuals.csv
    Ese archivo tiene columnas: sample_idx, step, true_value_normalized, 
                                pred_value_normalized, true_value_real, pred_value_real
    """
    results_dir = os.path.join("test_results", setting)
    pred_file = os.path.join(results_dir, "predictions_vs_actuals.csv")
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Test results not found: {pred_file}")
    
    df_results = pd.read_csv(pred_file)
    return df_results


def compute_metrics_from_df(df_results):
    """
    NUEVO: Calcula métricas desde el DataFrame de resultados.
    Usa valores reales (no normalizados) para comparación.
    """
    pred_vals = df_results['pred_value_real'].values
    true_vals = df_results['true_value_real'].values
    
    pred_np = np.array(pred_vals, dtype=float).reshape(-1, 1)
    true_np = np.array(true_vals, dtype=float).reshape(-1, 1)
    mae_r, mse_r, rmse_r, mape_r, mspe_r, rse_r = metric(pred_np, true_np)
    
    mean_true = true_np.mean()
    std_true = true_np.std() if true_np.std() > 0 else 1.0
    pred_norm = (pred_np - mean_true) / std_true
    true_norm = (true_np - mean_true) / std_true
    mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(pred_norm, true_norm)
    
    return {
        "mae_norm": mae_n,
        "mse_norm": mse_n,
        "rmse_norm": rmse_n,
        "mape_norm": mape_n,
        "mspe_norm": mspe_n,
        "rse_norm": rse_n,
        "mae_real": mae_r,
        "mse_real": mse_r,
        "rmse_real": rmse_r,
        "mape_real": mape_r,
        "mspe_real": mspe_r,
        "rse_real": rse_r,
    }


def append_result(row_dict):
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(RESULTS_CSV):
        df_row.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(RESULTS_CSV, mode="w", header=True, index=False)


def cleanup_model_outputs(model_id):
    paths = [
        os.path.join("checkpoints", model_id),
        os.path.join("results", model_id),
        os.path.join("test_results", model_id),
    ]
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)


def cleanup_temp_files(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def main():
    full_df = load_full_data()

    os.makedirs("datos", exist_ok=True)

    temp_files_created = []

    start_year = 2000
    end_year = 2005

    for t in range(start_year, end_year + 1):
        # ============================================
        # CAMBIO 1: Crear CSV que INCLUYE el año a predecir
        # ============================================
        train_start = pd.Timestamp(f"{t - TRAINING_YEARS}-01-01")
        # CAMBIO: train_end ahora es fin del año t (no t-1)
        train_end = pd.Timestamp(f"{t}-12-31")
        
        # Cargar datos que incluyen 10 años + el año actual
        df_combined = full_df[(full_df["date"] >= train_start) & (full_df["date"] <= train_end)].copy()

        if len(df_combined) < SEQ_LEN + 10:
            print(f"Skipping year {t}: not enough data ({len(df_combined)} rows).")
            continue

        # CSV único con toda la información
        combined_csv = f"datos/combined_temp_{t}.csv"
        save_temp_csv(df_combined, combined_csv)
        temp_files_created.append(combined_csv)

        for horizon_name, pred_len in HORIZONS.items():
            model_id = f"DFGCN_rolling_{t}_{pred_len}"
            des = f"rolling_{t}"

            print(f"\n=== Year {t} | Horizon {horizon_name} ({pred_len} days) ===")

            try:
                # ============================================
                # CAMBIO 2: Entrenar con datos que incluyen año t
                # ============================================
                # run.py automáticamente dividirá:
                # - 70% para entrenamiento (años t-10 a aproximadamente t-3)
                # - 10% para validación (años t-3 a t-1 aprox)
                # - 20% para test (año t)
                run_training(model_id, combined_csv, pred_len, des)
                
            except subprocess.CalledProcessError as e:
                print(f"Training failed for year {t}, horizon {pred_len}: {e}")
                cleanup_model_outputs(model_id)
                continue

            try:
                # ============================================
                # CAMBIO 3: Extraer métricas de test_results
                # ============================================
                # NO usamos predict_future.py
                # run.py ya hizo el test y guardó resultados
                df_results = extract_metrics_from_test(model_id)
                
            except FileNotFoundError as e:
                print(f"Could not load test results: {e}")
                cleanup_model_outputs(model_id)
                continue

            # ============================================
            # CAMBIO 4: Calcular métricas desde resultados
            # ============================================
            metrics = compute_metrics_from_df(df_results)

            append_result({
                "year": t,
                "horizon_days": pred_len,
                **metrics,
            })
            
            print(f"  MAE_real={metrics['mae_real']:.4f} | RMSE_real={metrics['rmse_real']:.4f} | MAPE_real={metrics['mape_real']:.6f}")

            cleanup_model_outputs(model_id)

    cleanup_temp_files(temp_files_created)
    print("\nRolling window evaluation completed. Results saved in results_rolling_window.csv")


if __name__ == "__main__":
    main()