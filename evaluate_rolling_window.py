import os
import sys
import subprocess
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import metric

TRAINING_YEARS = 10
DATA_PATH = "datos/tasa_cop_usd.csv"
RESULTS_CSV = "results_rolling_window.csv"
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


def run_prediction(model_id, input_csv, pred_len, des):
    cmd = [
        sys.executable, "predict_future.py",
        "--model_id", model_id,
        "--model", "DFGCN",
        "--data", "custom",
        "--data_path", input_csv,
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
        "--dropout", str(DROPOUT),
        "--activation", ACTIVATION,
        "--des", des,
        "--use_gpu", "False",
    ]
    subprocess.run(cmd, check=True)


def load_predictions(model_id):
    pred_file = os.path.join("resultados_futuro", f"{model_id}_future_predictions.csv")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    df = pd.read_csv(pred_file)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_metrics_pair(pred_vals, true_vals):
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


def cleanup_temp_files(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def main():
    full_df = load_full_data()

    os.makedirs("datos", exist_ok=True)
    os.makedirs("resultados_futuro", exist_ok=True)

    temp_files_created = []

    start_year = full_df["date"].dt.year.min() + TRAINING_YEARS
    end_year = 2025

    for t in range(start_year, end_year + 1):
        train_start = pd.Timestamp(f"{t - TRAINING_YEARS}-01-01")
        train_end = pd.Timestamp(f"{t - 1}-12-31")
        year_start = pd.Timestamp(f"{t}-01-01")
        year_end = pd.Timestamp(f"{t}-12-31")

        df_train = full_df[(full_df["date"] >= train_start) & (full_df["date"] <= train_end)].copy()
        df_year = full_df[(full_df["date"] >= year_start) & (full_df["date"] <= year_end)].copy()

        if len(df_train) < SEQ_LEN + 10:
            print(f"Skipping year {t}: not enough training data ({len(df_train)} rows).")
            continue

        if len(df_year) == 0:
            print(f"Skipping year {t}: no actual data for that year.")
            continue

        train_csv = f"datos/train_temp_{t}.csv"
        input_csv = f"datos/input_for_prediction_{t}.csv"
        save_temp_csv(df_train, train_csv)
        save_temp_csv(df_train, input_csv)
        temp_files_created.extend([train_csv, input_csv])

        for horizon_name, pred_len in HORIZONS.items():
            model_id = f"DFGCN_rolling_{t}_{pred_len}"
            des = f"rolling_{t}"

            print(f"\n=== Year {t} | Horizon {horizon_name} ({pred_len} days) ===")

            try:
                run_training(model_id, train_csv, pred_len, des)
            except subprocess.CalledProcessError as e:
                print(f"Training failed for year {t}, horizon {pred_len}: {e}")
                continue

            try:
                run_prediction(model_id, input_csv, pred_len, des)
            except subprocess.CalledProcessError as e:
                print(f"Prediction failed for year {t}, horizon {pred_len}: {e}")
                continue

            try:
                df_preds = load_predictions(model_id)
            except FileNotFoundError as e:
                print(f"Could not load predictions: {e}")
                continue

            pred_col = None
            for c in df_preds.columns:
                if c != "date":
                    pred_col = c
                    break
            if pred_col is None:
                print(f"No prediction column found for year {t}, horizon {pred_len}.")
                continue

            merged = pd.merge(df_preds[["date", pred_col]], df_year[["date", TARGET]], on="date", how="inner")

            if len(merged) == 0:
                n_to_use = min(pred_len, len(df_year), len(df_preds))
                pred_vals = df_preds[pred_col].values[:n_to_use]
                true_vals = df_year[TARGET].values[:n_to_use]
                if len(pred_vals) == 0 or len(true_vals) == 0:
                    print(f"No overlapping data for year {t}, horizon {pred_len}. Skipping.")
                    continue
                n = min(len(pred_vals), len(true_vals))
                pred_vals = pred_vals[:n]
                true_vals = true_vals[:n]
            else:
                pred_vals = merged[pred_col].values
                true_vals = merged[TARGET].values

            metrics = compute_metrics_pair(pred_vals, true_vals)

            row = {
                "year": t,
                "horizon_days": pred_len,
                **metrics,
            }
            append_result(row)
            print(f"  MAE_real={metrics['mae_real']:.4f} | RMSE_real={metrics['rmse_real']:.4f} | MAPE_real={metrics['mape_real']:.6f}")

    cleanup_temp_files(temp_files_created)
    print("\nRolling window evaluation completed. Results saved in results_rolling_window.csv")


if __name__ == "__main__":
    main()