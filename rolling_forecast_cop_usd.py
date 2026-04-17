"""
rolling_forecast_cop_usd.py
===========================
Script de rolling window para predicción de la tasa COP/USD usando DFGCN.
Reutiliza directamente los módulos del repositorio Forecasting-COP-USD.
No llama a subprocess ni a predict_future.py: todo se ejecuta en memoria.

Uso:
    python rolling_forecast_cop_usd.py --data_path datos/tasa_cop_usd.csv

Opcional:
    --root_path   Carpeta raíz del CSV (default: .)
    --data_path   Nombre o ruta completa del CSV
    --checkpoints Carpeta donde guardar checkpoints (default: ./checkpoints_rolling/)
    --results_dir Carpeta donde guardar CSVs de resultados (default: ./resultados_rolling/)
"""

import os
import sys
import time
import argparse
import warnings
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Asegurarse de que el directorio raíz del repositorio esté en el path
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Importaciones del repositorio
# ---------------------------------------------------------------------------
from utils.metrics import metric          # metric(pred, true) → mae,mse,rmse,mape,mspe,rse
from modelos.DFGCN import Model           # Modelo principal DFGCN
from modelos.RevIN import RevIN           # Normalización reversible
from layers.Embed import PositionalEmbedding  # Embedding posicional (usado por DFGCN)


# ===========================================================================
# DATASET INLINE (equivalente a Dataset_Custom pero sin leer CSV dos veces)
# ===========================================================================
class RollingDataset(Dataset):
    """
    Dataset que recibe un array numpy pre-procesado (ya escalado) y
    genera ventanas (seq_len, pred_len) deslizantes.
    Compatible con el formato esperado por DFGCN:
      seq_x  : (seq_len, N)
      seq_y  : (label_len + pred_len, N)
      x_mark : (seq_len, 4)   → características temporales simplificadas
      y_mark : (label_len + pred_len, 4)
    """

    def __init__(self, data, seq_len, label_len, pred_len, dates=None):
        """
        data     : np.ndarray shape (T, N) – datos ya escalados
        dates    : pd.DatetimeIndex o None
        """
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.T = data.shape[0]

        # Características temporales simples (mes, día, día semana, hora)
        if dates is not None:
            self.time_feat = self._build_time_feat(dates)
        else:
            self.time_feat = np.zeros((self.T, 4), dtype=np.float32)

    def _build_time_feat(self, dates):
        feat = np.zeros((len(dates), 4), dtype=np.float32)
        feat[:, 0] = (dates.month - 1) / 11.0 - 0.5
        feat[:, 1] = (dates.day - 1) / 30.0 - 0.5
        feat[:, 2] = dates.dayofweek / 6.0 - 0.5
        feat[:, 3] = 0.0  # hora = 0 para datos diarios
        return feat

    def __len__(self):
        return self.T - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.time_feat[s_begin:s_end]
        seq_y_mark = self.time_feat[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


# ===========================================================================
# CONSTRUCCIÓN DE ARGS COMPATIBLE CON DFGCN
# ===========================================================================
def build_dfgcn_args(seq_len, label_len, pred_len, enc_in, features,
                     d_model=128, n_heads=1, e_layers=1, d_ff=128,
                     patch_len=8, k=2, dropout=0.1, activation="sigmoid",
                     use_norm=1, batch_size=32):
    """
    Devuelve un objeto namespace con todos los parámetros que necesita
    DFGCN.Model.__init__. Se basa en los campos que aparecen en run.py
    y en modelos/DFGCN.py.
    """
    from types import SimpleNamespace

    # patch_len debe ser <= seq_len y (seq_len - patch_len) % patch_len == 0
    # Ajustamos automáticamente si es necesario.
    while patch_len > seq_len or (seq_len - patch_len) % patch_len != 0:
        patch_len = patch_len // 2
        if patch_len < 2:
            patch_len = 2
            break

    # k debe ser < enc_in (número de variables)
    k = min(k, max(1, enc_in - 1))

    args = SimpleNamespace(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=1,
        d_ff=d_ff,
        patch_len=patch_len,
        k=k,
        dropout=dropout,
        activation=activation,
        use_norm=use_norm,
        output_attention=False,
        batch_size=batch_size,
        features=features,
    )
    return args


# ===========================================================================
# ENTRENAMIENTO DE UNA VENTANA
# ===========================================================================
def train_window(model, train_loader, val_loader, device,
                 pred_len, features, n_epochs=20, patience=3,
                 learning_rate=0.0001, use_amp=False):
    """
    Entrena el modelo para una ventana de rolling.
    Devuelve el modelo con los mejores pesos según validación.
    """
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.98), eps=1e-8)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.3,
        epochs=n_epochs,
        max_lr=learning_rate
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            f_dim = -1 if features == "MS" else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

            loss = criterion_mse(outputs, batch_y) + criterion_l1(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Validación
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                outputs = model(batch_x)
                f_dim = -1 if features == "MS" else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y_cut = batch_y[:, -pred_len:, f_dim:].to(device)
                val_losses.append(criterion_mse(outputs, batch_y_cut).item())

        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ===========================================================================
# PREDICCIÓN CON UNA VENTANA
# ===========================================================================
def predict_window(model, input_seq, device, pred_len, features, enc_in):
    """
    Dado un array (seq_len, N) ya escalado, genera la predicción de pred_len pasos.
    Devuelve np.ndarray shape (pred_len, N) en escala normalizada.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, N)
        out = model(x)  # (1, pred_len, N) o (1, pred_len, 1)
        f_dim = -1 if features == "MS" else 0
        out = out[:, -pred_len:, f_dim:]  # (1, pred_len, ?)
        pred = out.squeeze(0).cpu().numpy()  # (pred_len, ?)
    return pred


# ===========================================================================
# INTERFAZ INTERACTIVA
# ===========================================================================
def pedir_entero(prompt, minimo=1, maximo=None):
    """Solicita un entero en consola con validación."""
    while True:
        try:
            val = int(input(prompt).strip())
            if val < minimo:
                print(f"  ✗ El valor debe ser ≥ {minimo}. Intenta de nuevo.")
                continue
            if maximo is not None and val > maximo:
                print(f"  ✗ El valor debe ser ≤ {maximo}. Intenta de nuevo.")
                continue
            return val
        except ValueError:
            print("  ✗ Por favor ingresa un número entero válido.")


def interfaz_interactiva(total_obs):
    """
    Guía al usuario por la configuración de la evaluación.
    Devuelve (horizon, step, años_entrenamiento).
    """
    DIAS_POR_ANIO = 252

    print("\n" + "=" * 65)
    print("  ROLLING WINDOW FORECAST – Tasa COP/USD con DFGCN")
    print("=" * 65)
    print(f"  Dataset cargado: {total_obs} observaciones (~{total_obs // DIAS_POR_ANIO} años)")
    print()

    # --- Horizonte ---
    print("Horizonte de predicción (días hábiles):")
    print("  5  = 1 semana  |  21  = 1 mes   |  63  = 3 meses")
    print("  126 = 6 meses  |  252 = 1 año")
    horizon = pedir_entero(">>> Ingresa el horizonte: ", minimo=1, maximo=total_obs // 3)

    # --- Step ---
    print(f"\nPaso de ventana (step) en días hábiles.")
    print(f"  Recomendado: 21 (mensual). El step controla cuánto avanza")
    print(f"  cada ventana. Menor step = más ventanas = más tiempo de cómputo.")
    step = pedir_entero(">>> Ingresa el step: ", minimo=1, maximo=total_obs // 2)

    # --- Años de entrenamiento ---
    minimo_obs_entren = 48 + 30 + horizon + 10  # seq_len + label_len + pred_len + margen
    min_anios = math.ceil(minimo_obs_entren / DIAS_POR_ANIO)
    max_anios = (total_obs - horizon) // DIAS_POR_ANIO - 1
    sugerido = max(min_anios, min(10, max_anios))

    print(f"\nAños de entrenamiento por ventana:")
    print(f"  Mínimo viable:  {min_anios} año(s)")
    print(f"  Máximo posible: {max_anios} año(s)")
    print(f"  Sugerido:       {sugerido} año(s) (balance cómputo/precisión)")

    while True:
        años = pedir_entero(
            f">>> Ingresa los años de entrenamiento [{sugerido}]: ",
            minimo=min_anios, maximo=max_anios
        )
        obs_entren = años * DIAS_POR_ANIO
        obs_necesarias = obs_entren + horizon
        if obs_necesarias > total_obs:
            print(f"  ✗ Con {años} años ({obs_entren} obs) + horizonte ({horizon}) = "
                  f"{obs_necesarias} obs, pero el dataset solo tiene {total_obs}.")
            continue
        # Verificar que quepan al menos 2 ventanas
        espacio_test = total_obs - obs_entren
        n_ventanas_est = max(0, (espacio_test - horizon) // step + 1)
        if n_ventanas_est < 2:
            print(f"  ✗ Con esa configuración solo habría {n_ventanas_est} ventana(s). "
                  f"Necesitas al menos 2. Reduce el step o los años de entrenamiento.")
            continue
        print(f"  ✓ Se ejecutarán aproximadamente {n_ventanas_est} ventana(s).")
        break

    # --- Hiperparámetros del modelo (opcional) ---
    print("\n¿Deseas usar los hiperparámetros por defecto del modelo DFGCN?")
    print("  Defaults: seq_len=48, label_len=30, d_model=128, n_heads=1,")
    print("            e_layers=1, d_ff=128, patch_len=8, epochs=20, patience=3")
    usar_default = input(">>> [S/n]: ").strip().lower()
    usar_default = usar_default not in ("n", "no")

    if usar_default:
        hparams = dict(seq_len=48, label_len=30, d_model=128, n_heads=1,
                       e_layers=1, d_ff=128, patch_len=8,
                       n_epochs=20, patience=3, lr=0.0001, batch_size=32)
    else:
        print("\nIngresa los hiperparámetros (Enter = usar valor por defecto):")
        def ask(msg, default, cast=int):
            r = input(f"  {msg} [{default}]: ").strip()
            return cast(r) if r else default

        hparams = dict(
            seq_len=ask("seq_len", 48),
            label_len=ask("label_len", 30),
            d_model=ask("d_model", 128),
            n_heads=ask("n_heads", 1),
            e_layers=ask("e_layers", 1),
            d_ff=ask("d_ff", 128),
            patch_len=ask("patch_len", 8),
            n_epochs=ask("n_epochs", 20),
            patience=ask("patience", 3),
            lr=ask("learning_rate", 0.0001, cast=float),
            batch_size=ask("batch_size", 32),
        )

    return horizon, step, años, hparams


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Rolling Window Forecast COP/USD – DFGCN")
    parser.add_argument("--root_path",   default=".",               help="Carpeta raíz")
    parser.add_argument("--data_path",   default="datos/tasa_cop_usd.csv", help="CSV de datos")
    parser.add_argument("--checkpoints", default="./checkpoints_rolling/", help="Dir checkpoints")
    parser.add_argument("--results_dir", default="./resultados_rolling/",  help="Dir resultados")
    cli_args = parser.parse_args()

    os.makedirs(cli_args.checkpoints, exist_ok=True)
    os.makedirs(cli_args.results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. CARGA DEL DATASET COMPLETO
    # -----------------------------------------------------------------------
    csv_full = os.path.join(cli_args.root_path, cli_args.data_path)
    if not os.path.exists(csv_full):
        csv_full = cli_args.data_path  # ruta absoluta o relativa directa

    print(f"\nCargando datos desde: {csv_full}")
    df_raw = pd.read_csv(csv_full)

    # Normalizar nombre de columna fecha
    date_col = [c for c in df_raw.columns if c.lower() == "date"]
    if not date_col:
        raise ValueError("El CSV debe tener una columna llamada 'date'.")
    date_col = date_col[0]
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    df_raw = df_raw.sort_values(date_col).reset_index(drop=True)

    # Renombrar la última columna a OT si no se llama así
    all_cols = list(df_raw.columns)
    numeric_cols = [c for c in all_cols if c != date_col]

    # Detectar univariado vs multivariado
    N = len(numeric_cols)  # número de variables
    if N == 1:
        features = "S"
    else:
        features = "MS"

    print(f"  Variables detectadas: {N}  →  features='{features}'")
    print(f"  Periodo: {df_raw[date_col].iloc[0].date()} → {df_raw[date_col].iloc[-1].date()}")

    # Array de datos numéricos (T, N)
    data_values = df_raw[numeric_cols].values.astype(np.float32)
    dates_all   = pd.DatetimeIndex(df_raw[date_col])
    T_total     = len(data_values)

    # Año de inicio y fin para el nombre del CSV de salida
    year_start_data = df_raw[date_col].iloc[0].year
    year_end_data   = df_raw[date_col].iloc[-1].year

    # -----------------------------------------------------------------------
    # 2. INTERFAZ INTERACTIVA
    # -----------------------------------------------------------------------
    horizon, step, años_entren, hparams = interfaz_interactiva(T_total)

    seq_len    = hparams["seq_len"]
    label_len  = hparams["label_len"]
    d_model    = hparams["d_model"]
    n_heads    = hparams["n_heads"]
    e_layers   = hparams["e_layers"]
    d_ff       = hparams["d_ff"]
    patch_len  = hparams["patch_len"]
    n_epochs   = hparams["n_epochs"]
    patience   = hparams["patience"]
    lr         = hparams["lr"]
    batch_size = hparams["batch_size"]

    obs_por_anio = 252
    obs_entren   = años_entren * obs_por_anio

    # -----------------------------------------------------------------------
    # 3. DEVICE – forzar CUDA
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Este script requiere GPU.\n"
            "Verifica tu instalación de PyTorch con soporte CUDA."
        )
    device = torch.device("cuda:0")
    print(f"\nUsando dispositivo: {device}  ({torch.cuda.get_device_name(0)})")

    # -----------------------------------------------------------------------
    # 4. NOMBRE DEL CSV DE SALIDA
    # -----------------------------------------------------------------------
    out_csv_name = (
        f"rolling_window_{year_start_data}-{year_end_data}"
        f"_H{horizon}_S{step}_Y{años_entren}.csv"
    )
    out_csv_path = os.path.join(cli_args.results_dir, out_csv_name)

    # -----------------------------------------------------------------------
    # 5. ROLLING WINDOW LOOP
    # -----------------------------------------------------------------------
    resultados = []
    tiempo_inicio_total = time.time()

    # Índice del primer inicio posible: necesitamos obs_entren + seq_len + pred_len
    primer_inicio = 0
    ventana_idx   = 0

    print("\n" + "=" * 65)
    print(f"  Iniciando Rolling Window")
    print(f"  Horizonte={horizon}d | Step={step}d | Entrenamiento={años_entren} años ({obs_entren} obs)")
    print("=" * 65)

    while True:
        # Índices de entrenamiento
        train_ini = primer_inicio
        train_fin = primer_inicio + obs_entren  # exclusive

        # Índices de predicción (test)
        test_ini  = train_fin
        test_fin  = test_ini + horizon  # exclusive

        # Verificar que hay datos suficientes
        if test_fin > T_total:
            print(f"\nNo hay más datos para continuar. Última ventana procesada: {ventana_idx}.")
            break

        ventana_idx += 1
        fecha_inicio_pred = dates_all[test_ini]
        periodo_label = fecha_inicio_pred.strftime("%Y-%m")

        print(f"\n[Ventana {ventana_idx}] Entrenamiento: {dates_all[train_ini].date()} → "
              f"{dates_all[train_fin - 1].date()}   |   "
              f"Predicción: {dates_all[test_ini].date()} → {dates_all[min(test_fin, T_total) - 1].date()}")

        # --- Escalar con los datos de entrenamiento ---
        scaler = StandardScaler()
        data_train_raw = data_values[train_ini:train_fin]
        scaler.fit(data_train_raw)

        data_all_scaled = scaler.transform(data_values)  # (T, N) escalado

        # --- Construir datasets ---
        # Entrenamiento: solo los índices de la ventana de entrenamiento
        data_train_scaled = data_all_scaled[train_ini:train_fin]
        dates_train       = dates_all[train_ini:train_fin]

        # Mínimo de observaciones que necesita el dataset de validación para
        # poder generar AL MENOS 1 muestra:  seq_len + label_len + horizon
        min_obs_val = seq_len + label_len + horizon

        # Intentamos reservar un 15% para validación, pero garantizamos el mínimo
        n_val_raw = int(len(data_train_scaled) * 0.15)
        n_val = max(n_val_raw, min_obs_val)

        # Si reservar ese mínimo deja al train sin datos suficientes,
        # desactivamos la validación separada y usamos todo para entrenar
        n_tr = len(data_train_scaled) - n_val
        min_obs_train = seq_len + label_len + horizon  # mínimo para 1 muestra de train

        if n_tr < min_obs_train:
            # No hay suficiente espacio para split → usamos todo para entrenar
            # y la validación se hace sobre los mismos datos (early stopping aproximado)
            print(f"  ⚠ Split train/val omitido: datos insuficientes ({len(data_train_scaled)} obs). "
                  f"Validando sobre train completo.")
            train_ds = RollingDataset(data_train_scaled, seq_len, label_len, horizon,
                                      dates=dates_train)
            val_ds   = None
        else:
            train_ds = RollingDataset(data_train_scaled[:n_tr], seq_len, label_len, horizon,
                                      dates=dates_train[:n_tr])
            val_ds   = RollingDataset(data_train_scaled[n_tr:], seq_len, label_len, horizon,
                                      dates=dates_train[n_tr:])

        if train_ds is None or len(train_ds) < 1:
            print(f"  ✗ Ventana {ventana_idx}: datos insuficientes para crear batches. Saltando.")
            primer_inicio += step
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        # Si val_ds existe y tiene al menos 1 muestra usarla, si no usar train
        if val_ds is not None and len(val_ds) >= 1:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        else:
            val_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # --- Construir modelo ---
        enc_in = N  # número de variables de entrada
        args_model = build_dfgcn_args(
            seq_len=seq_len, label_len=label_len, pred_len=horizon,
            enc_in=enc_in, features=features,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers,
            d_ff=d_ff, patch_len=patch_len,
            k=min(2, max(1, enc_in - 1)),
            dropout=0.1, activation="sigmoid",
            use_norm=1, batch_size=batch_size
        )

        model = Model(args_model).float().to(device)

        # --- Entrenamiento ---
        t0 = time.time()
        model = train_window(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            pred_len=horizon,
            features=features,
            n_epochs=n_epochs,
            patience=patience,
            learning_rate=lr,
        )
        t_train = time.time() - t0
        print(f"  Entrenamiento: {t_train:.1f}s")

        # --- Predicción ---
        # Tomamos las últimas seq_len observaciones antes del inicio del test
        input_seq = data_all_scaled[test_ini - seq_len: test_ini]  # (seq_len, N)

        pred_norm = predict_window(
            model=model, input_seq=input_seq,
            device=device, pred_len=horizon,
            features=features, enc_in=enc_in
        )  # (horizon, 1) si MS/S → última variable

        # Valores reales en escala normalizada
        true_raw_window = data_values[test_ini:test_fin]  # (horizon, N)
        true_norm_all   = scaler.transform(true_raw_window)  # (horizon, N)

        # En modo S o MS solo la última columna es objetivo
        if features in ("S", "MS"):
            true_norm = true_norm_all[:, -1:]   # (horizon, 1)
            true_real = true_raw_window[:, -1:] # (horizon, 1)
        else:
            true_norm = true_norm_all           # (horizon, N)
            true_real = true_raw_window

        # Si pred_norm tiene shape (horizon,) la expandimos
        if pred_norm.ndim == 1:
            pred_norm = pred_norm.reshape(-1, 1)

        # Desnormalizar predicciones para escala real
        # pred_norm es solo la columna objetivo → reconstruir para inverse_transform
        if features in ("S", "MS") and pred_norm.shape[1] == 1:
            # Construir un array dummy con N columnas para inverse_transform
            dummy = np.zeros((len(pred_norm), N), dtype=np.float32)
            dummy[:, -1] = pred_norm[:, 0]
            pred_real_full = scaler.inverse_transform(dummy)
            pred_real = pred_real_full[:, -1:]  # (horizon, 1)
        else:
            pred_real = scaler.inverse_transform(pred_norm)

        # --- Métricas normalizadas ---
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(pred_norm, true_norm)

        # --- Métricas en escala real (pesos COP) ---
        mae_r, mse_r, rmse_r, mape_r, mspe_r, rse_r = metric(pred_real, true_real)

        print(f"  MAE_pesos={mae_r:.2f} | RMSE_pesos={rmse_r:.2f} | "
              f"MAPE={mape_r * 100:.2f}%  |  MAE_norm={mae_n:.4f}")

        resultados.append({
            "inicio_periodo_predicho": periodo_label,
            "mae_normalizado":         round(float(mae_n),  6),
            "mse_normalizado":         round(float(mse_n),  6),
            "mape_normalizado":        round(float(mape_n), 6),
            "rmse_normalizado":        round(float(rmse_n), 6),
            "mae_escala_pesos":        round(float(mae_r),  4),
            "mse_escala_pesos":        round(float(mse_r),  4),
            "mape_escala_pesos":       round(float(mape_r), 6),
            "rmse_escala_pesos":       round(float(rmse_r), 4),
        })

        # Guardar incrementalmente (por si se interrumpe)
        df_res = pd.DataFrame(resultados)
        df_res.to_csv(out_csv_path, index=False)

        # Liberar GPU
        del model
        torch.cuda.empty_cache()

        # Avanzar ventana
        primer_inicio += step

    # -----------------------------------------------------------------------
    # 6. RESUMEN FINAL
    # -----------------------------------------------------------------------
    t_total = time.time() - tiempo_inicio_total
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)

    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res.to_csv(out_csv_path, index=False)

        cols_mostrar = [
            ("mae_escala_pesos",  "MAE (pesos COP)"),
            ("rmse_escala_pesos", "RMSE (pesos COP)"),
            ("mape_escala_pesos", "MAPE"),
            ("mae_normalizado",   "MAE normalizado"),
            ("mse_normalizado",   "MSE normalizado"),
        ]
        for col, nombre in cols_mostrar:
            vals = df_res[col].astype(float)
            if "mape" in col:
                print(f"  {nombre:25s}: media={vals.mean()*100:.2f}%  std={vals.std()*100:.2f}%")
            else:
                print(f"  {nombre:25s}: media={vals.mean():.4f}  std={vals.std():.4f}")

        print(f"\n  Ventanas ejecutadas : {len(resultados)}")
        print(f"  Tiempo total        : {t_total / 60:.1f} min ({t_total:.0f} s)")
        print(f"  Resultados guardados: {out_csv_path}")
    else:
        print("  No se completó ninguna ventana. Revisa los parámetros.")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()