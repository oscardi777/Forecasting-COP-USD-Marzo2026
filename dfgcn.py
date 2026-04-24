"""
dfgcn.py
===================================
Script de rolling window para prediccion de la tasa COP/USD usando DFGCN.
Reutiliza directamente los modulos del repositorio Forecasting-COP-USD.
Todo se ejecuta en memoria: no llama a subprocess o run.py

DISENO CLAVE — Periodos calendario reales:
  Las ventanas de entrenamiento y prediccion se definen por fechas exactas
  del calendario (inicio y fin de mes, ano, trimestre, semana), NO por un
  conteo fijo de dias habiles.

Uso: (guardar predicciones en csv)
    python dfgcn.py --data_path .\\datos\\tasas_BRL_COP_CHI_1993-2025.csv --results_dir ./resultados_rolling_ORates/ --save_predictions

    python dfgcn.py --data_path .\\datos\\data_rates_gold.csv --results_dir ./resultados_rolling_Rates_Gold/ --save_predictions

    python dfgcn.py --data_path .\\datos\\data_rates_bitcoin.csv --results_dir ./resultados_rolling_Rates_Bitcoin/ --save_predictions

Opcional:
    --root_path   Carpeta raiz del CSV (default: .)
    --data_path   Nombre o ruta completa del CSV
    --results_dir Carpeta donde guardar CSVs de resultados
                  (default: ./resultados_rolling/)
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
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
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.metrics import metric
from modelos.DFGCN import Model


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

# Minimo de periodos de entrenamiento recomendado segun granularidad
# (se puede bajar hasta MIN_PERIODOS_ABSOLUTO pero se advierte al usuario)
MIN_PERIODOS_RECOMENDADOS = {
    "1": 52,    # 1 semana → al menos 1 ano de historia
    "2": 26,    # 2 semanas → al menos 1 ano
    "3": 12,    # 1 mes → al menos 1 ano
    "4": 6,     # 2 meses → al menos 1 ano
    "5": 4,     # 3 meses → al menos 1 ano
    "6": 2,     # 6 meses → al menos 1 ano
    "7": 2,     # 1 ano → minimo 2 anos historia
    "8": 2,     # 2 anos → minimo 4 anos historia
}
MIN_PERIODOS_ABSOLUTO = 2  # nunca bajar de esto


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
# DATASET COMPATIBLE CON DFGCN
# ===========================================================================
class RollingDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, dates=None):
        self.data      = data.astype(np.float32)
        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len
        self.T         = len(data)
        self.tf        = self._time_feat(dates) if dates is not None \
                         else np.zeros((self.T, 4), dtype=np.float32)

    def _time_feat(self, dates):
        f = np.zeros((len(dates), 4), dtype=np.float32)
        f[:, 0] = (dates.month - 1) / 11.0 - 0.5
        f[:, 1] = (dates.day   - 1) / 30.0 - 0.5
        f[:, 2] = dates.dayofweek   / 6.0  - 0.5
        f[:, 3] = 0.0
        return f

    def __len__(self):
        return max(0, self.T - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        s_ini = idx
        s_fin = s_ini + self.seq_len
        r_ini = s_fin - self.label_len
        r_fin = r_ini + self.label_len + self.pred_len
        return (
            self.data[s_ini:s_fin],
            self.data[r_ini:r_fin],
            self.tf[s_ini:s_fin],
            self.tf[r_ini:r_fin],
        )


# ===========================================================================
# CONFIGURACION DE DFGCN
# ===========================================================================
def build_dfgcn_args(seq_len, label_len, pred_len, enc_in, features,
                     d_model, n_heads, e_layers, d_ff,
                     patch_len, k, dropout, activation, use_norm, batch_size):
    from types import SimpleNamespace

    pl = patch_len
    while pl > 1 and (seq_len < pl or (seq_len - pl) % pl != 0):
        pl = max(2, pl // 2)

    k_adj = min(k, max(1, enc_in - 1))

    return SimpleNamespace(
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=1,
        d_ff=d_ff, patch_len=pl, k=k_adj,
        dropout=dropout, activation=activation,
        use_norm=use_norm, output_attention=False,
        batch_size=batch_size, features=features,
    )


# ===========================================================================
# ENTRENAMIENTO
# ===========================================================================
def train_window(model, train_loader, val_loader, device,
                 pred_len, features, n_epochs, patience, lr):
    crit_mse = nn.MSELoss()
    crit_l1  = nn.L1Loss()
    opt      = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8)
    sched    = lr_scheduler.OneCycleLR(
        optimizer=opt, steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.3, epochs=n_epochs, max_lr=lr,
    )

    best_loss  = float("inf")
    best_state = None
    no_imp     = 0

    for _ in range(n_epochs):
        model.train()
        for bx, by, _, _ in train_loader:
            bx, by = bx.float().to(device), by.float().to(device)
            opt.zero_grad()
            out   = model(bx)
            f_dim = -1 if features == "MS" else 0
            out   = out[:, -pred_len:, f_dim:]
            tgt   = by[:, -pred_len:, f_dim:]
            loss  = crit_mse(out, tgt) + crit_l1(out, tgt)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for bx, by, _, _ in val_loader:
                bx, by = bx.float().to(device), by.float().to(device)
                out    = model(bx)
                f_dim  = -1 if features == "MS" else 0
                out    = out[:, -pred_len:, f_dim:]
                tgt    = by[:, -pred_len:, f_dim:]
                vl.append(crit_mse(out, tgt).item())

        val_loss = float(np.mean(vl))
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ===========================================================================
# PREDICCION
# ===========================================================================
def predict_window(model, input_sc, device, pred_len, features):
    model.eval()
    with torch.no_grad():
        x     = torch.tensor(input_sc, dtype=torch.float32).unsqueeze(0).to(device)
        out   = model(x)
        f_dim = -1 if features == "MS" else 0
        out   = out[:, -pred_len:, f_dim:]
        return out.squeeze(0).cpu().numpy()


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
    print("  ROLLING WINDOW FORECAST  ---  COP/USD  con  DFGCN")
    print("=" * 65)
    print(f"  Dataset: {f_min.date()} -> {f_max.date()}  (~{anios_datos:.1f} anos)")
    print()

    # 1. GRANULARIDAD
    print("Granularidad del periodo (horizonte = paso = 1 periodo completo):")
    print()
    for k, (nom, _) in GRANULARIDADES.items():
        print(f"  [{k}]  {nom}")
    print()

    clave = pedir_opcion(">>> Elige la granularidad [3]: ", list(GRANULARIDADES.keys()))
    etiq_gran, gran_delta = GRANULARIDADES[clave]
    print(f"  Granularidad: {etiq_gran}")

    # 2. NUMERO DE PERIODOS DE ENTRENAMIENTO
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
    print(f"Cuantos periodos completos de '{etiq_gran}' usar para entrenar?")
    print(f"  Minimo recomendado : {min_recomendado} periodos "
          f"(~{min_recomendado/ppa:.1f} anos de historia)")
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
            print(f"\n  ADVERTENCIA: {n_entren} periodos es menor al minimo recomendado")
            print(f"  ({min_recomendado} periodos). Con tan pocos datos el modelo puede")
            print(f"  no converger correctamente y los resultados seran poco confiables.")
            confirmar = pedir_opcion("  Continuar de todas formas? [s/n]: ", ["s", "n"])
            if confirmar == "s":
                break
            else:
                print()
                continue
        else:
            break

    n_ventanas_est = total_periodos - n_entren
    print(f"  Se ejecutaran aproximadamente {n_ventanas_est} ventana(s).")

    # 3. MODO: rolling vs expanding
    print()
    print("Modo de ventana:")
    print("  [1] Rolling   -> siempre los mismos N periodos, deslizandose.")
    print("  [2] Expanding -> desde el inicio del dataset, ventana creciente.")
    modo_r = pedir_opcion(">>> Modo [1]: ", ["1", "2"])
    modo   = "rolling" if modo_r == "1" else "expanding"
    print(f"  Modo: {modo}")

    # 4. HIPERPARAMETROS
    print()
    print("Usar hiperparametros por defecto?")
    print("  seq_len=48, label_len=30, d_model=128, n_heads=1,")
    print("  e_layers=1, d_ff=128, patch_len=8, epochs=20, patience=3")
    usar_def = input(">>> [S/n]: ").strip().lower() not in ("n", "no")

    if usar_def:
        hp = dict(seq_len=48, label_len=30, d_model=128, n_heads=1,
                  e_layers=1, d_ff=128, patch_len=8,
                  n_epochs=20, patience=3, lr=0.0001, batch_size=32)
    else:
        print("Ingresa valores (Enter = defecto):")
        def ask(msg, df, cast=int):
            r = input(f"  {msg} [{df}]: ").strip()
            return cast(r) if r else df
        hp = dict(
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

    return gran_delta, n_entren, modo, hp, etiq_gran, clave


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",   default=".")
    parser.add_argument("--data_path",   default="datos/tasa_cop_usd.csv")
    parser.add_argument("--results_dir", default="./resultados_rolling/")
    parser.add_argument("--save_predictions", action="store_true", default=False,
                        help="Guardar CSV con las predicciones reales de cada ventana")
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

    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError("El CSV necesita una columna 'date'.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    num_cols = [c for c in df.columns if c != date_col]
    N        = len(num_cols)
    features = "S" if N == 1 else "MS"

    print(f"  Variables: {N}  ->  features='{features}'")
    print(f"  Periodo  : {df[date_col].iloc[0].date()} -> {df[date_col].iloc[-1].date()}")

    data_values = df[num_cols].values.astype(np.float32)
    dates_all   = pd.DatetimeIndex(df[date_col])
    y_ini       = df[date_col].iloc[0].year
    y_fin       = df[date_col].iloc[-1].year

    # -------------------------------------------------------------------
    # 2. INTERFAZ INTERACTIVA
    # -------------------------------------------------------------------
    gran_delta, n_entren, modo, hp, etiq_gran, clave_gran = interfaz_interactiva(dates_all)

    seq_len    = hp["seq_len"]
    label_len  = hp["label_len"]
    n_epochs   = hp["n_epochs"]
    patience   = hp["patience"]
    lr         = hp["lr"]
    batch_size = hp["batch_size"]

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
    # 4. CUDA
    # -------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no disponible. Este script requiere GPU con CUDA.\n"
            "Verifica tu instalacion de PyTorch con soporte CUDA."
        )
    device = torch.device("cuda:0")
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------
    # 5. NOMBRES DE ARCHIVOS DE SALIDA
    # -------------------------------------------------------------------
    etiq_s  = etiq_gran.replace(" ", "_")
    modo_s  = "rol" if modo == "rolling" else "exp"

    out_csv = os.path.join(
        cli.results_dir,
        f"rolling_{y_ini}-{y_fin}_{etiq_s}_{n_entren}p_{modo_s}.csv",
    )

    pred_csv_path = os.path.join(
        cli.results_dir,
        f"predictions_{y_ini}-{y_fin}_{etiq_s}_{n_entren}p_{modo_s}.csv",
    )

    # Borrar archivo de predicciones anterior si existe para empezar limpio
    if cli.save_predictions and os.path.exists(pred_csv_path):
        os.remove(pred_csv_path)

    # -------------------------------------------------------------------
    # 6. LOOP PRINCIPAL
    # -------------------------------------------------------------------
    resultados   = []
    t_ini_global = time.time()

    print("\n" + "=" * 65)
    print(f"  Granularidad : {etiq_gran}")
    print(f"  Entrenamiento: {n_entren} periodo(s) por ventana")
    print(f"  Modo         : {modo}")
    print(f"  Ventanas     : {len(ventanas)}")
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

        if n_tr_obs < seq_len + label_len + 5:
            print(f"  Solo {n_tr_obs} obs en entrenamiento (minimo "
                  f"{seq_len + label_len + 5}). Saltando.")
            continue

        if n_pr_obs < 1:
            print("  Sin datos reales en el periodo de prediccion. Saltando.")
            continue

        # Escalado
        scaler = StandardScaler()
        data_tr_raw = data_values[idx_tr]
        scaler.fit(data_tr_raw)
        data_tr_sc  = scaler.transform(data_tr_raw)
        dates_tr    = dates_all[idx_tr]

        # Split train / val
        min_seg = seq_len + label_len + n_pr_obs
        n_val   = max(int(n_tr_obs * 0.15), min_seg)
        n_train = n_tr_obs - n_val

        if n_train < min_seg:
            print(f"  Sin espacio para split train/val. Validando sobre train.")
            ds_train = RollingDataset(data_tr_sc, seq_len, label_len, n_pr_obs, dates=dates_tr)
            ds_val   = None
        else:
            ds_train = RollingDataset(data_tr_sc[:n_train], seq_len, label_len, n_pr_obs,
                                      dates=dates_tr[:n_train])
            ds_val   = RollingDataset(data_tr_sc[n_train:], seq_len, label_len, n_pr_obs,
                                      dates=dates_tr[n_train:])

        if len(ds_train) < 1:
            print("  Dataset de entrenamiento sin muestras. Saltando.")
            continue

        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)

        if ds_val is not None and len(ds_val) >= 1:
            dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
        else:
            dl_val = DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=False)

        # Construir modelo
        args_m = build_dfgcn_args(
            seq_len=seq_len, label_len=label_len, pred_len=n_pr_obs,
            enc_in=N, features=features,
            d_model=hp["d_model"], n_heads=hp["n_heads"],
            e_layers=hp["e_layers"], d_ff=hp["d_ff"],
            patch_len=hp["patch_len"],
            k=min(2, max(1, N - 1)),
            dropout=0.1, activation="sigmoid", use_norm=1,
            batch_size=batch_size,
        )
        model = Model(args_m).float().to(device)

        t0 = time.time()
        model = train_window(
            model=model, train_loader=dl_train, val_loader=dl_val,
            device=device, pred_len=n_pr_obs, features=features,
            n_epochs=n_epochs, patience=patience, lr=lr,
        )
        print(f"  Entrenamiento: {time.time()-t0:.1f}s  "
              f"({n_pr_obs} dias habiles en el periodo predicho)")

        # Secuencia de entrada para la prediccion
        primer_idx_pr = idx_pr[0]
        if primer_idx_pr < seq_len:
            print(f"  No hay {seq_len} observaciones previas al periodo predicho. Saltando.")
            del model
            torch.cuda.empty_cache()
            continue

        input_raw = data_values[primer_idx_pr - seq_len: primer_idx_pr]
        input_sc  = scaler.transform(input_raw)

        pred_norm = predict_window(model, input_sc, device, n_pr_obs, features)
        if pred_norm.ndim == 1:
            pred_norm = pred_norm.reshape(-1, 1)

        true_raw   = data_values[idx_pr]
        true_norm  = scaler.transform(true_raw)

        true_norm_obj = true_norm[:, -1:]
        true_real_obj = true_raw[:,  -1:]

        dummy = np.zeros((n_pr_obs, N), dtype=np.float32)
        dummy[:, -1] = pred_norm[:, 0]
        pred_real_obj = scaler.inverse_transform(dummy)[:, -1:]

        # Metricas
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(pred_norm,     true_norm_obj)
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

        resultados.append({
            "periodo_predicho":   etiq,
            "fecha_pred_ini":     p_ini.strftime("%Y-%m-%d"),
            "fecha_pred_fin":     p_fin.strftime("%Y-%m-%d"),
            "dias_habiles_pred":  n_pr_obs,
            "fecha_train_ini":    t_ini_ef.strftime("%Y-%m-%d"),
            "fecha_train_fin":    t_fin.strftime("%Y-%m-%d"),
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

        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # 7. RESUMEN FINAL
    # -------------------------------------------------------------------
    t_total = time.time() - t_ini_global
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
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