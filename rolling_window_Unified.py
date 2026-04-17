import os, sys, time, argparse, warnings, math
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from types import SimpleNamespace
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from utils.metrics import metric
from modelos.DFGCN import Model

class RollingDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, dates=None):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.T = data.shape[0]
        self.time_feat = self._build_time_feat(dates) if dates is not None else np.zeros((self.T, 4), dtype=np.float32)

    def _build_time_feat(self, dates):
        feat = np.zeros((len(dates), 4), dtype=np.float32)
        feat[:, 0] = (dates.month - 1) / 11.0 - 0.5
        feat[:, 1] = (dates.day - 1) / 30.0 - 0.5
        feat[:, 2] = dates.dayofweek / 6.0 - 0.5
        feat[:, 3] = 0.0
        return feat

    def __len__(self):
        return max(0, self.T - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return self.data[s_begin:s_end], self.data[r_begin:r_end], self.time_feat[s_begin:s_end], self.time_feat[r_begin:r_end]

def build_dfgcn_args(seq_len, label_len, pred_len, enc_in, features, d_model=128, n_heads=1, e_layers=1, d_ff=128, patch_len=8, k=2, dropout=0.1, activation="sigmoid", use_norm=1, batch_size=32):
    pl = patch_len
    while pl > 1 and (seq_len < pl or (seq_len - pl) % pl != 0):
        pl = max(2, pl // 2)
    k = min(k, max(1, enc_in - 1))
    return SimpleNamespace(seq_len=seq_len, label_len=label_len, pred_len=pred_len, enc_in=enc_in, dec_in=enc_in, c_out=enc_in, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=1, d_ff=d_ff, patch_len=pl, k=k, dropout=dropout, activation=activation, use_norm=use_norm, output_attention=False, batch_size=batch_size, features=features)

def train_window(model, train_loader, val_loader, device, pred_len, features, n_epochs=20, patience=3, learning_rate=0.0001):
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-8)
    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, steps_per_epoch=max(1, len(train_loader)), pct_start=0.3, epochs=n_epochs, max_lr=learning_rate)
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
            scheduler.step()
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

def predict_window(model, input_seq, device, pred_len, features, enc_in):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        out = model(x)
        f_dim = -1 if features == "MS" else 0
        out = out[:, -pred_len:, f_dim:]
        pred = out.squeeze(0).cpu().numpy()
    return pred

def pedir_entero(prompt, minimo=1, maximo=None):
    while True:
        try:
            val = int(input(prompt).strip())
            if val < minimo:
                print(f"  Valor debe ser >= {minimo}")
                continue
            if maximo is not None and val > maximo:
                print(f"  Valor debe ser <= {maximo}")
                continue
            return val
        except ValueError:
            print("  Ingresa un entero valido")

def pedir_opcion(prompt, validas):
    while True:
        r = input(prompt).strip()
        if r in validas:
            return r
        print(f"  Opciones validas: {', '.join(validas)}")

GRANULARIDADES = {"1": ("1 semana", relativedelta(weeks=1)), "2": ("2 semanas", relativedelta(weeks=2)), "3": ("1 mes", relativedelta(months=1)), "4": ("2 meses", relativedelta(months=2)), "5": ("3 meses", relativedelta(months=3)), "6": ("6 meses", relativedelta(months=6)), "7": ("1 anio", relativedelta(years=1)), "8": ("2 anios", relativedelta(years=2))}

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
        n_obs_pred = ((dates_all >= pred_ini) & (dates_all <= pred_fin)).sum()
        if n_obs_train < 10 or n_obs_pred < 1:
            ptr = ptr + gran_delta
            continue
        train_ini_efectivo = inicio_base if modo == "expanding" else train_ini_rolling
        ventanas.append({"train_ini_efectivo": train_ini_efectivo, "train_fin": train_fin, "pred_ini": pred_ini, "pred_fin": pred_fin, "etiqueta": etiqueta_periodo(pred_ini, gran_delta)})
        ptr = ptr + gran_delta
    return ventanas

def generar_ventanas_dias_habiles(dates_all, horizon_days, step_days, anos_entren):
    ventanas = []
    dias_por_anio = 252
    obs_entren = anos_entren * dias_por_anio
    T_total = len(dates_all)
    primer_inicio = 0
    while True:
        train_ini = primer_inicio
        train_fin = primer_inicio + obs_entren
        test_ini = train_fin
        test_fin = test_ini + horizon_days
        if test_fin > T_total:
            break
        fecha_inicio_pred = dates_all[test_ini]
        periodo_label = fecha_inicio_pred.strftime("%Y-%m")
        ventanas.append({"idx_train_ini": train_ini, "idx_train_fin": train_fin, "idx_test_ini": test_ini, "idx_test_fin": test_fin, "fecha_train_ini": dates_all[train_ini], "fecha_train_fin": dates_all[train_fin - 1], "fecha_test_ini": dates_all[test_ini], "fecha_test_fin": dates_all[min(test_fin - 1, T_total - 1)], "periodo_label": periodo_label})
        primer_inicio += step_days
    return ventanas

def interfaz_modo_seleccion():
    print("\n" + "=" * 65)
    print("  ROLLING WINDOW FORECAST  -  COP/USD  con  DFGCN")
    print("=" * 65)
    print() 
    print("Selecciona el modo de rolling window:")
    print() 
    print("  [1] Periodos calendario (mes, trimestre, anio, etc.)")
    print("      Las ventanas se definen por fechas exactas del calendario")
    print("      Horizonte = Paso = 1 periodo completo")
    print("      Util para analisis estacional y analisis por periodo")
    print() 
    print("  [2] Dias habiles fijos")
    print("      Las ventanas se definen por numero de dias habiles")
    print("      Horizonte y paso definibles independientemente")
    print("      Util para analisis de corto/mediano/largo plazo")
    print() 
    modo = pedir_opcion(">>> Elige el modo [1]: ", ["1", "2"])
    return "calendario" if modo == "1" else "dias_habiles"

def interfaz_calendario(dates_all):
    f_min = pd.Timestamp(dates_all.min())
    f_max = pd.Timestamp(dates_all.max())
    anios_datos = (f_max - f_min).days / 365.25
    print(f"\n  Dataset: {f_min.date()} -> {f_max.date()}  (~{anios_datos:.1f} anios)")
    print() 
    print("Granularidad del periodo:")
    print() 
    for k, (nom, _) in GRANULARIDADES.items():
        print(f"  [{k}]  {nom}")
    print() 
    clave = pedir_opcion(">>> Elige la granularidad [3]: ", list(GRANULARIDADES.keys()))
    etiq_gran, gran_delta = GRANULARIDADES[clave]
    print(f"  Granularidad: {etiq_gran}")
    if gran_delta.years:
        ppa = 1.0 / gran_delta.years
    elif gran_delta.months:
        ppa = 12.0 / gran_delta.months
    else:
        ppa = 52.0 / (gran_delta.weeks or 1)
    total_periodos = int(anios_datos * ppa)
    min_entren = 2
    max_entren = max(2, total_periodos - 2)
    print() 
    print(f"Cuantos periodos completos de '{etiq_gran}' para entrenar?")
    n_entren = pedir_entero(f">>> Numero de periodos de entrenamiento [{max(2, min(10, max_entren))}]: ", minimo=min_entren, maximo=max_entren)
    print() 
    print("Modo de ventana:")
    print("  [1] Rolling   -> siempre los mismos N periodos, deslizandose")
    print("  [2] Expanding -> ventana creciente desde el inicio")
    modo_r = pedir_opcion(">>> Modo [1]: ", ["1", "2"])
    modo = "rolling" if modo_r == "1" else "expanding"
    print(f"  Modo: {modo}")
    return gran_delta, n_entren, modo, etiq_gran

def interfaz_dias_habiles(total_obs):
    dias_por_anio = 252
    print(f"\n  Dataset cargado: {total_obs} observaciones (~{total_obs // dias_por_anio} anios)")
    print() 
    print("Horizonte de prediccion (dias habiles):")
    print("  5  = 1 semana  |  21  = 1 mes   |  63  = 3 meses")
    print("  126 = 6 meses  |  252 = 1 anio")
    horizon = pedir_entero(">>> Horizonte: ", minimo=1, maximo=total_obs // 3)
    print(f"\nPaso de ventana (step) en dias habiles.")
    print(f"  Recomendado: 21 (mensual)")
    step = pedir_entero(">>> Step: ", minimo=1, maximo=total_obs // 2)
    minimo_obs_entren = 48 + 30 + horizon + 10
    min_anios = math.ceil(minimo_obs_entren / dias_por_anio)
    max_anios = (total_obs - horizon) // dias_por_anio - 1
    sugerido = max(min_anios, min(10, max_anios))
    print(f"\nAnios de entrenamiento por ventana:")
    print(f"  Minimo viable:  {min_anios} anio(s)")
    print(f"  Maximo posible: {max_anios} anio(s)")
    print(f"  Sugerido:       {sugerido} anio(s)")
    while True:
        anos = pedir_entero(f">>> Anios de entrenamiento [{sugerido}]: ", minimo=min_anios, maximo=max_anios)
        obs_entren = anos * dias_por_anio
        obs_necesarias = obs_entren + horizon
        if obs_necesarias > total_obs:
            print(f"  Datos insuficientes ({obs_necesarias} > {total_obs}). Intenta de nuevo.")
            continue
        espacio_test = total_obs - obs_entren
        n_ventanas_est = max(0, (espacio_test - horizon) // step + 1)
        if n_ventanas_est < 2:
            print(f"  Solo {n_ventanas_est} ventana(s). Reduce el step o los anios.")
            continue
        print(f"  Se ejecutaran aproximadamente {n_ventanas_est} ventana(s).")
        break
    return horizon, step, anos

def interfaz_hiperparametros():
    print("\nUsar hiperparametros por defecto?")
    print("  seq_len=48, label_len=30, d_model=128, n_heads=1,")
    print("  e_layers=1, d_ff=128, patch_len=8, epochs=20, patience=3")
    usar_default = input(">>> [S/n]: ").strip().lower() not in ("n", "no")
    if usar_default:
        return dict(seq_len=48, label_len=30, d_model=128, n_heads=1, e_layers=1, d_ff=128, patch_len=8, n_epochs=20, patience=3, lr=0.0001, batch_size=32)
    else:
        print("\nIngresa los hiperparametros (Enter = usar valor por defecto):")
    def ask(msg, default, cast=int):
        r = input(f"  {msg} [{default}]: ").strip()
        return cast(r) if r else default
    return dict(seq_len=ask("seq_len", 48), label_len=ask("label_len", 30), d_model=ask("d_model", 128), n_heads=ask("n_heads", 1), e_layers=ask("e_layers", 1), d_ff=ask("d_ff", 128), patch_len=ask("patch_len", 8), n_epochs=ask("n_epochs", 20), patience=ask("patience", 3), lr=ask("learning_rate", 0.0001, cast=float), batch_size=ask("batch_size", 32))

def main():
    parser = argparse.ArgumentParser(description="Rolling Window Forecast COP/USD - DFGCN Unificado")
    parser.add_argument("--root_path", default=".", help="Carpeta raiz")
    parser.add_argument("--data_path", default="datos/tasa_cop_usd.csv", help="CSV de datos")
    parser.add_argument("--results_dir", default="./resultados_rolling/", help="Dir resultados")
    cli_args = parser.parse_args()
    os.makedirs(cli_args.results_dir, exist_ok=True)
    csv_full = os.path.join(cli_args.root_path, cli_args.data_path)
    if not os.path.exists(csv_full):
        csv_full = cli_args.data_path
    print(f"\nCargando datos desde: {csv_full}")
    df_raw = pd.read_csv(csv_full)
    date_col = next((c for c in df_raw.columns if c.lower() == "date"), None)
    if not date_col:
        raise ValueError("El CSV debe tener una columna llamada 'date'.")
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    df_raw = df_raw.sort_values(date_col).reset_index(drop=True)
    all_cols = list(df_raw.columns)
    numeric_cols = [c for c in all_cols if c != date_col]
    N = len(numeric_cols)
    features = "S" if N == 1 else "MS"
    print(f"  Variables detectadas: {N}  ->  features='{features}'")
    print(f"  Periodo: {df_raw[date_col].iloc[0].date()} -> {df_raw[date_col].iloc[-1].date()}")
    data_values = df_raw[numeric_cols].values.astype(np.float32)
    dates_all = pd.DatetimeIndex(df_raw[date_col])
    T_total = len(data_values)
    year_start_data = df_raw[date_col].iloc[0].year
    year_end_data = df_raw[date_col].iloc[-1].year
    modo_seleccionado = interfaz_modo_seleccion()
    if modo_seleccionado == "calendario":
        gran_delta, n_entren, modo_ventana, etiq_gran = interfaz_calendario(dates_all)
        ventanas = generar_ventanas_calendario(dates_all, n_entren, gran_delta, modo_ventana)
        out_csv_name = (f"rolling_{year_start_data}-{year_end_data}_{etiq_gran.replace(' ', '_')}" 
                      f"_{n_entren}p_{modo_ventana}.csv")
    else:
        horizon, step, anos_entren = interfaz_dias_habiles(T_total)
        ventanas = generar_ventanas_dias_habiles(dates_all, horizon, step, anos_entren)
        out_csv_name = (f"rolling_{year_start_data}-{year_end_data}_H{horizon}_S{step}_Y{anos_entren}.csv")
    hparams = interfaz_hiperparametros()
    seq_len = hparams["seq_len"]
    label_len = hparams["label_len"]
    d_model = hparams["d_model"]
    n_heads = hparams["n_heads"]
    e_layers = hparams["e_layers"]
    d_ff = hparams["d_ff"]
    patch_len = hparams["patch_len"]
    n_epochs = hparams["n_epochs"]
    patience = hparams["patience"]
    lr = hparams["lr"]
    batch_size = hparams["batch_size"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible. Este script requiere GPU.\nVerifica tu instalacion de PyTorch con soporte CUDA.")
    device = torch.device("cuda:0")
    print(f"\nUsando dispositivo: {device}  ({torch.cuda.get_device_name(0)})")
    out_csv_path = os.path.join(cli_args.results_dir, out_csv_name)
    resultados = []
    tiempo_inicio_total = time.time()
    ventana_idx = 0
    print("\n" + "=" * 65)
    if modo_seleccionado == "calendario":
        print(f"  Granularidad : {etiq_gran}")
        print(f"  Entrenamiento: {n_entren} periodo(s) por ventana")
        print(f"  Modo         : {modo_ventana}")
    else:
        print(f"  Horizonte={horizon}d | Step={step}d | Entrenamiento={anos_entren} anios")
        print(f"  Ventanas     : {len(ventanas)}")
    print("=" * 65)
    for v in ventanas:
        ventana_idx += 1
        if modo_seleccionado == "calendario":
            t_ini_ef = v["train_ini_efectivo"]
            t_fin = v["train_fin"]
            p_ini = v["pred_ini"]
            p_fin = v["pred_fin"]
            etiq = v["etiqueta"]
            mask_tr = (dates_all >= t_ini_ef) & (dates_all <= t_fin)
            mask_pr = (dates_all >= p_ini) & (dates_all <= p_fin)
            idx_tr = np.where(mask_tr)[0]
            idx_pr = np.where(mask_pr)[0]
            n_pr_obs = len(idx_pr)
            print(f"\n[{ventana_idx}/{len(ventanas)}]  Entrena: {t_ini_ef.date()} -> {t_fin.date()}" 
                  f"   Predice: {p_ini.date()} -> {p_fin.date()}  [{etiq}]")
        else:
            idx_tr = np.arange(v["idx_train_ini"], v["idx_train_fin"])
            idx_pr = np.arange(v["idx_test_ini"], v["idx_test_fin"])
            n_pr_obs = len(idx_pr)
            etiq = v["periodo_label"]
            print(f"\n[{ventana_idx}/{len(ventanas)}]  Entrena: {v['fecha_train_ini'].date()} -> " 
                  f"{v['fecha_train_fin'].date()}   Predice: {v['fecha_test_ini'].date()} -> " 
                  f"{v['fecha_test_fin'].date()}  [{etiq}]")
        n_tr_obs = len(idx_tr)
        if n_tr_obs < seq_len + label_len + 5:
            print(f"  Solo {n_tr_obs} obs en entrenamiento. Saltando.")
            continue
        if n_pr_obs < 1:
            print(f"  Sin datos reales en periodo de prediccion. Saltando.")
            continue
        scaler = StandardScaler()
        data_tr_raw = data_values[idx_tr]
        scaler.fit(data_tr_raw)
        data_tr_sc = scaler.transform(data_tr_raw)
        dates_tr = dates_all[idx_tr]
        min_seg = seq_len + label_len + n_pr_obs
        n_val = max(int(n_tr_obs * 0.15), min_seg)
        n_train = n_tr_obs - n_val
        if n_train < min_seg:
            print(f"  Sin espacio para split train/val. Validando sobre train.")
            ds_train = RollingDataset(data_tr_sc, seq_len, label_len, n_pr_obs, dates=dates_tr)
            ds_val = None
        else:
            ds_train = RollingDataset(data_tr_sc[:n_train], seq_len, label_len, n_pr_obs, dates=dates_tr[:n_train])
            ds_val = RollingDataset(data_tr_sc[n_train:], seq_len, label_len, n_pr_obs, dates=dates_tr[n_train:])
        if len(ds_train) < 1:
            print(f"  Dataset de entrenamiento sin muestras. Saltando.")
            continue
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
        if ds_val is not None and len(ds_val) >= 1:
            dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
        else:
            dl_val = DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=False)
        args_model = build_dfgcn_args(seq_len=seq_len, label_len=label_len, pred_len=n_pr_obs, enc_in=N, features=features, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, patch_len=patch_len, k=min(2, max(1, N - 1)), dropout=0.1, activation="sigmoid", use_norm=1, batch_size=batch_size)
        model = Model(args_model).float().to(device)
        t0 = time.time()
        model = train_window(model=model, train_loader=dl_train, val_loader=dl_val, device=device, pred_len=n_pr_obs, features=features, n_epochs=n_epochs, patience=patience, learning_rate=lr)
        print(f"  Entrenamiento: {time.time()-t0:.1f}s")
        input_idx_ini = idx_pr[0] - seq_len
        if input_idx_ini < 0:
            print(f"  No hay {seq_len} observaciones previas. Saltando.")
            del model
            torch.cuda.empty_cache()
            continue
        input_raw = data_values[input_idx_ini: idx_pr[0]]
        input_sc = scaler.transform(input_raw)
        pred_norm = predict_window(model, input_sc, device, n_pr_obs, features, N)
        if pred_norm.ndim == 1:
            pred_norm = pred_norm.reshape(-1, 1)
        true_raw = data_values[idx_pr]
        true_norm = scaler.transform(true_raw)
        true_norm_obj = true_norm[:, -1:]
        true_real_obj = true_raw[:, -1:]
        dummy = np.zeros((n_pr_obs, N), dtype=np.float32)
        dummy[:, -1] = pred_norm[:, 0]
        pred_real_obj = scaler.inverse_transform(dummy)[:, -1:]
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n = metric(pred_norm, true_norm_obj)
        mae_r, mse_r, rmse_r, mape_r, mspe_r, rse_r = metric(pred_real_obj, true_real_obj)
        print(f"  MAE={mae_r:.2f} COP  |  RMSE={rmse_r:.2f} COP  |  MAPE={mape_r*100:.2f}%")
        resultados.append({"periodo_predicho": etiq, "mae_normalizado": round(float(mae_n), 6), "mse_normalizado": round(float(mse_n), 6), "rmse_normalizado": round(float(rmse_n), 6), "mape_normalizado": round(float(mape_n), 6), "mspe_normalizado": round(float(mspe_n), 6), "rse_normalizado": round(float(rse_n), 6), "mae_escala_pesos": round(float(mae_r), 4), "mse_escala_pesos": round(float(mse_r), 4), "rmse_escala_pesos": round(float(rmse_r), 4), "mape_escala_pesos": round(float(mape_r), 6), "mspe_escala_pesos": round(float(mspe_r), 6), "rse_escala_pesos": round(float(rse_r), 6)})
        pd.DataFrame(resultados).to_csv(out_csv_path, index=False)
        del model
        torch.cuda.empty_cache()
        t_total = time.time() - tiempo_inicio_total
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res.to_csv(out_csv_path, index=False)
        cols_mostrar = [("mae_escala_pesos", "MAE (pesos COP)"), ("rmse_escala_pesos", "RMSE (pesos COP)"), ("mape_escala_pesos", "MAPE"), ("mae_normalizado", "MAE normalizado"), ("mse_normalizado", "MSE normalizado")]
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
        print("  No se completo ninguna ventana. Revisa los parametros.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()