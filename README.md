# Forecasting COP/USD con DFGCN

## Descripción del Proyecto

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Este repositorio implementa el modelo **DFGCN (Dual Frequency Graph Convolutional Network)** para la predicción de la tasa de cambio **Peso Colombiano / Dólar Estadounidense (COP/USD)**.

DFGCN es un modelo de aprendizaje profundo diseñado para predicción multivariada de series de tiempo. Combina dos componentes de Redes Neuronales de Grafos (GCN) que operan en frecuencias duales:

1. **GNN_time**: Modela las dependencias temporales entre parches (patches) de la serie de tiempo, construyendo un grafo dinámico basado en correlación de Pearson entre segmentos temporales.
2. **GNN_variate**: Modela las dependencias entre variables (canales), también mediante correlación de Pearson, capturando relaciones entre distintos indicadores financieros.

Ambas ramas se fusionan al final para producir una predicción robusta que captura tanto la estructura temporal como las interdependencias entre variables.

---

## ¿Por qué Python?

Python es la herramienta estándar para el desarrollo de modelos de aprendizaje profundo y análisis de datos por varias razones:

- **PyTorch**: La librería de deep learning más utilizada en investigación, con soporte para GPU, diferenciación automática y módulos reutilizables (`nn.Module`).
- **PyTorch Geometric**: Extiende PyTorch con soporte nativo para Redes Neuronales de Grafos (GNN/GCN), lo que permite construir y entrenar grafos dinámicos de forma eficiente.
- **Ecosistema científico**: `NumPy`, `Pandas`, `Scikit-learn` y `Matplotlib` forman un ecosistema maduro para manipulación de datos, preprocesamiento, evaluación y visualización.
- **Facilidad de prototipado**: Python permite iterar rápidamente sobre ideas y arquitecturas de modelos con código limpio y legible.
- **Comunidad y recursos**: La mayoría de los papers de investigación en series de tiempo financieras publican su código en Python.

---

## Estructura del Repositorio

```
Forecasting-COP-USD/
│
├── modelos/                     # Arquitecturas de modelos
│   ├── DFGCN.py                 # Modelo principal DFGCN
│   ├── RevIN.py                 # Normalización Reversible de Instancia
│   └── RandomWalk.py            # Modelo de Caminata Aleatoria (benchmark)
│
├── layers/                      # Capas de la red neuronal
│   ├── Embed.py                 # Capas de embedding (posicional, temporal, token)
│   ├── GNN_time.py              # GCN para dependencias temporales
│   ├── GNN_variate.py           # GCN para dependencias entre variables
│   └── Transformer_encoder.py  # Encoder Transformer con atención global
│
├── experiments/                 # Pipelines de entrenamiento y evaluación
│   ├── exp_basic.py             # Clase base de experimentos
│   ├── exp_term_forecasting.py  # Pipeline principal (train/vali/test)
│   └── exp_long_term_forecasting_partial.py  # Entrenamiento parcial (zero-shot)
│
├── data_provider/               # Carga y preparación de datos
│   ├── data_factory.py          # Selector de dataset según argumentos
│   └── data_loader.py           # Clases Dataset para distintos formatos
│
├── utils/                       # Utilidades
│   ├── metrics.py               # Métricas de evaluación (MAE, MSE, RMSE, MAPE, RSE)
│   ├── tools.py                 # Herramientas: EarlyStopping, ajuste de LR, visualización
│   ├── timefeatures.py          # Características temporales (hora, día, mes, etc.)
│   └── masking.py               # Máscaras causales para atención
│
├── datos/                       # 📁 Carpeta para sus datos (ponga aquí sus archivos CSV)
│
├── run.py                       # Script principal de ejecución
├── requirements.txt             # Dependencias de Python
└── README.md                    # Este archivo
```

---

## Formato de los Datos

### Estructura del archivo CSV

Coloque sus datos en la carpeta `datos/`. El archivo debe seguir este formato:

```csv
date,variable1,variable2,...,variableN,COP_USD
2020-01-01,valor1,valor2,...,valorN,3750.50
2020-01-02,valor1,valor2,...,valorN,3762.30
...
```

**Requisitos obligatorios:**
| Campo | Descripción |
|-------|-------------|
| `date` | Columna de fecha (primera columna). Formatos aceptados: `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS` |
| Columnas numéricas | Las demás columnas deben ser numéricas (float o int). No se admiten valores nulos (se recomienda interpolación previa) |
| Variable objetivo | Debe ser la **última columna** del CSV, o indicarla con el parámetro `--target` |

**Ejemplo para tasa COP/USD diaria:**
```csv
date,PIB_col,tasa_interes,inflacion,petroleo_brent,COP_USD
2020-01-02,0.023,4.25,3.80,68.50,3730.45
2020-01-03,0.023,4.25,3.80,67.80,3745.20
2020-01-06,0.023,4.25,3.82,68.10,3758.90
```

**Ejemplo univariado (solo la tasa):**
```csv
date,COP_USD
2020-01-02,3730.45
2020-01-03,3745.20
2020-01-06,3758.90
```

### División de datos

El modelo divide automáticamente los datos en:
- **Entrenamiento**: 70% del total
- **Validación**: 10% del total
- **Prueba/Test**: 20% del total

> ⚠️ Se recomienda un mínimo de **500 observaciones** para un entrenamiento adecuado. Para datos diarios, esto equivale a ~2 años de historia.

### Frecuencias disponibles (`--freq`)
| Código | Descripción |
|--------|-------------|
| `d`    | Diaria (recomendada para COP/USD) |
| `b`    | Días hábiles |
| `h`    | Horaria |
| `t`    | Por minuto |
| `w`    | Semanal |
| `m`    | Mensual |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/oadiazg/Forecasting-COP-USD.git
cd Forecasting-COP-USD
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota sobre PyTorch Geometric**: Si tiene problemas con `torch-geometric`, instale las dependencias manualmente siguiendo las instrucciones en https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Uso del Modelo DFGCN

---

### Entendiendo `--enc_in` y `--features`

El parámetro `--features` controla qué columnas se usan como entrada y cuál se predice:

| Modo | Descripción |
|------|-------------|
| `S` | **Univariado** — usa sólo la columna `--target` como entrada y la predice |
| `M` | **Multivariado** — usa todas las columnas como entrada y predice todas |
| `MS` | **Multi→Uni** — usa todas las columnas como entrada, predice sólo `--target` |

El parámetro `--enc_in` **debe ser igual** al número de columnas de datos (excluyendo `date`) que el modelo recibe como entrada:
- En modo `S`: siempre `--enc_in 1`
- En modos `M` y `MS`: `--enc_in` = total de columnas numéricas en el CSV

**Tabla de referencia:**

| Columnas en CSV (sin `date`) | Modo | `--enc_in` | `--dec_in` | `--c_out` | `--k` |
|------------------------------|------|-----------|-----------|----------|-------|
| 1 (`COP_USD`) | `S` | 1 | 1 | 1 | 1 |
| 3 (`var1, var2, COP_USD`) | `M` | 3 | 3 | 3 | 2 |
| 3 (`var1, var2, COP_USD`) | `MS` | 3 | 3 | 1 | 2 |
| 7 (`var1…var6, COP_USD`) | `M` | 7 | 7 | 7 | 2 |

> ⚠️ **Restricción importante sobre `--k`**: el valor de `k` debe ser **menor** que `enc_in`.
> Para `enc_in=1` use siempre `--k 1`. Para `enc_in=3` use `--k 2`. El modelo detecta automáticamente cuando no hay vecinos suficientes y usa auto-conexiones.

---

### Modo 1: Entrenamiento

#### Entrenamiento con 1 variable (univariado)

Para una serie de tiempo con **una sola columna** (por ejemplo, sólo `date` y `COP_USD`):

```
python run.py --is_training 1 --model_id COP_USD_experimento1 --model DFGCN --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv --features S --target COP_USD --freq d --seq_len 96 --label_len 48 --pred_len 30 --enc_in 1 --dec_in 1 --c_out 1 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 1 --use_norm 1 --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj type7 --dropout 0.1 --des entrenamiento_inicial
```

> **Nota**: `--k 1` es obligatorio cuando `enc_in=1`. El grafo de variables no puede tener más vecinos que `enc_in - 1`.

#### Entrenamiento con N variables (multivariado)

Reemplace los valores entre `<` `>` con los de su dataset:

```
python run.py --is_training 1 --model_id COP_USD_experimento1 --model DFGCN --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv --features M --target COP_USD --freq d --seq_len 96 --label_len 48 --pred_len 30 --enc_in <N> --dec_in <N> --c_out <N> --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 2 --use_norm 1 --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj type7 --dropout 0.1 --des entrenamiento_inicial
```

Donde `<N>` es el número total de columnas numéricas en su CSV (sin contar `date`).

**Ejemplo con 7 variables:**
```
python run.py --is_training 1 --model_id COP_USD_7vars --model DFGCN --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv --features M --target COP_USD --freq d --seq_len 96 --label_len 48 --pred_len 30 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 2 --use_norm 1 --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj type7 --dropout 0.1 --des entrenamiento_inicial
```

#### Restricciones importantes

| Restricción | Detalle |
|-------------|---------|
| **Observaciones mínimas** | Se recomienda ≥ 500 filas. Con `seq_len=96`, `pred_len=30` el modelo necesita al menos `seq_len + pred_len = 126` filas para tener al menos 1 muestra de test. |
| **`k` < `enc_in`** | Siempre `k ≤ enc_in - 1`. Si `enc_in=1`, use `k=1` (se aplica auto-conexión). |
| **`patch_len` divide `seq_len`** | `(seq_len - patch_len) % patch_len == 0`. Con `seq_len=96` y `patch_len=8`: `(96-8)/8 = 11` exacto → ✓. |
| **`seq_len > patch_len`** | El tamaño del parche debe ser menor que la ventana de entrada. |

---

### Modo 2: Validación

La validación se ejecuta automáticamente al final del entrenamiento. Para ejecutar **sólo la evaluación** sobre un modelo ya entrenado:

```
python run.py --is_training 0 --model_id COP_USD_experimento1 --model DFGCN --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv --features S --target COP_USD --freq d --seq_len 96 --label_len 48 --pred_len 30 --enc_in 1 --dec_in 1 --c_out 1 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 1 --use_norm 1 --des entrenamiento_inicial
```

> Los resultados se guardan en `./results/<setting>/` y `./test_results/<setting>/`. Las métricas se registran en `result_long_term_forecast.txt`. También se exportan automáticamente `predictions_vs_actuals.csv` y `metrics_summary.csv`.

---

### Modo 3: Simulación (Random Walk como benchmark)

El modelo `RandomWalk.py` permite generar simulaciones de Monte Carlo para comparar con DFGCN:

```bash
# Simulación básica con 30 días de predicción
python modelos/RandomWalk.py \
  --data_path datos/tasa_cop_usd.csv \
  --target_col COP_USD \
  --pred_len 30 \
  --num_simulations 1000

# Con más simulaciones y guardando el gráfico
python modelos/RandomWalk.py \
  --data_path datos/tasa_cop_usd.csv \
  --target_col COP_USD \
  --pred_len 60 \
  --num_simulations 5000 \
  --output_plot resultados/random_walk_60d.png

# Sin graficar (solo guardar CSV con predicciones)
python modelos/RandomWalk.py \
  --data_path datos/tasa_cop_usd.csv \
  --pred_len 30 \
  --no_plot
```

---

## Parámetros del Modelo DFGCN

### Parámetros de Datos
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--data` | str | `custom` | Tipo de dataset. Use `custom` para datos propios |
| `--root_path` | str | `./datos/` | Carpeta donde están los datos |
| `--data_path` | str | — | Nombre del archivo CSV |
| `--features` | str | `M` | `M`=multivariada, `S`=univariada, `MS`=multi→uni |
| `--target` | str | `OT` | Columna objetivo (para `S` o `MS`) |
| `--freq` | str | `d` | Frecuencia: `d`=diaria, `h`=horaria, `b`=hábil, etc. |

### Parámetros de la Ventana de Tiempo
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--seq_len` | int | `96` | Ventana de entrada: cuántos pasos históricos usa el modelo |
| `--label_len` | int | `48` | Longitud del token inicial del decoder |
| `--pred_len` | int | `30` | Horizonte de predicción: cuántos pasos a futuro predice |

### Parámetros de Arquitectura
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--enc_in` | int | `7` | Número de variables de entrada (debe coincidir con columnas del CSV) |
| `--d_model` | int | `128` | Dimensión interna del modelo (mayor = más capacidad, más lento) |
| `--n_heads` | int | `1` | Cabezas de atención multi-head |
| `--e_layers` | int | `1` | Número de capas del encoder |
| `--d_ff` | int | `128` | Dimensión de la capa feed-forward |
| `--patch_len` | int | `8` | Tamaño del parche temporal (divide la serie en segmentos de este tamaño) |
| `--k` | int | `2` | Número de vecinos k-NN para construir el grafo de correlación |
| `--use_norm` | int | `1` | Normalización RevIN: `1`=activa (recomendado), `0`=desactiva |
| `--dropout` | float | `0.1` | Dropout para regularización (0.0 a 1.0) |
| `--activation` | str | `sigmoid` | Función de activación de las GCN: `sigmoid` o `relu` |

### Parámetros de Entrenamiento
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--is_training` | int | `1` | `1`=entrenar, `0`=solo evaluar |
| `--train_epochs` | int | `10` | Número máximo de épocas |
| `--batch_size` | int | `32` | Tamaño del batch (reducir si hay problemas de memoria) |
| `--learning_rate` | float | `0.0001` | Tasa de aprendizaje del optimizador Adam |
| `--patience` | int | `3` | Épocas de paciencia para early stopping |
| `--lradj` | str | `type1` | Estrategia de ajuste de LR (ver tabla abajo) |
| `--use_amp` | flag | `False` | Precisión mixta (recomendado con GPU moderna) |

### Estrategias de Ajuste de Learning Rate (`--lradj`)
| Valor | Descripción |
|-------|-------------|
| `type1` | Reduce LR a la mitad cada época |
| `type2` | Reducción escalonada predefinida |
| `type3` | LR constante las primeras 3 épocas, luego decaimiento exponencial |
| `constant` | LR constante durante todo el entrenamiento |
| `cosine` | Decaimiento coseno suave |
| `type7` | Usa OneCycleLR (recomendado para DFGCN) |

### Parámetros de GPU
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--use_gpu` | bool | `True` | Usar GPU si está disponible |
| `--gpu` | int | `0` | Índice de la GPU (0 = primera) |
| `--use_multi_gpu` | flag | `False` | Entrenar en múltiples GPUs |
| `--devices` | str | `0,1,2,3` | IDs de GPUs para modo multi-GPU |

### Modos de Experimento (`--exp_name`)
| Valor | Descripción |
|-------|-------------|
| `None` | Entrenamiento y prueba estándar |
| `partial_train` | Entrenamiento con subconjunto de variables, prueba con todas |

---

## Métricas de Evaluación

Al finalizar el entrenamiento/prueba, se reportan las siguientes métricas:

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error Absoluto Medio — promedio de los errores absolutos |
| **MSE** | Error Cuadrático Medio — penaliza errores grandes |
| **RMSE** | Raíz del MSE — en las mismas unidades que la variable |
| **MAPE** | Error Porcentual Absoluto Medio — error relativo en % |
| **RSE** | Error Cuadrático Relativo — normalizado por la varianza de los datos reales |

---

## Uso del Repositorio en Visual Studio Code

### Configuración inicial en VS Code

1. **Abrir el repositorio**
   - Abra VS Code
   - `File > Open Folder...` → seleccione la carpeta `Forecasting-COP-USD`

2. **Instalar la extensión Python**
   - Vaya a `Extensions` (Ctrl+Shift+X)
   - Busque e instale **"Python"** de Microsoft

3. **Seleccionar el intérprete de Python**
   - Presione `Ctrl+Shift+P` → busque **"Python: Select Interpreter"**
   - Seleccione el entorno virtual `venv` que creó (aparecerá como `./venv/Scripts/python.exe` en Windows o `./venv/bin/python` en Linux/Mac)

4. **Abrir una terminal integrada**
   - `Terminal > New Terminal` o `` Ctrl+` ``
   - Active el entorno virtual si no está activo:
     ```bash
     # Windows
     venv\Scripts\activate
     # Linux/Mac
     source venv/bin/activate
     ```

5. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

6. **Colocar sus datos**
   - Copie su archivo CSV en la carpeta `datos/`
   - Asegúrese de que tenga columna `date` y las demás columnas numéricas

### Ejecutar el modelo desde VS Code

**Opción A: Desde la terminal integrada**

```bash
# Entrenamiento completo
python run.py --is_training 1 --model_id COP_USD --model DFGCN \
  --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv \
  --features M --target COP_USD --freq d \
  --seq_len 96 --label_len 48 --pred_len 30 \
  --enc_in 7 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 \
  --patch_len 8 --k 2 --use_norm 1 \
  --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --patience 5

# Random Walk
python modelos/RandomWalk.py --data_path datos/tasa_cop_usd.csv \
  --target_col COP_USD --pred_len 30
```

**Opción B: Crear una configuración de lanzamiento (launch.json)**

1. Vaya a `Run > Add Configuration...`
2. Seleccione **Python > Python File**
3. Reemplace el contenido de `.vscode/launch.json` con:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "DFGCN - Entrenamiento",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run.py",
            "args": [
                "--is_training", "1",
                "--model_id", "COP_USD",
                "--model", "DFGCN",
                "--data", "custom",
                "--root_path", "./datos/",
                "--data_path", "tasa_cop_usd.csv",
                "--features", "M",
                "--target", "COP_USD",
                "--freq", "d",
                "--seq_len", "96",
                "--label_len", "48",
                "--pred_len", "30",
                "--enc_in", "7",
                "--d_model", "128",
                "--n_heads", "1",
                "--e_layers", "1",
                "--d_ff", "128",
                "--patch_len", "8",
                "--k", "2",
                "--use_norm", "1",
                "--train_epochs", "20",
                "--batch_size", "32",
                "--learning_rate", "0.0001",
                "--patience", "5",
                "--lradj", "type7",
                "--dropout", "0.1",
                "--des", "experimento_1"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "DFGCN - Solo Evaluación",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run.py",
            "args": [
                "--is_training", "0",
                "--model_id", "COP_USD",
                "--model", "DFGCN",
                "--data", "custom",
                "--root_path", "./datos/",
                "--data_path", "tasa_cop_usd.csv",
                "--features", "M",
                "--target", "COP_USD",
                "--freq", "d",
                "--seq_len", "96",
                "--label_len", "48",
                "--pred_len", "30",
                "--enc_in", "7",
                "--d_model", "128",
                "--n_heads", "1",
                "--e_layers", "1",
                "--d_ff", "128",
                "--patch_len", "8",
                "--k", "2",
                "--use_norm", "1",
                "--des", "experimento_1"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Random Walk - Simulación",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/modelos/RandomWalk.py",
            "args": [
                "--data_path", "datos/tasa_cop_usd.csv",
                "--target_col", "COP_USD",
                "--pred_len", "30",
                "--num_simulations", "1000"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

4. Para ejecutar, presione `F5` o vaya a `Run > Start Debugging`
5. Seleccione la configuración deseada en el menú desplegable

### Visualizar resultados

Los gráficos de predicción se guardan automáticamente en:
- `./test_results/<nombre_experimento>/` — imágenes `.pdf` cada 200 batches
- `./results/<nombre_experimento>/` — matrices numpy con predicciones y valores reales
- `result_long_term_forecast.txt` — registro de métricas por experimento

---

## Referencia

El modelo DFGCN está basado en el repositorio original:
- **Repositorio**: https://github.com/junjieyeys/DFGCN
- **Artículo**: "DFGCN: Dual Frequency Graph Convolutional Network for Multivariate Time Series Forecasting"

---

## Usar el Modelo Entrenado para Predicciones Futuras

### Concepto: ¿Qué predice el modelo?

Cuando configura `--pred_len 30`, el modelo predice los **próximos 30 pasos** después de la última ventana de `seq_len` días del conjunto de test — no necesariamente después de la última fecha en su CSV.

Durante la evaluación con `--is_training 0`, el modelo evalúa sobre el split de test predefinido (último 20 % del CSV). Para obtener predicciones **verdaderamente futuras** (más allá de la última fila de su CSV), use el script `predict_future.py`.

---

### Flujo completo: del entrenamiento a la predicción futura

#### Paso 1 — Entrenar el modelo

```
python run.py --is_training 1 --model_id COP_USD_experimento1 --model DFGCN --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv --features S --target COP_USD --freq d --seq_len 96 --label_len 48 --pred_len 30 --enc_in 1 --dec_in 1 --c_out 1 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 1 --use_norm 1 --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj type7 --dropout 0.1 --des entrenamiento_inicial
```

Esto guarda:
- `./checkpoints/<setting>/checkpoint.pth` — pesos del modelo
- `./checkpoints/<setting>/scaler.pkl` — escalador StandardScaler (para invertir la normalización)

#### Paso 2 — Evaluar en el conjunto de test (split predefinido)

```
python run.py --is_training 0 --model_id COP_USD_experimento1 --model DFGCN --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv --features S --target COP_USD --freq d --seq_len 96 --label_len 48 --pred_len 30 --enc_in 1 --dec_in 1 --c_out 1 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 1 --use_norm 1 --des entrenamiento_inicial
```

Esto genera en `./results/<setting>/`:
- `predictions_vs_actuals.csv` — predicciones vs. valores reales en escala normalizada y real
- `metrics_summary.csv` — todas las métricas en escala normalizada y real

#### Paso 3 — Predecir valores futuros (más allá del último dato)

```
python predict_future.py --model_id COP_USD_experimento1 --data_path datos/tasa_cop_usd.csv --seq_len 96 --pred_len 30 --enc_in 1 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 1 --use_norm 1 --des entrenamiento_inicial --freq d
```

Esto guarda el resultado en `./resultados_futuro/COP_USD_experimento1_future_predictions.csv` con columnas `date` y `predicted_COP_USD`.

---

### Limitación actual

El método `test()` sólo evalúa sobre el split de test predefinido (las últimas 20 % filas del CSV). Para predecir valores después de la última fila de datos, use `predict_future.py` que:
1. Carga los últimos `seq_len` registros del CSV como ventana de entrada
2. Aplica el mismo escalador que se usó durante el entrenamiento
3. Ejecuta un único forward pass del modelo
4. Invierte la normalización para obtener valores en COP/USD real
5. Genera fechas futuras usando `pandas.bdate_range`

---

### Parámetros de `predict_future.py`

| Parámetro | Descripción |
|-----------|-------------|
| `--model_id` | Identificador del experimento (igual al usado en el entrenamiento) |
| `--data_path` | Ruta al CSV (e.g., `datos/tasa_cop_usd.csv`) |
| `--seq_len` | Ventana de entrada (igual al entrenamiento) |
| `--pred_len` | Número de pasos futuros a predecir |
| `--enc_in` | Número de variables de entrada (igual al entrenamiento) |
| `--k` | Vecinos k-NN (igual al entrenamiento; `k=1` para univariado) |
| `--des` | Descripción del experimento (igual al entrenamiento) |

Todos los parámetros de arquitectura (`--d_model`, `--n_heads`, `--e_layers`, `--d_ff`, `--patch_len`, `--use_norm`) deben coincidir exactamente con los usados durante el entrenamiento.
