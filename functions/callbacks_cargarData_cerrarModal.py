from dash import Dash, dcc, html, Input, Output, State, callback, ctx
import torch
import torch.nn as nn
import polars as pl
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

from pathlib import Path
from functools import partial


class BitcoinDirectionNet(nn.Module):
    """Red neuronal optimizada para predecir dirección del precio de Bitcoin."""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_prob=0.3):
        super(BitcoinDirectionNet, self).__init__()

        layers = []
        prev_dim = input_dim

        # Construcción dinámica de capas
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # BatchNorm para mejor estabilidad
                nn.LeakyReLU(0.1),  # LeakyReLU para evitar neuronas muertas
                nn.Dropout(dropout_prob)
            ])
            prev_dim = hidden_dim

        # Capa final de clasificación
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Sigmoid para probabilidades [0,1]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# TRANSFORMA RSULTADO DE PREDICCIONES A -1 ó 1 SEGUN CONDICIONES
def sign_transform(arr: np.ndarray) -> np.ndarray:
    ''' Transforma 0 a 1 y -1 a 0 en un arreglo Numpy'''
    return np.where(arr > 0, 1, np.where(arr < 0, -1, 0))


# TRANSFORMA RSULTADO DE PREDICCIONES A -1 ó 1 SEGUN CONDICIONES
def sign_transform_binario(arr: np.ndarray) -> np.ndarray:
    ''' Tranforma menores de 0.5 a 0 y mayores de 0.5 a 1 en un arreglo Numpy'''
    return np.where(arr > 0.5, 1, np.where(arr < 0.5, -1, 0))


# AÑADE FEATURES A LA DATA PARA AYUDAR A PREDICION DE MODELOS
def add_features(df: pl.DataFrame, columnas: list) -> pl.DataFrame:
    """_summary_
        Agrega Features, como medias moviles y desviaciones standar en varios periodos de fechas, asi como datos calendario y otros para enriquecer la
        posibilidad de que los modelos de encuentren Patrones en la Data

    Args:
        df (pl.DataFrame): Data Original
        columnas (list): Lista de Features Originales, que despues de sufrir las tranformaciones el modelo dio mejores resultados

    Returns:
        pl.DataFrame: DataFrame con todas las Features nuevas añadidas
    """
    # Lags para features de mercado
    for column in columnas:
        if column in ['date', 'price']:
            continue
        df = df.with_columns(pl.col(column).shift(1).alias(f'{column}_lag_1'))

    # **CALCULAR _change_1d
    for column in columnas:
        if column in ['date', 'total_volume']:
            continue
        df = df.with_columns(((pl.col(column) - pl.col(column).shift(1)) / pl.col(column).shift(1)).alias(f"{column}_change_1d"))

    # Ratios intermercado
    for column in columnas:
        if column in ['date', 'price', 'total_volume']:
            continue
        df = df.with_columns((pl.col("price") / pl.col(column)).alias(f"btc_{column}_ratio"))

    df = df.with_columns([
        pl.when(pl.col("price_change_1d") > 0)
          .then(1)
          .when(pl.col("price_change_1d") < 0)
          .then(-1)
          .otherwise(0)
          .alias("price_direction_1d"),
    ])
    # Features temporales básicas
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.weekday().alias("weekday"),
    ])
    # Lags de precio Variados
    lags = [15, 30, 45, 60, 75, 90]
    for lag in lags:
        df = df.with_columns([
            # Cambios en lags propuestos
            pl.col("price").shift(lag).alias(f"btc_lag_{lag}"),
        ])
    # Rolling statistics de precio
    windows = [3, 7, 14, 21, 30]
    for window in windows:
        df = df.with_columns([
            pl.col("price").rolling_std(window).alias(f"btc_std_{window}"),
        ])
    windows = [30, 60, 90]
    for window in windows:
        df = df.with_columns([
            # Medias de los Instrumentos en Ciertas Ventanas
            pl.col("price").rolling_mean(window).alias(f"price_ma_{window}"),
            # Momentum adicional basado en precio (no confundir con price_change_1d)
            ((pl.col("price") / pl.col(f"btc_lag_{window}")) - 1).alias(f"btc_momentum_{window}d"),
        ])
    # **TARGET: precio de mañana (SIEMPRE AL FINAL)**
    df = df.with_columns(pl.col("price").shift(-1).alias("price_tomorrow"))
    df = df.with_columns(
        pl.when((pl.col('price_tomorrow') - pl.col('price')) > 0)
        .then(1)
        .when((pl.col('price_tomorrow') - pl.col('price')) < 0)
        .then(-1)
        .otherwise(0)
        .alias('target_direction')
    )
    # Eliminar NaNs (ÚLTIMO PASO)
    max_offset = max(max(lags, default=0), max(windows, default=0), 1)
    return (df.slice(max_offset, df.shape[0] - max_offset))


def cargar_modelos() -> tuple:
    '''Carga los Modelos entrenados y el scaler con que fue entrenado el modelo de Pytorch '''
    # Rutas de Modelos
    carpeta_models = Path('models')
    ruta_model_xgboost = carpeta_models / 'xgboost_model_data_2025.pkl'
    ruta_model_lightgbm = carpeta_models / 'lightgbm_model_data_2025.pkl'

    ruta_state_dict_pytorch = carpeta_models / 'pytorch_model_data_2025_binario.pth'

    ruta_scaler_pytorch = carpeta_models / 'scaler_global_pytorch_binario.joblib'

    ruta_instancia_dict = carpeta_models / 'config_pytorch_2025_binario.json'

    # Cargando los Modelos Machinne Learning
    model_xgboost = joblib.load(ruta_model_xgboost)
    model_lightgbm = joblib.load(ruta_model_lightgbm)

    # Cargando Modelo Pytorch
    # 1.- scaler Con que se entrenó el modelo
    scaler_pytorch = joblib.load(ruta_scaler_pytorch)

    # 2.- Parametros de la Instancia de la Calse
    with open(ruta_instancia_dict, "r") as f:
        instancia_dict = json.load(f)

    # 3.- Enviar al dispositivo adecuado
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4.- Instanciando la Clase
    model_pytorch = BitcoinDirectionNet(**instancia_dict).to(device)
    model_pytorch.load_state_dict(torch.load(ruta_state_dict_pytorch))
    model_pytorch.eval()

    return model_xgboost, model_lightgbm, model_pytorch, scaler_pytorch


def inferencia_modelos(model_xgboost, X_input_xgboost, X_input_xgboost_ayer,
                       model_lightgbm, X_input_lightgbm, X_input_lightgbm_ayer,
                       model_pytorch, X_input_pytorch, X_input_pytorch_ayer):
    """_summary_
        Simplemente consiste en hacer las inferencias con las datas tranformadas identicamente que cuando se entrenaron los modelos

    Args:
        model_xgboost:         Modelo XgBoost
        X_input_xgboost:       Entrada ya transformada para inferencia de la data (Todos los dias menos ayer)
        X_input_xgboost_ayer:  Entrada ya transformada para inferencia de la data de ayer
        model_lightgbm:        Modelo LightGbm
        X_input_lightgbm:      Entrada ya transformada para inferencia de la data (Todos los dias menos ayer)
        X_input_lightgbm_ayer: Entrada ya transformada para inferencia de la data de ayer
        model_pytorch:         Modelo Pytorch
        X_input_pytorch:       Entrada ya transformada para inferencia de la data (Todos los dias menos ayer)
        X_input_pytorch_ayer:  Entrada ya transformada para inferencia de la data de ayer

    Returns:
        _tuple[numpy.arrays]:  Inferencias de obtenidoas al aplicar las prediciones de los modelos sobre las datas de entrada
    """
    # XGBOOST
    # Predicciones de todo el 2025
    y_pred_xgboost = model_xgboost.predict(X_input_xgboost)
    y_pred_xgboost = sign_transform(arr=y_pred_xgboost)

    # Predicción de ayer para hoy
    y_pred_xgboost_ayer = model_xgboost.predict(X_input_xgboost_ayer)
    y_pred_xgboost_ayer = sign_transform(arr=y_pred_xgboost_ayer)

    # LIGHTGBM
    # Predicciones de todo el 2025
    y_pred_lightgbm = model_lightgbm.predict(X_input_lightgbm)

    # Predicción de ayer para hoy
    y_pred_lightgbm_ayer = model_lightgbm.predict(X_input_lightgbm_ayer)

    # PYTORCH
    # Diagnosticando el device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # Transformado los Inputs a Tensores
    X_input_pytorch_t = torch.tensor(X_input_pytorch, dtype=torch.float32)
    X_input_pytorch_ayer_t = torch.tensor(X_input_pytorch_ayer, dtype=torch.float32)

    # Predicciones de todo el 2025
    with torch.no_grad():
        y_pred_pytorch_t = model_pytorch(X_input_pytorch_t.to(device))

    y_pred_pytorch = y_pred_pytorch_t.cpu().detach().numpy().flatten()
    y_pred_pytorch = sign_transform_binario(arr=y_pred_pytorch)
    # Predicción de ayer para hoy
    with torch.no_grad():
        y_pred_pytorch_ayer_t = model_pytorch(X_input_pytorch_ayer_t.to(device))

    y_pred_pytorch_ayer = y_pred_pytorch_ayer_t.cpu().detach().numpy().flatten()
    y_pred_pytorch_ayer = sign_transform_binario(arr=y_pred_pytorch_ayer)

    return y_pred_xgboost, y_pred_xgboost_ayer, y_pred_lightgbm, y_pred_lightgbm_ayer, y_pred_pytorch, y_pred_pytorch_ayer


def retorna_2025_y_ayer(df_ayer, y_pred_xgboost_ayer, y_pred_lightgbm_ayer, y_pred_pytorch_ayer,
                        df_2025, y_pred_xgboost, y_pred_lightgbm, y_pred_pytorch):
    ''' JUNTANDO LAS PREDICCIONES CON LA DATA DEL DATAFRAME ORIGINAL '''
    # ***********   AYER ***********************
    df_ayer = df_ayer.with_columns([
        pl.Series('prediccion_xgboost', y_pred_xgboost_ayer),
        pl.Series('prediccion_lightgbm', y_pred_lightgbm_ayer),
        pl.Series('prediccion_pytorch', y_pred_pytorch_ayer)
    ])

    df_ayer = df_ayer.with_columns(
        pl.when((pl.col('prediccion_xgboost') + pl.col('prediccion_lightgbm') + pl.col('prediccion_pytorch')) > 0)
        .then(1).otherwise(-1).alias('prediccion_mayoria')
    )

    # ***********   2025 ***********************
    df_2025 = df_2025.with_columns(
        pl.when(pl.col('price') < pl.col('price_tomorrow'))
        .then(1)
        .when(pl.col('price') > pl.col('price_tomorrow'))
        .then(-1)
        .otherwise(0)
        .alias('real_direction')
    )
    df_2025 = df_2025.with_columns([
        pl.Series('prediccion_xgboost', y_pred_xgboost),
        pl.Series('prediccion_lightgbm', y_pred_lightgbm),
        pl.Series('prediccion_pytorch', y_pred_pytorch)
    ])

    df_2025 = df_2025.with_columns(
        pl.when((pl.col('prediccion_xgboost') + pl.col('prediccion_lightgbm') + pl.col('prediccion_pytorch')) > 0)
        .then(1).otherwise(-1).alias('prediccion_mayoria')
    )

    df_2025 = df_2025.with_columns(pl.Series('solo_compro_al_inicio', np.ones(len(y_pred_pytorch))))

    df_2025 = df_2025.with_columns([
        pl.when(pl.col('real_direction') == pl.col('prediccion_xgboost')).then(1).otherwise(0).alias('Acierto_XgBoost'),
        pl.when(pl.col('real_direction') == pl.col('prediccion_lightgbm')).then(1).otherwise(0).alias('Acierto_LightGbm'),
        pl.when(pl.col('real_direction') == pl.col('prediccion_pytorch')).then(1).otherwise(0).alias('Acierto_PyTorch'),
        pl.when(pl.col('real_direction') == pl.col('prediccion_mayoria')).then(1).otherwise(0).alias('Acierto_Mayoria'),
        pl.when(pl.col('real_direction') == pl.col('solo_compro_al_inicio')).then(1).otherwise(0).alias('Acierto_solo_compro')
    ])

    return df_ayer, df_2025


def retorna_df_tabla(df: pl.DataFrame) -> pl.DataFrame:
    '''PREPARANDO UN DF PARA LA TABLA DE LA APP.  CAMBIANDO 1 POR "SUBE" Y -1 POR "BAJA"'''
    def retorna_direccion(numero: int) -> str:
        if numero == 1:
            return "SUBE"
        if numero == -1:
            return "BAJA"
        return "IGUAL"

    df_tabla = df.select(['date', 'price', 'real_direction', 'prediccion_xgboost', 'prediccion_lightgbm', 'prediccion_pytorch', 'prediccion_mayoria'])

    df_tabla = df_tabla.with_columns([
        pl.col('real_direction').map_elements(retorna_direccion, return_dtype=pl.String).alias("Dir-Real"),
        pl.col('prediccion_xgboost').map_elements(retorna_direccion, return_dtype=pl.String).alias("XgBoost"),
        pl.col('prediccion_lightgbm').map_elements(retorna_direccion, return_dtype=pl.String).alias("LightGbm"),
        pl.col('prediccion_pytorch').map_elements(retorna_direccion, return_dtype=pl.String).alias("PyTorch"),
        pl.col('prediccion_mayoria').map_elements(retorna_direccion, return_dtype=pl.String).alias("Mayoria"),
    ])
    df_tabla = df_tabla.select(['date', 'price', 'Dir-Real', 'XgBoost', 'LightGbm', 'PyTorch', 'Mayoria'])
    return df_tabla


# 🪟 Callback para cerrar el modal cuando la data esté lista
@callback(
    Output("modal-loading", "is_open"),
    Input("store-df-2025", "data"),
    prevent_initial_call=True
)
def cerrar_modal(data):
    '''CERRANDO MODAL CUANDO SE CARGA LA DATA AL INICIO'''
    return False if data else True


# 🧠 Callback para cargar la data
@callback(
    Output("store-df-2025", "data"),
    Output("store-df-ayer", "data"),
    Output("store-df-tabla", "data"),
    Input("interval-loader", "n_intervals"),
    prevent_initial_call=True
)
def cargar_data(n):
    ''' CARGA Y TRANSFOMA LA DATA NECESARIA AL INICIO DE LA APP. SE APOYA DE TODAS LA FUNCIONES DE ESTE ARCHIVO PARA NO SER TAN LARGA UNA SOLA FUNCION'''
    url = "https://raw.githubusercontent.com/aliskairraul/ActualizaData/main/db/db.parquet"
    carpeta_data = Path('db')
    ruta_db = carpeta_data / 'db.parquet'
    ruta_db_backup = carpeta_data / 'backup' / 'db.parquet'

    # Llamando a la Función que carga los Modelos
    model_xgboost, model_lightgbm, model_pytorch, scaler_pytorch = cargar_modelos()

    # Columnas Con que empieza cada modelo
    columns_xgboost = ['date', 'price', 'total_volume']
    columns_lightgbm = ['date', 'price', 'rate_US10Y', 'price_gold']
    columns_pytorch = ['date', 'price', 'total_volume', 'stock_index_sp500', 'price_gold', 'rate_US10Y', 'stock_index_ni225']

    # Creando el respaldo
    df = pl.read_parquet(ruta_db)
    df.write_parquet(ruta_db_backup)

    # Jalando la data que esta en github
    response = requests.get(url)
    if response.status_code == 200:
        with open(str(ruta_db.resolve()), "wb") as f:
            f.write(response.content)

    # Cargando el dataFrame actualizado y elaborando los Dataframes para cada Modelo
    df = pl.read_parquet(ruta_db)

    ultima_fecha = df['date'].max()

    # Llamado la Funcion `add_features` para la data de cada modelo
    df_xgboost = df.select(columns_xgboost).pipe(add_features, columns_xgboost).sort('date')
    df_lightgbm = df.select(columns_lightgbm).pipe(add_features, columns_lightgbm).sort('date')
    df_pytorch = df.select(columns_pytorch).pipe(add_features, columns_pytorch).sort('date')

    # Ahora Debo dividir entre la Data de Ayer (de la cual todavia no conozco el futuro) y el resto de la Data del 2025
    df_xgboost_ayer = df_xgboost.filter(df_xgboost['date'] == ultima_fecha)
    df_xgboost = df_xgboost.filter((df_xgboost['date'] < ultima_fecha) & (df_xgboost['date'] >= date(2025, 1, 1)))

    df_lightgbm_ayer = df_lightgbm.filter(df_lightgbm['date'] == ultima_fecha)
    df_lightgbm = df_lightgbm.filter((df_lightgbm['date'] < ultima_fecha) & (df_lightgbm['date'] >= date(2025, 1, 1)))

    df_pytorch_ayer = df_pytorch.filter(df_pytorch['date'] == ultima_fecha)
    df_pytorch = df_pytorch.filter((df_pytorch['date'] < ultima_fecha) & (df_pytorch['date'] >= date(2025, 1, 1)))

    # Features usadas por cada Modelo
    features_xgboost = [col for col in df_xgboost.columns if col not in ["date", 'target_direction', 'price_tomorrow']]
    features_lightgbm = [col for col in df_lightgbm.columns if col not in ["date", 'target_direction', 'price_tomorrow']]
    features_pytorch = [col for col in df_pytorch.columns if col not in ["date", 'target_direction', 'price_tomorrow']]

    # Inputs de Cada Modelo (Total Data)
    X_input_xgboost = df_xgboost.sort('date').select(features_xgboost).to_numpy()
    X_input_xgboost_ayer = df_xgboost_ayer.select(features_xgboost).to_numpy()

    X_input_lightgbm = df_lightgbm.sort('date').select(features_lightgbm).to_numpy()
    X_input_lightgbm_ayer = df_lightgbm_ayer.select(features_lightgbm).to_numpy()

    X_input_pytorch = scaler_pytorch.transform(df_pytorch.sort('date').select(features_pytorch).to_numpy())
    X_input_pytorch_ayer = scaler_pytorch.transform(df_pytorch_ayer.select(features_pytorch).to_numpy())

    # Llamado a Funcion `inferencia_modelos`
    y_pred_xgboost, y_pred_xgboost_ayer, y_pred_lightgbm, y_pred_lightgbm_ayer, y_pred_pytorch, y_pred_pytorch_ayer = inferencia_modelos(
        model_xgboost, X_input_xgboost, X_input_xgboost_ayer,
        model_lightgbm, X_input_lightgbm, X_input_lightgbm_ayer,
        model_pytorch, X_input_pytorch, X_input_pytorch_ayer
    )

    # Necesito solo `date` y `price` de cualquiera de los Dataframes de ayer para df_ayer y de los dataframes completos para df_2025
    df_ayer = df_pytorch_ayer.select(['date', 'price'])
    df_2025 = df_pytorch.select(['date', 'price', 'price_tomorrow'])

    df_ayer, df_2025 = retorna_2025_y_ayer(df_ayer, y_pred_xgboost_ayer, y_pred_lightgbm_ayer, y_pred_pytorch_ayer,
                                           df_2025, y_pred_xgboost, y_pred_lightgbm, y_pred_pytorch)

    df_tabla = retorna_df_tabla(df=df_2025)

    return df_2025.to_dicts(), df_ayer.to_dicts(), df_tabla.to_dicts()
