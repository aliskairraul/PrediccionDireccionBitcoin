from dash import dash, Input, Output, State, callback, exceptions
import polars as pl
from datetime import date, timedelta, timezone
from classes.modelo_backtesting import ModeloBacktesting
from components.component_evolucion_capital import retorna_grafico_scatter
from components.component_accuracy_direccional import retorna_grafico_accuracy
from components.component_proyeccion_anualizada import retorna_proyeccion_anualizada
from components.component_tabla import retorna_tabla
from components.componet_cards import retorna_cards


@callback(
    Output("store-ready", "data"),
    Input("store-df-2025", "data"),
    Input("store-df-ayer", "data"),
)
def marcar_datos_listos(data_2025, data_ayer):
    '''FUNCION ESTRATEGIA PARA QUE NO SE INTENTE "actualizar_componentes" SI LA DATA NECESARIA NO SE HA CARGADO Y TRANSFORMADO '''
    return bool(data_2025) and bool(data_ayer)


@callback(
    Output("card-1", "children"),
    Output("card-2", "children"),
    Output("card-3", "children"),
    Output("card-4", "children"),
    Output("tabla", "children"),
    Output("grafico-scatter", "children"),
    Output("grafico-izquierda", "children"),
    Output("grafico-derecha", "children"),
    Input("dropdown-modelo", "value"),
    Input("store-ready", "data"),
    State("store-df-2025", "data"),
    State("store-df-ayer", "data"),
    State("store-df-tabla", "data"),
    prevent_initial_call=True
)
def actualizar_componentes(filtro, ready, data_2025, data_ayer, data_tabla):
    '''RECIBE LOS DATAFRAMES YA PREPARADOS Y LLAMA A LAS DISTINTAS FUNCIONES QUE ARMAN LOS COMPONENTES PARA RETORNARLOS'''
    if not ready:
        raise exceptions.PreventUpdate

    evaluacion_xgboost_columns = ['date', 'price', 'price_tomorrow', 'real_direction', 'prediccion_xgboost', 'Acierto_XgBoost']
    evaluacion_lightgbm_columns = ['date', 'price', 'price_tomorrow', 'real_direction', 'prediccion_lightgbm', 'Acierto_LightGbm']
    evaluacion_pytorch_columns = ['date', 'price', 'price_tomorrow', 'real_direction', 'prediccion_pytorch', 'Acierto_PyTorch']
    evaluacion_mayoria_columns = ['date', 'price', 'price_tomorrow', 'real_direction', 'prediccion_mayoria', 'Acierto_Mayoria']
    evaluacion_compra_inicial_columns = ['date', 'price', 'price_tomorrow', 'real_direction', 'solo_compro_al_inicio', 'Acierto_solo_compro']

    # 🧱 Convertir a DataFrame Polars
    df_2025 = pl.DataFrame(data_2025) if data_2025 else pl.DataFrame()
    df_ayer = pl.DataFrame(data_ayer) if data_ayer else pl.DataFrame()
    df_tabla = pl.DataFrame(data_tabla)

    # 🗓️ Asegurar que la columna 'date' sea de tipo date
    if "date" in df_2025.columns:
        df_2025 = df_2025.with_columns(pl.col("date").cast(pl.Date))
    if "date" in df_ayer.columns:
        df_ayer = df_ayer.with_columns(pl.col("date").cast(pl.Date))
    if "date" in df_tabla.columns:
        df_tabla = df_tabla.with_columns(pl.col("date").cast(pl.Date))

    if filtro == "all":
        df = df_2025.clone()

    # 📅 Filtrar según el dropdown
    if filtro != "all":
        dias = int(filtro)
        # fecha_limite = date.today(timezone.utc) - timedelta(days=dias)
        fecha_limite = df_2025['date'].max() - timedelta(days=dias)
        df = df_2025.filter(pl.col("date") >= fecha_limite)
        df_tabla = df_tabla.filter(pl.col("date") >= fecha_limite)
        # df_ayer = df_ayer.filter(pl.col("date") >= fecha_limite)

    evalua_xgboost = ModeloBacktesting(df=df, columnas_modelo=evaluacion_xgboost_columns, nombre_modelo='XgBoost')
    evalua_lightgbm = ModeloBacktesting(df=df, columnas_modelo=evaluacion_lightgbm_columns, nombre_modelo='LightGbm')
    evalua_pytorch = ModeloBacktesting(df=df, columnas_modelo=evaluacion_pytorch_columns, nombre_modelo='PyTorch')
    evalua_mayoria = ModeloBacktesting(df=df, columnas_modelo=evaluacion_mayoria_columns, nombre_modelo='Votación por Mayoría')
    evalua_compra_inicial = ModeloBacktesting(df=df, columnas_modelo=evaluacion_compra_inicial_columns, nombre_modelo='No tradear')
    modelos_backtesting = [evalua_xgboost, evalua_lightgbm, evalua_pytorch, evalua_mayoria]
    modelos_backtesting_2 = [evalua_xgboost, evalua_lightgbm, evalua_pytorch, evalua_mayoria, evalua_compra_inicial]

    component_grafico_scatter = retorna_grafico_scatter(modelos_backtesting_2=modelos_backtesting_2)
    component_grafico_accuracy = retorna_grafico_accuracy(modelos_backtesting=modelos_backtesting)
    component_grafico_proyeccion = retorna_proyeccion_anualizada(modelos_backtesting=modelos_backtesting)
    component_tabla = retorna_tabla(df=df_tabla)

    card_1, card_2, card_3, card_4 = retorna_cards(modelos_backtesting=modelos_backtesting, df_ayer=df_ayer)

    # 🧪 Por ahora devolvemos None a todos los outputs
    return card_1, card_2, card_3, card_4, component_tabla, component_grafico_scatter, component_grafico_accuracy, component_grafico_proyeccion
    # return None, None, None, None, None, None, None, None
