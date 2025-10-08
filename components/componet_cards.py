from dash import html
import polars as pl
from classes.modelo_backtesting import ModeloBacktesting

colores = {
    'LightGbm': '#ff8c00',
    'XgBoost': '#4169e1',
    'PyTorch': '#2e8b57',
    'Votación por Mayoría': '#ab63fa',
}


def retorna_cards(modelos_backtesting: list[ModeloBacktesting], df_ayer: pl.DataFrame) -> tuple:
    '''RETORNA LOS CUATRO COMPONENTES PARA LAS TARJETAS DEL DASHBOARD'''
    for modelo in modelos_backtesting:
        modelo.evaluar()

    ayer_cards = df_ayer['date'][0]
    cierre_ayer = df_ayer['price'][0]
    columnas = ['prediccion_xgboost', 'prediccion_lightgbm', 'prediccion_pytorch', 'prediccion_mayoria']
    predicciones_cierre_hoy = []
    for columna in columnas:
        if df_ayer[columna][0] == -1:
            predicciones_cierre_hoy.append("BAJA")
        elif df_ayer[columna][0] == 1:
            predicciones_cierre_hoy.append("SUBE")
        else:
            predicciones_cierre_hoy.append("IGUAL")

    nombres = [m.nombre_modelo for m in modelos_backtesting]
    ganancias_netas = [m.resumen["Ganancia Neta"] for m in modelos_backtesting]
    dir_accuracys = [m.resumen["Direction Accuracy"] for m in modelos_backtesting]

    cards = []
    for i in range(4):
        nombre = nombres[i]
        color_modelo = colores.get(nombre, "#ffffff")
        accuracy = f"{dir_accuracys[i] * 100:.2f}%"
        ganancia = ganancias_netas[i]
        prediccion = predicciones_cierre_hoy[i]

        if ganancia > 0:
            ganancia_texto = html.Small(f"+{ganancia:.2f}% ▲", style={"color": "#2ecc71"})
            ganancia_vs = html.Small("Vs Capital Inicial", style={"color": "#2ecc71"})
        else:
            ganancia_texto = html.Small(f"{ganancia:.2f}% ▼", style={"color": "#e74c3c"})
            ganancia_vs = html.Small("Vs Capital Inicial", style={"color": "#e74c3c"})

        if prediccion == "SUBE":
            prediccion_layout = html.Span("SUBE", style={"color": "#2ecc71", "fontSize": "14px", "fontWeight": "bold", "marginLeft": "8px"})
        else:
            prediccion_layout = html.Span("BAJA", style={"color": "#e74c3c", "fontSize": "14px", "fontWeight": "bold", "marginLeft": "8px"})

        card = html.Div([
            html.Div(nombre, style={
                "color": color_modelo,
                "fontWeight": "bold",
                "fontSize": "24px",
                "marginBottom": "2vh"
            }),
            html.Div([
                html.Div("✅ Tasa de Acierto", style={
                    "color": "#cccccc",
                    "fontSize": "14px",
                    "fontWeight": "bold",
                }),
                html.Div(accuracy, style={
                    "color": "white",
                    "fontSize": "36px",
                    "fontWeight": "bold",
                    "marginBottom": "2vh"
                }),
                ganancia_texto,
                html.Br(),
                ganancia_vs,
                html.Div("Próxima Predicción", style={
                    "color": "#cccccc",
                    "fontSize": "14px",
                    "fontWeight": "bold",
                    "marginTop": "2vh"
                }),
                html.Div(f"último cierre: {ayer_cards} | Price: {cierre_ayer:,.2f}$", style={
                    "color": "#cccccc",
                    "fontSize": "14px",
                    "fontWeight": "bold"
                }),
                html.Div([
                    html.Div("Predicción cierre hoy: ", style={
                        "color": "#cccccc",
                        "fontSize": "12px",
                        "fontWeight": "bold"
                    }),
                    prediccion_layout], style={"display": "flex"})
            ], style={"marginLeft": "1vw", "textAlign": "left"})
        ], style={
            "backgroundColor": "#1e1e1e",
            "padding": "12px",
            "borderRadius": "10px",
            "boxShadow": "0 0 6px rgba(0,0,0,0.3)",
            "height": "100%",
            "width": "100%"
        })

        cards.append(card)

    return tuple(cards)
