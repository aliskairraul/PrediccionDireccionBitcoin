from dash import dcc, html
import plotly.graph_objects as go
import polars as pl
from classes.modelo_backtesting import ModeloBacktesting


def grafico_evolucion_capital(modelos_backtesting: list):
    '''DEVUELVE EL GRAFICO SCATTER SOBRE COMO EVOLUCIONA EL CAPITAL INICIAL INVERTIDO (EN CADA MODELO) DESDE LA FECHA SELECCIONADA POR EL USUARIO'''
    colores = {
        'LightGbm': '#ff8c00',
        'XgBoost': '#4169e1',
        'PyTorch': '#2e8b57',
        'Votación por Mayoría': '#ab63fa',
        'No tradear': '#ffffff'
    }

    fig = go.Figure()
    for modelo in modelos_backtesting:
        df = modelo.df_resultado
        fig.add_trace(go.Scatter(
            x=df['date'].to_list(),
            y=df['capital_diario'].to_list(),
            mode='lines',
            name=modelo.nombre_modelo,
            line=dict(color=colores[modelo.nombre_modelo])
        ))
    fig.update_layout(
        title="Evolución del Capital Diario (Comienzo con 100$)",
        xaxis_title=None,
        yaxis_title="Capital (USD)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),  # Ajuste de márgenes
        showlegend=True
    )
    return fig


def retorna_grafico_scatter(modelos_backtesting_2: list[ModeloBacktesting]) -> dcc.Graph:
    for modelo in modelos_backtesting_2:
        modelo.evaluar()

    figure = grafico_evolucion_capital(modelos_backtesting=modelos_backtesting_2)

    return html.Div(
        children=dcc.Graph(
            figure=figure,
            config={"displayModeBar": False},
            style={"height": "98%", "width": "98%"}
        ),
        style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
            "height": "100%",
            "width": "100%"
        }
    )
