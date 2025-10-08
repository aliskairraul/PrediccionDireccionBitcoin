from dash import dcc, html
import plotly.graph_objects as go
import polars as pl
from classes.modelo_backtesting import ModeloBacktesting


def grafico_accuracy(modelos_backtesting):
    ''' DEVUELVE EL GRAFICO DE BARRAS DEL PORCENTAJE DE ACIERTOS DE CADA MODELO'''
    colores = {
        'LightGbm': '#ff8c00',
        'XgBoost': '#4169e1',
        'PyTorch': '#2e8b57',
        'Votación por Mayoría': '#ab63fa'
    }

    nombres = [m.nombre_modelo for m in modelos_backtesting]
    accuracy = [m.resumen["Direction Accuracy"] * 100 for m in modelos_backtesting]

    menor = min(accuracy)
    mayor = max(accuracy)
    margen = 5

    fig = go.Figure()
    for nombre, valor in zip(nombres, accuracy):
        fig.add_trace(go.Bar(
            name=nombre,
            x=[nombre],  # mantener para separación
            y=[valor],
            marker_color=colores[nombre],
            showlegend=True  # mantener para la leyenda
        ))
    fig.update_layout(
        title="Accuracy Direccional (% de Aciertos de Cada Modelo)",
        yaxis=dict(range=[menor - margen, mayor + margen]),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),  # 👈 oculta nombres
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_layout(
        legend=dict(
            orientation="h",         # horizontal
            yanchor="bottom",        # anclaje inferior
            y=0.98,                  # justo encima del gráfico
            xanchor="center",
            x=0.5
        )
    )
    return fig


def retorna_grafico_accuracy(modelos_backtesting: list[ModeloBacktesting]) -> dcc.Graph:
    for modelo in modelos_backtesting:
        modelo.evaluar()

    figure = grafico_accuracy(modelos_backtesting=modelos_backtesting)

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
