from dash import dcc, html
import plotly.graph_objects as go
import polars as pl
from classes.modelo_backtesting import ModeloBacktesting


def grafico_proyeccion_anualizada(modelos_backtesting: list):
    ''' DEVUELVE EL COMPONENTE CON EL GRAFICO DE PROYECCION ANUALIZADA'''
    colores = {
        'LightGbm': '#ff8c00',
        'XgBoost': '#4169e1',
        'PyTorch': '#2e8b57',
        'Votación por Mayoría': '#ab63fa',
        'No tradear': '#ffffff'
    }

    nombres = [m.nombre_modelo for m in modelos_backtesting]
    anualizada = [m.resumen["Proyección Anualizada"] for m in modelos_backtesting]

    fig = go.Figure()
    for nombre, valor in zip(nombres, anualizada):
        fig.add_trace(go.Bar(
            name=nombre,
            x=[nombre],
            y=[valor],
            marker_color=colores[nombre],
            showlegend=True
        ))
    fig.update_layout(
        # title="Proyección Anualizada (% de Rentabilidad)",
        title={
            "text": "Proyección Anualizada",
            "y": 0.99,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        template="plotly_dark",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_layout(
        legend=dict(
            orientation="h",         # horizontal
            yanchor="bottom",        # anclaje inferior
            y=0.97,                  # justo encima del gráfico
            xanchor="center",
            x=0.5
        )
    )
    return fig


def retorna_proyeccion_anualizada(modelos_backtesting: list[ModeloBacktesting]) -> dcc.Graph:
    for modelo in modelos_backtesting:
        modelo.evaluar()

    figure = grafico_proyeccion_anualizada(modelos_backtesting=modelos_backtesting)

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
