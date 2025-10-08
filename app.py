from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from functions.callbacks_cargarData_cerrarModal import cargar_data, cerrar_modal
from functions.callbacks_component_dropdown import actualizar_componentes, marcar_datos_listos
from datetime import datetime, timezone, date

ahora_utc = datetime.now(timezone.utc)
hoy_formateado = ahora_utc.strftime("%B %d, %Y %H:%M (UTC)")
# 🪟 Modal de carga
modal_loading = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Cargando Data")),
        dbc.ModalBody("Por favor espere mientras se carga la información..."),
    ],
    id="modal-loading",
    is_open=True,
    backdrop="static",
    keyboard=False,
    centered=True
)

# ⏱️ Intervalo para disparar la carga
interval_loader = dcc.Interval(id="interval-loader", interval=500, n_intervals=0, max_intervals=1)

# 📦 Store para guardar Data individualizada de cada sesion de Usuario
store_cargo_data_correctamente = dcc.Store(id="cargo-data-externa", data={"cargo_correctamente": False})
store_df_2025 = dcc.Store(id="store-df-2025")
store_df_ayer = dcc.Store(id="store-df-ayer")
store_df_tabla = dcc.Store(id="store-df-tabla")
store_ready = dcc.Store(id="store-ready")

# 🚀 App Dash
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Prediccion Direccion Bitcoin"
)
server = app.server

app.layout = html.Div([
    modal_loading,
    interval_loader,
    store_df_2025,
    store_df_ayer,
    store_df_tabla,
    store_ready,

    # 🔷 Zona 1: Encabezado ajustado
    html.Div([
        html.Div([
            html.H1("Predicción Dirección Bitcoin", className="titulo"),
            html.Div(hoy_formateado, style={"fontWeight": "bold", "color": "#ffffff"}, id="fecha-actual", className="fecha")
        ], className="fila-encabezado"),

        html.Div([
            dcc.Dropdown(
                id="dropdown-modelo",
                options=[
                    {"label": "Desde 01/01/2025", "value": "all"},
                    {"label": "Últimos 30 días", "value": "30"},
                    {"label": "Últimos 60 días", "value": "60"},
                    {"label": "Últimos 90 días", "value": "90"},
                    {"label": "Últimos 120 días", "value": "120"},
                    {"label": "Últimos 150 días", "value": "150"},
                    {"label": "Últimos 180 días", "value": "180"},
                ],
                value="all",  # Valor por defecto
                className="dropdown",
                clearable=False
            )
        ], className="fila-dropdown")
    ], className="zona-1"),

    # 🔶 Zona 2: Cuerpo principal
    html.Div([
        # 🟫 Bloque Izquierdo (40%)
        html.Div([
            # Sub-Bloque 1 (80%): 4 tarjetas
            html.Div([
                html.Div(id="card-1", className="tarjeta"),
                html.Div(id="card-2", className="tarjeta"),
                html.Div(id="card-3", className="tarjeta"),
                html.Div(id="card-4", className="tarjeta")
            ], className="subbloque-izq-1"),

            # Sub-Bloque 2 (20%): 1 tarjeta
            html.Div([
                html.Div(id="card-5", className="tarjeta")
            ], className="subbloque-izq-2")
        ], className="bloque-izquierdo"),

        # 🟦 Bloque Derecho (60%)
        html.Div([
            # Sub-Bloque 1 (35%): Scatter
            html.Div([
                html.Div(id="grafico-scatter", className="grafico")
            ], className="subbloque-der-1"),

            # Sub-Bloque 2 (65%): 2 gráficos horizontales
            html.Div([
                html.Div([
                    html.Div(id="grafico-izquierda", className="grafico")
                ], className="grafico-container"),
                html.Div([
                    html.Div(id="grafico-derecha", className="grafico")
                ], className="grafico-container")
            ], className="subbloque-der-2")
        ], className="bloque-derecho")
    ], className="zona-2")
])


# 🏁 Ejecutar la app
# if __name__ == "__main__":
# app.run_server(debug=False, host="0.0.0.0", port=7860)
# app.run(debug=True)
