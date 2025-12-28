from dash import dash_table, html
import polars as pl


def retorna_tabla(df: pl.DataFrame) -> html.Div:
    '''DEVUELVE EL COMPONENTE CON LA TABLA DE LA APP'''
    # Formateamos el precio con separador de miles y 2 decimales
    df = df.with_columns([
        pl.col("price").round(2).alias("price")
    ])
    df = df.with_columns([
        pl.col("price").map_elements(lambda x: f"{x:,.2f}", return_dtype=pl.String)
    ])

    # Convertimos a dict para Dash
    data = df.to_dicts()

    # Estilos condicionales para sombrear según valor
    conditional_styles = []
    colores = {
        "SUBE": "#2ecc71",   # Verde
        "BAJA": "#e74c3c",   # Rojo
        "IGUAL": "#3498db"   # Azul
    }

    columnas_direccion = [
        "Dir-Real", "XgBoost", "LightGbm", "PyTorch", "Mayoria"
    ]

    for col in columnas_direccion:
        for valor, color in colores.items():
            conditional_styles.append({
                "if": {"column_id": col, "filter_query": f"{{{col}}} = '{valor}'"},
                "backgroundColor": color,
                "color": "white"
            })

    # Estilo fijo para columnas 'date' y 'price'
    conditional_styles.extend([
        {
            "if": {"column_id": "date"},
            "backgroundColor": "#2c3e50",
            "color": "white"
        },
        {
            "if": {"column_id": "price"},
            "backgroundColor": "#2c3e50",
            "color": "white"
        }
    ])
    style_cell_conditional = [
        {"if": {"column_id": "date"}, "width": "4.9vw"},
        {"if": {"column_id": "price"}, "width": "4.9vw"},
        {"if": {"column_id": "Dir-Real"}, "width": "5.5vw"},
        {"if": {"column_id": "XgBoost"}, "width": "5vw"},
        {"if": {"column_id": "LightGbm"}, "width": "5vw"},
        {"if": {"column_id": "PyTorch"}, "width": "5vw"},
        {"if": {"column_id": "Mayoria"}, "width": "5vw"},
    ]
    style_header_conditional = [
        {"if": {"column_id": "LightGbm"}, "backgroundColor": "#ff8c00", "color": "white"},
        {"if": {"column_id": "XgBoost"}, "backgroundColor": "#4169e1", "color": "white"},
        {"if": {"column_id": "PyTorch"}, "backgroundColor": "#2e8b57", "color": "white"},
        {"if": {"column_id": "Mayoria"}, "backgroundColor": "#ab63fa", "color": "white"},
    ]

    return html.Div(
        children=dash_table.DataTable(
            data=data,
            columns=[{"name": col, "id": col} for col in df.columns],
            style_table={
                # "height": "600px",
                "width": "99%",
                "height": "39vh",
                "overflowY": "auto",
                "overflowX": "auto"
            },
            style_cell={
                "textAlign": "center",
                "padding": "6px",
                "fontFamily": "Arial",
                "fontSize": "14px",
                "whiteSpace": "normal"
            },
            style_header={
                "backgroundColor": "#1e1e1e",
                "color": "white",
                "fontWeight": "bold",
                "position": "sticky",
                "top": 0,
                "zIndex": 1
            },
            style_data_conditional=conditional_styles,
            style_header_conditional=style_header_conditional,
            style_cell_conditional=style_cell_conditional,
            fixed_rows={"headers": True},
            page_action="none"
        ),
        style={
            "width": "99%",
            "height": "95%",
            "padding": "3px"
        }
    )
