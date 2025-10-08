import polars as pl


class ModeloBacktesting:
    '''MODELO DE EVALUACION BACKTESTING QUE SE REPITE PARA LOS 4 MODELOS, ENCAPSULADO EN UNA CLASE PARA DISMINUIR CODIGO Y FACILIDAD DE USO'''
    def __init__(self, df: pl.DataFrame, columnas_modelo: list, nombre_modelo: str, transaction_cost_pct: float = 0.002):
        self.df_original = df.select(columnas_modelo)
        self.nombre_modelo = nombre_modelo
        self.transaction_cost_pct = transaction_cost_pct
        self.df_resultado = None
        self.resumen = None

    def evaluar(self, monto_inicial: float = 100):
        df = self.df_original.clone()
        df.columns = ['date', 'price', 'price_tomorrow', 'real_direction', 'prediccion', 'Acierto']

        prices = df['price'].to_list()
        prices_tomorrow = df['price_tomorrow'].to_list()
        aciertos = df['Acierto'].to_list()

        montos = []
        cumulative_return = []
        monto = monto_inicial

        for i in range(len(prices)):
            montos.append(monto)
            costo_operacion = monto * self.transaction_cost_pct
            cambio_porcentual = abs((prices_tomorrow[i] - prices[i]) / prices[i])
            ganancia_perdida = monto * cambio_porcentual
            monto += ganancia_perdida if aciertos[i] == 1 else -ganancia_perdida
            monto -= costo_operacion
            cumulative_return.append(monto - monto_inicial)

        direction_accuracy = sum(aciertos) / len(aciertos)
        ganancia_neta = cumulative_return[-1]
        proyeccion_anualizada = ganancia_neta / len(cumulative_return) * 365

        df = df.with_columns([
            pl.Series('ganancia_neta', cumulative_return, dtype=pl.Float64),
            pl.Series('capital_diario', montos, dtype=pl.Float64)
        ])

        self.df_resultado = df
        self.resumen = {
            "Modelo": self.nombre_modelo,
            "Direction Accuracy": round(direction_accuracy, 4),
            "Ganancia Neta": round(ganancia_neta, 2),
            "Proyección Anualizada": round(proyeccion_anualizada, 2),
            "N_Days": len(cumulative_return)
        }

        return self.df_resultado, self.resumen
