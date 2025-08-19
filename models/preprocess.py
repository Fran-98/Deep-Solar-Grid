import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Dataset():
    def __init__(self, train_path, test_path, n_pasos, seq2seq = False):
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        df_train['Hora de inicio'] = pd.to_datetime(df_train['Hora de inicio'])
        df_train.set_index('Hora de inicio', inplace=True)
        df_test['Hora de inicio'] = pd.to_datetime(df_test['Hora de inicio'])
        df_test.set_index('Hora de inicio', inplace=True)

        df_train['hora_sin'] = np.sin(2 * np.pi * df_train.index.hour / 23.0)
        df_train['hora_cos'] = np.cos(2 * np.pi * df_train.index.hour / 23.0)
        df_test['hora_sin'] = np.sin(2 * np.pi * df_test.index.hour / 23.0)
        df_test['hora_cos'] = np.cos(2 * np.pi * df_test.index.hour / 23.0)

        # df_train = df_train.clip(lower=0)
        # df_test = df_test.clip(lower=0)

        # df_train = df_train.diff().fillna(0)
        df_train.drop('Energia_MPPT_Total(kWh)', axis=1, inplace=True)
        # df_test = df_test.diff().fillna(0)
        df_test.drop('Energia_MPPT_Total(kWh)', axis=1, inplace=True)

        # --- 2. PREPARACIÓN DE DATOS PARA MODELO MULTIVARIADO ---
        # <-- MEJORA: Seleccionamos todas las variables que usaremos como entrada (features).
        # La variable a predecir ('Potencia activa(kW)') debe ir PRIMERO en la lista.
        self.features = [
            'Potencia activa(kW)',
            'Corriente_FV_Total(A)',
            'Tension_FV_Promedio(V)',
            'Potencia_Entrada_FV_Total(kW)',
            'Eficiencia del inversor(%)',
            'MO', # Mes
            # 'ALLSKY_SFC_SW_DWN', # Irradiancia
            # 'T2M', # Temperatura a 2 m
            # 'WS10M', # Velocidad del viento a 10m
            # 'RH2M', # Humedad relativa a 2m
            'hora_sin', # seno de la hora (no es necesario mencionar)
            'hora_cos', # cos de la hora (no es necesario mencionar)
        ]
        self.n_features = len(self.features) # <-- Guardamos el número de features

        # Seleccionamos el subconjunto de datos con nuestras features
        train_data_multi = df_train[self.features].values
        test_data_multi = df_test[self.features].values

        # Inicializar el escalador
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Ajustar y transformar los datos de entrenamiento
        train_scaled = self.scaler.fit_transform(train_data_multi)
        # Solo transformar los datos de prueba (usando el scaler ya ajustado)
        self.test_scaled = self.scaler.transform(test_data_multi)

        
        self.n_pasos = n_pasos

        # Creamos las ventanas usando la nueva función
        if not seq2seq:
            self.X_train, self.y_train = self.crear_ventanas_multivariado(train_scaled, n_pasos)
            self.X_test, self.y_test = self.crear_ventanas_multivariado(self.test_scaled, n_pasos)
        else:
            self.X_train, self.y_train = self.crear_ventanas_seq2seq(train_scaled, n_pasos, n_pasos_salida=10)
            self.X_test, self.y_test = self.crear_ventanas_seq2seq(self.test_scaled, n_pasos, n_pasos_salida=10)

        self.POTENCIA_MAXIMA = df_train['Potencia activa(kW)'].max() * 1.1

        self.df_test = df_test
        self.df_train = df_train
        self.df_test_scaled = pd.DataFrame(self.test_scaled,
                                           index=df_test.index,
                                           columns=self.features)
        
    def crear_ventanas_multivariado(self, dataset, n_pasos_entrada):
            X, y = [], []
            for i in range(len(dataset) - n_pasos_entrada):
                # La ventana de entrada (X) contiene TODAS las features
                ventana_x = dataset[i : i + n_pasos_entrada]
                X.append(ventana_x)
                # La salida (y) es solo el valor de la PRIMERA columna (nuestra variable objetivo)
                valor_y = dataset[i + n_pasos_entrada, 0]
                y.append(valor_y)
            return np.array(X), np.array(y)
    
    def crear_ventanas_seq2seq(self, dataset, n_pasos_entrada, n_pasos_salida):
        X, y = [], []
        # Aseguramos que haya suficientes datos para la última ventana completa
        for i in range(len(dataset) - n_pasos_entrada - n_pasos_salida + 1):
            ventana_x = dataset[i : i + n_pasos_entrada]
            X.append(ventana_x)

            # La 'y' ahora es una secuencia de los siguientes 'n_pasos_salida'
            # de la primera columna (la potencia)
            ventana_y = dataset[i + n_pasos_entrada : i + n_pasos_entrada + n_pasos_salida, 0]
            y.append(ventana_y)

        return np.array(X), np.array(y)