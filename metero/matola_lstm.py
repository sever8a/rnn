# Procesar los datos y entrenar el modelo.
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Cargar el dataset
data = pd.read_csv('metero\matola\matola_total.csv', parse_dates=['fecha']) # Convierte la característica fecha a tipo date.
data['fecha'] = pd.to_datetime(data['fecha'])
data = data.set_index('fecha')

# Crear una serie temporal solo con la temperatura máxima
temp_max = data[['temp_max']]

# Escalar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_temp_max = scaler.fit_transform(temp_max)

# Guardar el escalador para uso futuro
joblib.dump(scaler, 'metero\scalers\matola_lstm_scaler_temp_max.pkl')

# Crear las secuencias para el entrenamiento
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 3
X, y = create_sequences(scaled_temp_max, seq_length)

# Dividir los datos en entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Guardar el modelo
model.save('metero\modelos\lstm_temp_max_model.h5')

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Pérdida en el set de prueba: {loss}')
