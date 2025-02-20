import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Cargar el modelo y el escalador
model = load_model('metero\modelos\lstm_temp_max_model.h5')
scaler = joblib.load('metero\scalers\matola_lstm_scaler_temp_max.pkl')

# Función para preparar los datos de entrada y hacer predicciones
def predict_next_temp_max(input_data):
    input_data = np.array(input_data).reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    X_input = np.array([scaled_input[-3:]])
    prediction = model.predict(X_input)
    next_temp_max = scaler.inverse_transform(prediction)
    return next_temp_max[0][0]

# Ejemplo de uso
input_data = [23.3, 17.9, 21.0]  # Temperatura máxima de los tres días anteriores
next_temp_max = predict_next_temp_max(input_data)
print(f'Temperatura máxima predicha para el día siguiente: {next_temp_max}')
