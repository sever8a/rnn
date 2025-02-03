import torch
import torch.nn as nn

# Definir la estructura de la red neuronal (debe ser la misma que la del modelo guardado)
class TemperatureConversionNN(nn.Module):
    def __init__(self):
        super(TemperatureConversionNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Capa oculta con 10 neuronas
        self.fc2 = nn.Linear(10, 1)  # Capa de salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear una instancia del modelo
model = TemperatureConversionNN()

# Cargar los pesos guardados en el modelo
model.load_state_dict(torch.load('temperature_conversion_model.pth'))
model.eval()  # Cambiar el modelo a modo de evaluación

# Probar el modelo cargado
test_celsius = torch.tensor([[50.0]], dtype=torch.float32)
predicted_fahrenheit = model(test_celsius)
print(f'50°C en Fahrenheit es aproximadamente: {predicted_fahrenheit.item():.2f}°F')
