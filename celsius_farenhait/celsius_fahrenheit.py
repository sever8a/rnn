import torch
import torch.nn as nn
import torch.optim as optim

# Definir la estructura de la red neuronal
class TemperatureConversionNN(nn.Module):
    def __init__(self):
        super(TemperatureConversionNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Capa oculta con 10 neuronas
        self.fc2 = nn.Linear(10, 1)  # Capa de salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear la red neuronal
model = TemperatureConversionNN()

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Datos de ejemplo (temperaturas en grados Celsius y Fahrenheit)
celsius = torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0], [50.0]], dtype=torch.float32)
fahrenheit = torch.tensor([[32.0], [50.0], [68.0], [86.0], [104.0], [122.0]], dtype=torch.float32)

# Entrenar la red neuronal
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(celsius)
    loss = criterion(outputs, fahrenheit)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Guardar el modelo
torch.save(model.state_dict(), 'temperature_conversion_model.pth')
print('Modelo guardado como temperature_conversion_model.pth')

# Probar la red neuronal
model.eval()
test_celsius = torch.tensor([[100.0]], dtype=torch.float32)
predicted_fahrenheit = model(test_celsius)
print(f'100°C en Fahrenheit es aproximadamente: {predicted_fahrenheit.item():.2f}°F')
