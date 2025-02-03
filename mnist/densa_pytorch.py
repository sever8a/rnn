import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Preprocesamiento de datos
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte las imágenes a tensores
    transforms.Normalize((0.5,), (0.5,))  # Normaliza los valores de píxeles
])

# Cargar el conjunto de datos MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Definir la red neuronal fully connected
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instanciar el modelo, la función de pérdida y el optimizador
model_fc = FullyConnectedNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fc.parameters(), lr=0.001)

# Entrenamiento
for epoch in range(5):  # Número de épocas
    model_fc.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_fc(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluación
model_fc.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_fc(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Precisión en el conjunto de prueba (red fully connected): {100 * correct / total:.2f}%')