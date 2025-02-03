import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocesamiento de los datos
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255

# Convertir las etiquetas a one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Construir el modelo de red neuronal fully connected
model_fc = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 clases (dígitos del 0 al 9)
])

# Compilar el modelo
model_fc.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Entrenar el modelo
model_fc.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Evaluar el modelo
test_loss, test_acc = model_fc.evaluate(x_test, y_test)
print(f'Precisión en el conjunto de prueba (red fully connected): {test_acc:.4f}')