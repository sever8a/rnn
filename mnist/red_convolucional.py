# Preprocesamiento de los datos para la CNN
x_train_cnn = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test_cnn = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Construir el modelo de red convolucional
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 clases (dígitos del 0 al 9)
])

# Compilar el modelo
model_cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Entrenar el modelo
model_cnn.fit(x_train_cnn, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Evaluar el modelo
test_loss, test_acc = model_cnn.evaluate(x_test_cnn, y_test)
print(f'Precisión en el conjunto de prueba (CNN): {test_acc:.4f}')