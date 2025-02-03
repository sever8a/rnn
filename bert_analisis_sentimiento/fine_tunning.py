# pip install transformers datasets torch

# preparación de los datos
import pandas as pd
from datasets import Dataset

# Cargar el conjunto de datos
data = pd.read_csv('ruta_a_tus_datos.csv')

# Convertir a un formato de dataset adecuado para Hugging Face
dataset = Dataset.from_pandas(data)


# Tokenización de los datos

from transformers import BertTokenizer

# Cargar el tokenizador preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizar los datos
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Cargar modelo preentrenado.
from transformers import BertForSequenceClassification

# Cargar el modelo preentrenado
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Configuración del entrenamiento del modelo

from transformers import TrainingArguments, Trainer

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
)


# Entrenamiento del modelo

trainer.train()


## Evaluación del modelo

eval_results = trainer.evaluate()
print(f"Resultados de evaluación: {eval_results}")


## Hacer predicciones

# Ejemplo de nuevo texto
new_text = ["Este es un gran producto.", "No me gustó para nada."]

# Tokenizar el nuevo texto
inputs = tokenizer(new_text, return_tensors="pt", padding=True, truncation=True)

# Hacer predicciones
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(f"Predicciones: {predictions}")
